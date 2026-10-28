from __future__ import annotations

import os
import copy
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Callable

import torch
import torch.nn as nn

from .vggt_mot import (
    DEFAULT_VGGT_WEIGHTS,
    load_aggregator_weights,
    validate_vggt_images,
)
from .vggt_mot_recon_backbone import VGGTMoTReconBackbone


class VGGTMoEReconBackbone(VGGTMoTReconBackbone):
    """Hard-routed VGGT streams with configurable input/head sharing.

    ``high_noise_vggt`` (``va1``) can optionally use language, while
    ``low_noise_vggt`` (``va2``) is always language-conditioned. Patch
    embedding, camera/register prefixes and geometry heads can
    independently be shared or duplicated. RoPE remains shared. Defaults keep
    the historical module names and parameter registration order.

    Tensor flow for a mixed batch is::

        images [B,V,3,H,W]
          -> shared patch embedding [B*V,S,D]
          -> route subsets [Bh,V,S,D] / [Bl,V,S,D]
          -> optionally language-conditioned high / visual+language low
          -> route K/V and native 2048-wide reconstruction features
          -> K/V restored to original batch order [B,Hv,Sv,Dh]

    Only the selected route runs the expensive VGGT transformer blocks for a
    sample.  ``forward_mixed_with_reconstruction`` returns route-local head
    predictions so reconstruction targets can be indexed without scattering
    large depth tensors back to the full batch.
    """

    def __init__(self, *args, share_geometry_heads=True,
                 high_noise_use_language_condition=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_geometry_heads = bool(share_geometry_heads)
        self.high_noise_use_language_condition = bool(high_noise_use_language_condition)
        # Keep the legacy head names and registration order in shared mode.
        if not self.share_geometry_heads:
            self.low_camera_head = copy.deepcopy(self.camera_head)
            self.low_dense_head = copy.deepcopy(self.dense_head)

    def geometry_heads(self, route):
        if route not in ("high", "low"):
            raise ValueError(f"Unknown VGGT route: {route}")
        if route == "low" and not self.share_geometry_heads:
            return self.low_camera_head, self.low_dense_head
        return self.camera_head, self.dense_head

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Geometry decoders consume FP32 features, even after .to(bfloat16).
        for route in ("high", "low"):
            for head in self.geometry_heads(route):
                head.float()
        return self

    def _language(self, context, mask, *, dtype, device):
        if context is None or context.ndim != 3 or mask is None or mask.shape != context.shape[:2]:
            raise ValueError("Language context/mask must be [B,L,D] and [B,L].")
        valid = mask.to(device=device, dtype=torch.bool)
        # Mask before projection as NaN * 0 is still NaN.
        language = self.language_projector(
            context.to(device=device, dtype=dtype).masked_fill(~valid[..., None], 0))
        return language.masked_fill(~valid[..., None], 0), valid

    @property
    def high_noise_vggt(self):
        return self.va1

    @property
    def low_noise_vggt(self):
        return self.va2

    @staticmethod
    def _zero_anchor(result: Mapping[str, object], reference: torch.Tensor) -> torch.Tensor:
        """Keep all route parameters in the autograd graph for empty routes."""
        anchor = reference.reshape(-1)[:0].sum()
        for entry in result["kv_by_layer"].values():
            anchor = anchor + entry["k"].reshape(-1)[:0].sum() + entry["v"].reshape(-1)[:0].sum()
        for key in ("depth", "depth_conf", "pose_enc"):
            anchor = anchor + result[key].reshape(-1)[:0].sum()
        # The language projector is not connected by the visual
        # K/V path when its route is empty; connect it through its projected
        # language output when the caller supplies one.
        language_state = result.get("language_state")
        if isinstance(language_state, torch.Tensor):
            anchor = anchor + language_state.reshape(-1)[:0].sum()
        return anchor

    def _prepare_patch_tokens(self, images, aggregator):
        """[Br,V,3,H,W] -> [Br,V,Npatch,D], without camera/register tokens.

        Shared patch embedding runs once on B; independent embeddings run on
        their route subset Br. Token prefixes are attached separately so their
        sharing choice is independent of the patch embedding choice.
        """
        batch_size, num_views, channels, height, width = images.shape
        if num_views != len(self.input_camera_keys):
            raise ValueError("VGGT view count does not match configured cameras.")
        normalized = (images - aggregator._resnet_mean) / aggregator._resnet_std
        patch_dtype = next(aggregator.patch_embed.parameters()).dtype
        patch_tokens = aggregator.patch_embed(normalized.to(patch_dtype).reshape(
            batch_size * num_views, channels, height, width))
        if isinstance(patch_tokens, Mapping):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        frame_dtype = next(aggregator.frame_blocks[0].parameters()).dtype
        patch_tokens = patch_tokens.to(frame_dtype).reshape(batch_size, num_views, -1, patch_tokens.shape[-1])
        grid = (height // int(aggregator.patch_size), width // int(aggregator.patch_size))
        with torch.no_grad():
            sin, cos = aggregator.rope_embed(H=grid[0], W=grid[1])
        return patch_tokens, (sin.to(images.device, torch.float32), cos.to(images.device, torch.float32))

    def prepare_route_tokens(self, images, route, *, patch_tokens=None, frame_rope=None):
        """Attach the selected route's prefix: [Br,V,Npatch,D] -> [Br,V,N,D]."""
        aggregator = self.high_noise_vggt if route == "high" else self.low_noise_vggt
        if route not in ("high", "low"):
            raise ValueError(f"Unknown VGGT route: {route}")
        if patch_tokens is None:
            patch_tokens, frame_rope = self._prepare_patch_tokens(images, aggregator)
        b, v, n, d = patch_tokens.shape
        tokens = self._expand_input_tokens(aggregator, patch_tokens.reshape(b * v, n, d),
                                          batch_size=b, num_frames=v)
        return tokens, frame_rope

    def _prepare_shared_tokens(self, images):
        """Legacy helper for callers inspecting the high/shared visual stream."""
        return self.prepare_route_tokens(images, "high")

    def _run_visual_route(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
        *,
        aggregator: nn.Module,
        frame_rope,
        context: torch.Tensor | None,
        language_context_mask: torch.Tensor | None,
        use_language: bool,
        reconstruct: bool = True,
    ) -> dict[str, object]:
        """Run one route over ``tokens`` and decode its selected geometry heads.

        ``tokens`` is ``[Br,V,N,D]``.  Frame attention is always local to the
        selected Aggregator. When enabled, language is appended only to
        the active inter-frame sequence and is never exposed as ActionDiT K/V.
        """
        batch_size, num_views, num_tokens, embed_dim = tokens.shape
        language = language_valid = None
        if use_language:
            language, language_valid = self._language(
                context,
                language_context_mask,
                dtype=tokens.dtype,
                device=tokens.device,
            )

        route = "high" if aggregator is self.high_noise_vggt else "low"
        camera_head, dense_head = self.geometry_heads(route)
        features: list[torch.Tensor | None] = [None] * int(aggregator.depth)
        cache: dict[int, dict[str, object]] = {}
        selected = set(self.callback_layer_indices)
        # DenseHead needs its fixed feature pyramid even when action capture
        # uses a sparse layer mapping. CameraHead always reads the final layer.
        head_layers = set(getattr(dense_head, "intermediate_layer_idx", ()))
        head_layers.add(int(aggregator.depth) - 1)

        for layer_idx in range(int(aggregator.depth)):
            _, frame_output = aggregator._run_frame_block(
                tokens,
                batch_size,
                num_views,
                num_tokens,
                embed_dim,
                layer_idx,
                frame_rope,
            )
            tokens = frame_output
            block = aggregator.inter_frame_blocks[layer_idx]
            attention_type = aggregator.inter_frame_attention_types[layer_idx]
            active, restore = self._active_inter_frame_tokens(
                tokens,
                attention_type=attention_type,
                patch_token_start=int(aggregator.patch_token_start),
            )
            q, k, v = self._attention_qkv(block, active)

            if not use_language:
                mixed = self._sdpa(q, k, v)
                updated = self._post_inter_frame_attention(block, active, mixed)
            else:
                assert language is not None and language_valid is not None
                joined = torch.cat([active, language], dim=1)
                q_all, k_all, v_all = self._attention_qkv(block, joined)
                valid = torch.cat(
                    [
                        torch.ones(
                            (batch_size, active.shape[1]),
                            dtype=torch.bool,
                            device=active.device,
                        ),
                        language_valid,
                    ],
                    dim=1,
                )
                mixed_all = self._sdpa(q_all, k_all, v_all, valid)
                mixed = mixed_all[:, : active.shape[1]]
                mixed_language = mixed_all[:, active.shape[1] :]
                updated_all = self._post_inter_frame_attention(
                    block,
                    joined,
                    torch.cat([mixed, mixed_language], dim=1),
                )
                updated = updated_all[:, : active.shape[1]]
                language = updated_all[:, active.shape[1] :]
                language = language * language_valid[..., None].to(language.dtype)

            tokens = restore(updated).contiguous()
            if reconstruct and layer_idx in head_layers:
                features[layer_idx] = torch.cat([frame_output, tokens], dim=-1)
            if layer_idx in selected:
                cache[layer_idx] = {
                    "layer_idx": layer_idx,
                    "attention_type": attention_type,
                    # K/V contain visual tokens only.  Language conditions the
                    # low route indirectly through its updated visual tokens.
                    "k": k.to(torch.bfloat16 if self.fp32_storage_bf16_compute else images.dtype),
                    "v": v.to(torch.bfloat16 if self.fp32_storage_bf16_compute else images.dtype),
                }

        missing = selected.difference(cache)
        if missing:
            raise RuntimeError(f"VGGT route did not capture layers: {sorted(missing)}")

        if not reconstruct:
            return {"kv_by_layer": cache}
        features = [None if value is None else value.float() for value in features]
        with torch.autocast(device_type=images.device.type, enabled=False):
            pose = camera_head(features, patch_token_start=int(aggregator.patch_token_start))
            depth, depth_conf = dense_head(
                features,
                images=images.float(),
                patch_token_start=int(aggregator.patch_token_start),
            )
        return {
            "kv_by_layer": cache,
            "pose_enc": pose,
            "depth": depth,
            "depth_conf": depth_conf,
            "language_state": language,
        }

    @staticmethod
    def _merge_route_memory(
        high_result: Mapping[str, object] | None,
        low_result: Mapping[str, object] | None,
        high_indices: torch.Tensor,
        low_indices: torch.Tensor,
        *,
        callback_layer_indices: Sequence[int],
        batch_size: int,
        dtype: torch.dtype,
    ) -> dict[int, dict[str, object]]:
        """Merge route-local K/V and restore original sample order."""
        if high_result is None and low_result is None:
            raise RuntimeError("At least one route must contain samples.")
        merged: dict[int, dict[str, object]] = {}
        route_order = torch.cat([high_indices, low_indices], dim=0)
        inverse = torch.argsort(route_order)
        for layer in callback_layer_indices:
            high_entry = None if high_result is None else high_result["kv_by_layer"][layer]
            low_entry = None if low_result is None else low_result["kv_by_layer"][layer]
            if high_entry is not None and low_entry is not None:
                k = torch.cat([high_entry["k"], low_entry["k"]], dim=0)
                v = torch.cat([high_entry["v"], low_entry["v"]], dim=0)
            elif high_entry is not None:
                k, v = high_entry["k"], high_entry["v"]
            else:
                assert low_entry is not None
                k, v = low_entry["k"], low_entry["v"]
            if int(k.shape[0]) != batch_size:
                raise RuntimeError("Routed memory batch dimension mismatch.")
            if high_result is not None and low_result is not None:
                k = k.index_select(0, inverse)
                v = v.index_select(0, inverse)
            merged[int(layer)] = {
                "layer_idx": int(layer),
                "k": k.to(dtype=dtype),
                "v": v.to(dtype=dtype),
            }
        return merged

    def forward_mixed_with_reconstruction(
        self,
        images: torch.Tensor,
        high_noise_mask: torch.Tensor,
        *,
        context: torch.Tensor,
        language_context_mask: torch.Tensor,
    ) -> dict[str, object]:
        """Run only the selected route and return action/reconstruction outputs."""
        validate_vggt_images(images, patch_size=int(self.va1.patch_size))
        if high_noise_mask.dtype != torch.bool or high_noise_mask.ndim != 1:
            raise ValueError("`high_noise_mask` must be boolean [B].")
        if high_noise_mask.shape[0] != images.shape[0]:
            raise ValueError("Route mask batch dimension does not match images.")
        if images.shape[0] == 0 or high_noise_mask.device != images.device:
            raise ValueError("Route batch must be nonempty and mask/images must share a device.")
        if context.ndim != 3 or language_context_mask.ndim != 2:
            raise ValueError("Language context must be [B,L,D] and [B,L].")
        if context.shape[:2] != language_context_mask.shape or context.shape[0] != images.shape[0]:
            raise ValueError("Language context/mask batch dimensions must match images.")

        high_indices = torch.nonzero(high_noise_mask, as_tuple=False).flatten()
        low_indices = torch.nonzero(~high_noise_mask, as_tuple=False).flatten()
        high_result = low_result = None
        anchor = images.sum() * 0.0
        grad_context = nullcontext() if self.trainable else torch.no_grad()

        with grad_context, self._autocast_context(images.device):
            # Share only patch extraction, never accidentally the route prefixes.
            patches = rope = None
            if self.share_patch_embedding:
                patches, rope = self._prepare_patch_tokens(images, self.va1)
            for route, indices in (("high", high_indices), ("low", low_indices)):
                active = indices.numel() > 0
                if not active and not (self.trainable and self.training):
                    continue
                chosen = indices if active else torch.zeros(1, device=images.device, dtype=torch.long)
                route_images = images.index_select(0, chosen)
                route_tokens, frame_rope = self.prepare_route_tokens(
                    route_images, route,
                    patch_tokens=None if patches is None else patches.index_select(0, chosen),
                    frame_rope=rope,
                )
                use_language = route == "low" or self.high_noise_use_language_condition
                result = self._run_visual_route(
                    route_tokens, route_images,
                    aggregator=self.va1 if route == "high" else self.va2,
                    frame_rope=frame_rope,
                    context=context.index_select(0, chosen) if use_language else None,
                    language_context_mask=language_context_mask.index_select(0, chosen) if use_language else None,
                    use_language=use_language,
                )
                if not active:
                    anchor = anchor + self._zero_anchor(result, images)
                elif route == "high":
                    high_result = result
                else:
                    low_result = result

        memory = self._merge_route_memory(
            high_result,
            low_result,
            high_indices,
            low_indices,
            callback_layer_indices=self.callback_layer_indices,
            batch_size=int(images.shape[0]),
            dtype=(torch.bfloat16 if self.fp32_storage_bf16_compute else images.dtype),
        )
        for entry in memory.values():
            entry["high_noise_mask"] = high_noise_mask
        return {
            "kv_by_layer": memory,
            "high_result": high_result,
            "low_result": low_result,
            "high_indices": high_indices,
            "low_indices": low_indices,
            "inactive_anchor": anchor,
        }


def build_vggt_moe_recon_backbone(
    *,
    weights_path: str = DEFAULT_VGGT_WEIGHTS,
    aggregator_checkpoint_path: str | None = None,
    callback_layer_indices: Sequence[int],
    input_camera_keys: Sequence[str],
    trainable: bool = True,
    text_dim: int = 4096,
    share_patch_embedding: bool = True,
    share_camera_register_tokens: bool = True,
    share_geometry_heads: bool = True,
    high_noise_use_language_condition: bool = False,
    aggregator_factory: Callable[..., nn.Module] | None = None,
) -> VGGTMoEReconBackbone:
    """Build pretrained streams with independent choices for input/head sharing."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not aggregator_checkpoint_path:
        raise ValueError("Multi-rank reconstruction VGGT requires aggregator_checkpoint_path.")
    if aggregator_factory is None:
        from vggt_omega.models.aggregator import Aggregator

        aggregator_factory = Aggregator
    high = aggregator_factory()
    low = aggregator_factory()
    checkpoint = aggregator_checkpoint_path or weights_path
    load_aggregator_weights(high, checkpoint, aggregator_only=aggregator_checkpoint_path is not None)
    load_aggregator_weights(low, checkpoint, aggregator_only=aggregator_checkpoint_path is not None)

    from vggt_omega.models.heads import CameraHead, DenseHead

    camera_head = CameraHead(dim_in=2048)
    dense_head = DenseHead(dim_in=2048, patch_size=int(high.patch_size))
    full = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(full, dict) and "state_dict" in full:
        full = full["state_dict"]
    camera_head.load_state_dict(
        {key[12:]: value for key, value in full.items() if key.startswith("camera_head.")},
        strict=True,
    )
    dense_head.load_state_dict(
        {key[11:]: value for key, value in full.items() if key.startswith("dense_head.")},
        strict=True,
    )

    model = VGGTMoEReconBackbone(
        high,
        low,
        callback_layer_indices,
        input_camera_keys=input_camera_keys,
        va1_camera_keys=input_camera_keys,
        va2_camera_keys=input_camera_keys,
        trainable=trainable,
        share_patch_embedding=share_patch_embedding,
        share_camera_register_tokens=share_camera_register_tokens,
        share_geometry_heads=share_geometry_heads,
        high_noise_use_language_condition=high_noise_use_language_condition,
        isolate_vggt_streams=True,
        text_dim=text_dim,
        va2_use_language_condition=True,
        action_vggt_source="both",
        camera_head=camera_head,
        dense_head=dense_head,
    )
    return model


__all__ = ["VGGTMoEReconBackbone", "build_vggt_moe_recon_backbone"]
