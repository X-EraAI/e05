from __future__ import annotations

"""Single-stream VGGT with language-conditioned future reconstruction.

The released VGGT Aggregator does not expose a context input.  This module
keeps its image/frame-attention path intact and replays the inter-frame blocks
inside FastWAM so a projected language sequence can participate in every
inter-frame layer.  Only visual K/V are returned to the ActionDiT bridge;
language influences the bridge indirectly through the updated visual state.
"""

import os
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vggt_mot import (
    DEFAULT_VGGT_WEIGHTS,
    VGGT_PATCH_SIZE,
    FrozenVGGTBackbone,
    load_aggregator_weights,
    validate_vggt_images,
)
from .vggt_mot_backbone import VGGTMoTBackbone


class VGGTLanguageReconBackbone(FrozenVGGTBackbone):
    """Language-conditioned VGGT that predicts future geometry from current RGB.

    Tensor flow for one batch is:

    ``images [B,V,3,H,W]``
      -> patch tokens ``[B,V,S,1024]``
      -> active visual tokens ``[B,V*S_active,1024]``
      + projected language ``[B,L,1024]``
      -> joint inter-frame attention ``[B,V*S_active+L,1024]``
      -> visual K/V for ActionDiT and 2048-wide head features.

    Language is appended only to the active inter-frame sequence.  It is not
    copied per camera and is never exposed as ActionDiT memory.  Register
    layers update camera/register tokens while restoring patch tokens exactly
    as the native VGGT Aggregator does.
    """

    def __init__(
        self,
        aggregator: nn.Module,
        callback_layer_indices: Sequence[int],
        *,
        text_dim: int,
        trainable: bool = True,
        camera_head: nn.Module | None = None,
        dense_head: nn.Module | None = None,
    ) -> None:
        super().__init__(
            aggregator,
            callback_layer_indices,
            trainable=trainable,
            enable_reconstruction_heads=False,
            camera_head=camera_head,
            dense_head=dense_head,
        )
        width = int(aggregator.camera_token.shape[-1])
        self.text_dim = int(text_dim)
        self.language_projector = nn.Sequential(
            nn.Linear(self.text_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        if self.camera_head is not None:
            self.camera_head.float()
        if self.dense_head is not None:
            self.dense_head.float()

    def load_pretrained_checkpoint(self, path: str) -> None:
        """Restore the complete pretrained backbone, including language projection."""
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("vggt_backbone"), Mapping):
            raise ValueError("VGGT backbone checkpoint must contain a `vggt_backbone` state dict.")
        self.load_state_dict(payload["vggt_backbone"], strict=True)

    def _language(
        self,
        context: torch.Tensor,
        language_context_mask: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project ``[B,L,text_dim]`` context to ``[B,L,D]`` VGGT tokens."""
        if context is None:
            raise ValueError("Language-conditioned VGGT requires `context`.")
        if context.ndim != 3:
            raise ValueError(
                "`context` must be [B,L,text_dim], "
                f"got {tuple(context.shape)}."
            )
        if language_context_mask is None or language_context_mask.ndim != 2:
            raise ValueError("`language_context_mask` must be [B,L].")
        if tuple(context.shape[:2]) != tuple(language_context_mask.shape):
            raise ValueError(
                "`context` and `language_context_mask` shape mismatch: "
                f"context={tuple(context.shape)}, "
                f"mask={tuple(language_context_mask.shape)}."
            )
        if int(context.shape[-1]) != self.text_dim:
            raise ValueError(
                f"`context` last dim must be {self.text_dim}, "
                f"got {context.shape[-1]}."
            )
        valid = language_context_mask.to(device=device, dtype=torch.bool)
        state = self.language_projector(
            context.to(device=device, dtype=dtype).masked_fill(~valid[..., None], 0)
        )
        # Zeroing padded values also protects the language residual/MLP path;
        # the SDPA key mask below prevents padded values from being attended to.
        state = state.masked_fill(~valid[..., None], 0)
        return state, valid

    @staticmethod
    def _sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run head-major SDPA and return flattened ``[B,S,D]`` output."""
        if key_valid is None:
            output = F.scaled_dot_product_attention(q, k, v)
        else:
            # PyTorch SDPA boolean masks use True for permitted key positions.
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=key_valid[:, None, None, :],
            )
        return output.transpose(1, 2).contiguous().flatten(2)

    def _run_language_layer(
        self,
        tokens: torch.Tensor,
        language: torch.Tensor,
        language_valid: torch.Tensor,
        *,
        layer_idx: int,
        attention_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply one frame-independent inter-frame layer.

        ``tokens`` is ``[B,V,N,D]`` and ``language`` is ``[B,L,D]``.  For a
        global layer, all visual tokens are active.  For a register layer,
        only ``V * patch_token_start`` camera/register tokens are active; patch
        tokens are restored unchanged after attention.  The returned K/V pair
        contains visual keys and values only, so no language token enters the
        ActionDiT bridge sequence.
        """
        block = self.aggregator.inter_frame_blocks[layer_idx]
        active_visual, restore = VGGTMoTBackbone._active_inter_frame_tokens(
            tokens,
            attention_type=attention_type,
            patch_token_start=int(self.aggregator.patch_token_start),
        )
        q_visual, k_visual, v_visual = VGGTMoTBackbone._attention_qkv(
            block, active_visual
        )
        q_language, k_language, v_language = VGGTMoTBackbone._attention_qkv(
            block, language
        )

        q_joint = torch.cat([q_visual, q_language], dim=2)
        k_joint = torch.cat([k_visual, k_language], dim=2)
        v_joint = torch.cat([v_visual, v_language], dim=2)
        visual_valid = torch.ones(
            (tokens.shape[0], active_visual.shape[1]),
            dtype=torch.bool,
            device=tokens.device,
        )
        key_valid = torch.cat([visual_valid, language_valid], dim=1)
        mixed_joint = self._sdpa(q_joint, k_joint, v_joint, key_valid)

        joint_residual = torch.cat([active_visual, language], dim=1)
        updated_joint = VGGTMoTBackbone._post_inter_frame_attention(
            block,
            joint_residual,
            mixed_joint,
        )
        visual_length = int(active_visual.shape[1])
        updated_visual = restore(updated_joint[:, :visual_length]).contiguous()
        updated_language = updated_joint[:, visual_length:]
        updated_language = updated_language * language_valid[..., None].to(
            updated_language.dtype
        )
        return updated_visual, updated_language, (k_visual, v_visual)

    def _initialize_state(
        self,
        images: torch.Tensor,
        context: torch.Tensor,
        language_context_mask: torch.Tensor,
    ):
        """Tokenize current observations and language exactly once."""
        validate_vggt_images(
            images,
            patch_size=int(getattr(self.aggregator, "patch_size", VGGT_PATCH_SIZE)),
        )
        batch_size, num_views, channels, height, width = images.shape
        aggregator = self.aggregator
        normalized = (images - aggregator._resnet_mean) / aggregator._resnet_std
        patch_dtype = next(aggregator.patch_embed.parameters()).dtype
        patch_tokens = aggregator.patch_embed(
            normalized.to(patch_dtype).reshape(
                batch_size * num_views, channels, height, width
            )
        )
        if isinstance(patch_tokens, Mapping):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        frame_dtype = next(aggregator.frame_blocks[0].parameters()).dtype
        patch_tokens = patch_tokens.to(frame_dtype)
        tokens = VGGTMoTBackbone._expand_input_tokens(
            aggregator,
            patch_tokens,
            batch_size=batch_size,
            num_frames=num_views,
        )

        grid = (height // int(aggregator.patch_size), width // int(aggregator.patch_size))
        with torch.no_grad():
            rope_sin, rope_cos = aggregator.rope_embed(H=grid[0], W=grid[1])
        frame_rope = (
            rope_sin.to(device=images.device, dtype=torch.float32),
            rope_cos.to(device=images.device, dtype=torch.float32),
        )
        language, language_valid = self._language(
            context,
            language_context_mask,
            dtype=frame_dtype,
            device=images.device,
        )
        return tokens, language, language_valid, frame_rope

    def _run_pass(
        self, tokens, language, language_valid, frame_rope, *,
        memory_dtype, capture_memory=True, capture_features=True,
    ):
        """Advance visual/language state through the shared Aggregator."""
        aggregator = self.aggregator
        batch_size, num_views = tokens.shape[:2]
        cache: dict[int, dict[str, object]] = {}
        features: list[torch.Tensor | None] = [None] * int(aggregator.depth)
        callback_layers = set(self.callback_layer_indices) if capture_memory else set()
        head_layers = set(getattr(aggregator, "cached_layer_indices", ())) if capture_features else set()

        for layer_idx in range(int(aggregator.depth)):
            # Frame attention is visual-only and runs independently per view.
            _, frame_tokens = aggregator._run_frame_block(
                tokens,
                batch_size,
                num_views,
                tokens.shape[2],
                tokens.shape[3],
                layer_idx,
                frame_rope,
            )
            tokens = frame_tokens
            attention_type = aggregator.inter_frame_attention_types[layer_idx]
            tokens, language, visual_kv = self._run_language_layer(
                tokens,
                language,
                language_valid,
                layer_idx=layer_idx,
                attention_type=attention_type,
            )
            if capture_features and (layer_idx in head_layers or layer_idx in callback_layers):
                # CameraHead/DenseHead consume native 2048-wide features:
                # frame output (1024) concatenated with inter-frame output (1024).
                features[layer_idx] = torch.cat([frame_tokens, tokens], dim=-1)
            if layer_idx in callback_layers:
                key, value = visual_kv
                cache[layer_idx] = {
                    "layer_idx": layer_idx,
                    "attention_type": attention_type,
                    "k": key.to(dtype=memory_dtype),
                    "v": value.to(dtype=memory_dtype),
                }

        missing = callback_layers.difference(cache)
        if missing:
            raise RuntimeError(
                f"Language-conditioned VGGT did not capture layers: {sorted(missing)}"
            )
        return tokens, language, {
            "kv_by_layer": cache,
            "future_features": features,
            "patch_token_start": int(aggregator.patch_token_start),
        }

    def _forward_impl_reconstruction(self, images, context, language_context_mask):
        tokens, language, valid, rope = self._initialize_state(
            images, context, language_context_mask
        )
        _, _, result = self._run_pass(
            tokens, language, valid, rope,
            memory_dtype=(torch.bfloat16 if self.fp32_storage_bf16_compute else images.dtype),
        )
        return result

    def forward(
        self,
        images: torch.Tensor,
        *,
        context: torch.Tensor,
        language_context_mask: torch.Tensor,
    ) -> dict[int, dict[str, object]]:
        with nullcontext() if self.trainable else torch.no_grad():
            with self._autocast_context(images.device):
                return self._forward_impl_reconstruction(
                    images, context, language_context_mask
                )["kv_by_layer"]

    def forward_with_reconstruction(
        self,
        images: torch.Tensor,
        *,
        context: torch.Tensor,
        language_context_mask: torch.Tensor,
    ) -> dict[str, object]:
        with nullcontext() if self.trainable else torch.no_grad():
            with self._autocast_context(images.device):
                result = self._forward_impl_reconstruction(
                    images, context, language_context_mask
                )
        return self._decode_reconstruction(result, images)

    def _decode_reconstruction(self, result, images):
        if self.camera_head is None or self.dense_head is None:
            raise RuntimeError("Language reconstruction heads are not enabled.")
        features = [None if value is None else value.float() for value in result["future_features"]]
        with torch.autocast(device_type=images.device.type, enabled=False):
            future_pose = self.camera_head(
                features,
                patch_token_start=result["patch_token_start"],
            )
            future_depth, future_conf = self.dense_head(
                features,
                images=images.float(),
                patch_token_start=result["patch_token_start"],
            )
        result.update(
            future_pose=future_pose,
            future_depth=future_depth,
            future_depth_conf=future_conf,
        )
        return result


class VGGTLoopLanguageReconBackbone(VGGTLanguageReconBackbone):
    """Shared-weight visual/language recurrence with optional per-pass readouts."""

    def __init__(self, *args, num_vggt_forwards=2, **kwargs):
        if isinstance(num_vggt_forwards, bool) or not isinstance(num_vggt_forwards, int) or num_vggt_forwards < 1:
            raise ValueError("num_vggt_forwards must be a positive integer.")
        super().__init__(*args, **kwargs)
        self.num_vggt_forwards = num_vggt_forwards

    def _passes(self, images, context, language_context_mask, *, all_passes, reconstruct):
        with nullcontext() if self.trainable else torch.no_grad():
            tokens, language, valid, rope = self._initialize_state(
                images, context, language_context_mask
            )
            results = []
            for index in range(self.num_vggt_forwards):
                capture = all_passes or index == self.num_vggt_forwards - 1
                tokens, language, result = self._run_pass(
                    tokens, language, valid, rope, memory_dtype=images.dtype,
                    capture_memory=capture, capture_features=capture and reconstruct,
                )
                if capture:
                    if reconstruct:
                        result = self._decode_reconstruction(result, images)
                    results.append(result)
        return results

    def forward_all_reconstructions(self, images, *, context, language_context_mask):
        return self._passes(images, context, language_context_mask, all_passes=True, reconstruct=True)

    def forward(self, images, *, context, language_context_mask):
        return self._passes(images, context, language_context_mask, all_passes=False, reconstruct=False)[-1]["kv_by_layer"]

    def forward_with_reconstruction(self, images, *, context, language_context_mask):
        return self._passes(images, context, language_context_mask, all_passes=False, reconstruct=True)[-1]


def build_vggt_language_recon_backbone(
    *,
    weights_path: str = DEFAULT_VGGT_WEIGHTS,
    aggregator_checkpoint_path: str | None = None,
    head_checkpoint_path: str | None = None,
    callback_layer_indices: Sequence[int],
    text_dim: int,
    trainable: bool = True,
    skip_vggt_load_from_pretrain: bool = False,
    aggregator_factory: Callable[..., nn.Module] | None = None,
    num_vggt_forwards: int | None = None,
) -> VGGTLanguageReconBackbone:
    """Build one Aggregator plus pretrained FP32 geometry heads."""
    if (
        not skip_vggt_load_from_pretrain
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
        and not aggregator_checkpoint_path
    ):
        raise ValueError(
            "Multi-rank language reconstruction VGGT requires "
            "`aggregator_checkpoint_path`."
        )
    if aggregator_factory is None:
        from vggt_omega.models.aggregator import Aggregator

        aggregator_factory = Aggregator
    aggregator = aggregator_factory(
        callback_layer_indices=tuple(int(layer) for layer in callback_layer_indices)
    )
    if not skip_vggt_load_from_pretrain:
        checkpoint = aggregator_checkpoint_path or weights_path
        load_aggregator_weights(
            aggregator,
            checkpoint,
            aggregator_only=aggregator_checkpoint_path is not None,
        )

    from vggt_omega.models.heads import CameraHead, DenseHead

    camera_head = CameraHead(dim_in=2048)
    dense_head = DenseHead(dim_in=2048, patch_size=int(aggregator.patch_size))
    if not skip_vggt_load_from_pretrain:
        full = torch.load(
            head_checkpoint_path or weights_path, map_location="cpu", weights_only=True
        )
        if isinstance(full, Mapping) and "state_dict" in full:
            full = full["state_dict"]
        camera_state = {
            key[len("camera_head.") :]: value
            for key, value in full.items()
            if key.startswith("camera_head.")
        }
        dense_state = {
            key[len("dense_head.") :]: value
            for key, value in full.items()
            if key.startswith("dense_head.")
        }
        if not camera_state or not dense_state:
            raise RuntimeError(
                "Full VGGT checkpoint is required to initialize reconstruction heads."
            )
        camera_head.load_state_dict(camera_state, strict=True)
        dense_head.load_state_dict(dense_state, strict=True)

    camera_head.float()
    dense_head.float()
    backbone_cls = VGGTLanguageReconBackbone if num_vggt_forwards is None else VGGTLoopLanguageReconBackbone
    loop_kwargs = {} if num_vggt_forwards is None else {"num_vggt_forwards": num_vggt_forwards}
    return backbone_cls(
        aggregator,
        callback_layer_indices,
        text_dim=int(text_dim),
        trainable=bool(trainable),
        camera_head=camera_head,
        dense_head=dense_head,
        **loop_kwargs,
    )


__all__ = [
    "VGGTLanguageReconBackbone",
    "VGGTLoopLanguageReconBackbone",
    "build_vggt_language_recon_backbone",
]
