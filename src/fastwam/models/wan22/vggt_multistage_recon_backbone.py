"""Independent N-stage VGGT experts; no dependency on the dual-stream topology."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .vggt_mot import DEFAULT_VGGT_WEIGHTS, load_aggregator_weights, validate_vggt_images


class VGGTMultiStageReconBackbone(nn.Module):
    def __init__(self, experts, callback_layer_indices, *, input_camera_keys,
                 text_dim, camera_head, dense_head, stage_use_language_condition,
                 trainable=True, share_patch_embedding=True,
                 share_camera_register_tokens=True, share_geometry_heads=True):
        super().__init__()
        if not experts or len({id(x) for x in experts}) != len(experts):
            raise ValueError("Experts must be nonempty and independently instantiated.")
        self.stage_names = tuple(f"coarse{i+1}" for i in range(len(experts)))
        self.stage_use_language_condition = tuple(stage_use_language_condition)
        if len(self.stage_use_language_condition) != len(experts) or any(type(x) is not bool for x in self.stage_use_language_condition):
            raise ValueError("Language switches must be one bool per stage.")
        self.callback_layer_indices = tuple(int(x) for x in callback_layer_indices)
        first = experts[0]
        if not self.callback_layer_indices or len(set(self.callback_layer_indices)) != len(self.callback_layer_indices) or not set(self.callback_layer_indices) <= set(range(first.depth)):
            raise ValueError("Invalid callback layers.")
        self.input_camera_keys = tuple(input_camera_keys)
        if not self.input_camera_keys or len(set(self.input_camera_keys)) != len(self.input_camera_keys):
            raise ValueError("Camera keys must be nonempty and unique.")
        self.trainable = bool(trainable)
        self.fp32_storage_bf16_compute = False
        self.share_patch_embedding = bool(share_patch_embedding)
        self.share_camera_register_tokens = bool(share_camera_register_tokens)
        self.share_geometry_heads = bool(share_geometry_heads)
        for expert in experts[1:]:
            for attr in ("depth", "patch_size", "patch_token_start", "inter_frame_attention_types"):
                if getattr(expert, attr) != getattr(first, attr):
                    raise ValueError(f"Expert {attr} mismatch.")
            if self.share_patch_embedding:
                expert.patch_embed = first.patch_embed
            expert.rope_embed = first.rope_embed
            if self.share_camera_register_tokens:
                expert.camera_token = first.camera_token
                expert.register_token = first.register_token
        self.experts = nn.ModuleDict(dict(zip(self.stage_names, experts)))
        width = first.camera_token.shape[-1]
        self.language_projector = (nn.Sequential(nn.Linear(text_dim, width), nn.GELU(), nn.Linear(width, width))
                                   if any(self.stage_use_language_condition) else None)
        head_names = ("shared",) if self.share_geometry_heads else self.stage_names
        self.camera_heads = nn.ModuleDict({name: camera_head if i == 0 else copy.deepcopy(camera_head)
                                          for i, name in enumerate(head_names)})
        self.dense_heads = nn.ModuleDict({name: dense_head if i == 0 else copy.deepcopy(dense_head)
                                         for i, name in enumerate(head_names)})
        self.requires_grad_(self.trainable)
        self.train(self.trainable)

    @property
    def aggregator(self):
        return self.experts[self.stage_names[0]]

    def geometry_heads(self, route):
        if route not in self.experts:
            raise ValueError(f"Unknown stage {route}")
        key = "shared" if self.share_geometry_heads else route
        return self.camera_heads[key], self.dense_heads[key]

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self.fp32_storage_bf16_compute:
            self.experts.float()
            if self.language_projector is not None:
                self.language_projector.float()
        self.camera_heads.float()
        self.dense_heads.float()
        return self

    def train(self, mode=True):
        return super().train(bool(mode) and self.trainable)

    def _autocast_context(self, device: torch.device):
        """Use BF16 kernels when VGGT weights are intentionally kept in FP32.

        Reconstruction and MoE reconstruction backbones share this class, so
        the precision policy belongs here rather than on FrozenVGGTBackbone.
        """
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bool(getattr(self, "fp32_storage_bf16_compute", False))
            and device.type == "cuda",
        )

    @staticmethod
    def _expand_input_tokens(
        aggregator: nn.Module,
        patch_tokens: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Build shared ``[B,V,S,D]`` camera/register/patch input tokens."""
        camera = aggregator.camera_token[:, 0:1].expand(
            batch_size, 1, *aggregator.camera_token.shape[2:]
        )
        if num_frames > 1:
            camera = torch.cat(
                [
                    camera,
                    aggregator.camera_token[:, 1:].expand(
                        batch_size, num_frames - 1, *aggregator.camera_token.shape[2:]
                    ),
                ],
                dim=1,
            )
        camera = camera.reshape(batch_size * num_frames, 1, -1)

        register = aggregator.register_token[:, 0:1].expand(
            batch_size, 1, *aggregator.register_token.shape[2:]
        )
        if num_frames > 1:
            register = torch.cat(
                [
                    register,
                    aggregator.register_token[:, 1:].expand(
                        batch_size, num_frames - 1, *aggregator.register_token.shape[2:]
                    ),
                ],
                dim=1,
            )
        register = register.reshape(
            batch_size * num_frames, register.shape[2], register.shape[3]
        )
        tokens = torch.cat([camera, register, patch_tokens], dim=1)
        return tokens.reshape(batch_size, num_frames, tokens.shape[1], tokens.shape[2])

    @staticmethod
    def _attention_qkv(
        block: nn.Module,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project one VGGT expert to head-major ``[B,H,S,Dh]`` Q/K/V."""
        attention = block.attn
        qkv = attention.qkv(block.norm1(tokens))
        batch_size, seq_len, _ = qkv.shape
        width = int(attention.qkv.in_features)
        qkv = qkv.reshape(
            batch_size,
            seq_len,
            3,
            int(attention.num_heads),
            width // int(attention.num_heads),
        )
        q, k, v = torch.unbind(qkv, dim=2)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        if bool(attention.use_qk_norm):
            q = attention.q_norm(q)
            k = attention.k_norm(k)
        return q, k, v

    @staticmethod
    def _post_inter_frame_attention(
        block: nn.Module,
        residual_tokens: torch.Tensor,
        attention_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the owning expert's output projection, residual, and FFN."""
        attention = block.attn
        projected = attention.proj(attention_output)
        projected = attention.proj_drop(projected)
        x = residual_tokens + block.ls1(projected)
        return x + block.ls2(block.mlp(block.norm2(x)))

    @staticmethod
    def _active_inter_frame_tokens(
        tokens: torch.Tensor,
        *,
        attention_type: str,
        patch_token_start: int,
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        """Flatten the token subset used by a VGGT inter-frame block.

        Global layers mix all ``V*S`` tokens.  Register layers mix only the
        ``V*P`` camera/register prefix and pass patch tokens through unchanged.
        """
        batch_size, num_frames, num_tokens, width = tokens.shape
        if attention_type == "global":
            active = tokens.reshape(batch_size, num_frames * num_tokens, width)

            def restore(updated: torch.Tensor) -> torch.Tensor:
                return updated.reshape(batch_size, num_frames, num_tokens, width)

            return active, restore
        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type!r}.")

        prefix = tokens[:, :, :patch_token_start]
        patches = tokens[:, :, patch_token_start:]
        active = prefix.reshape(batch_size, num_frames * patch_token_start, width)

        def restore(updated: torch.Tensor) -> torch.Tensor:
            updated_prefix = updated.reshape(
                batch_size, num_frames, patch_token_start, width
            )
            return torch.cat([updated_prefix, patches], dim=2)

        return active, restore

    @staticmethod
    def _sdpa(q, k, v, key_valid=None):
        if key_valid is None:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            # SDPA boolean masks use True for permitted key positions.
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_valid[:, None, None, :])
        return out.transpose(1, 2).contiguous().flatten(2)

    def _language(self, context, mask, *, dtype, device):
        if context is None or context.ndim != 3 or mask is None or mask.shape != context.shape[:2]:
            raise ValueError("Language context/mask must be [B,L,D] and [B,L].")
        valid = mask.to(device=device, dtype=torch.bool)
        # Mask before projection as NaN * 0 is still NaN.
        language = self.language_projector(
            context.to(device=device, dtype=dtype).masked_fill(~valid[..., None], 0))
        return language.masked_fill(~valid[..., None], 0), valid

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
        aggregator = self.experts[route]
        if route not in self.experts:
            raise ValueError(f"Unknown VGGT route: {route}")
        if patch_tokens is None:
            patch_tokens, frame_rope = self._prepare_patch_tokens(images, aggregator)
        b, v, n, d = patch_tokens.shape
        tokens = self._expand_input_tokens(aggregator, patch_tokens.reshape(b * v, n, d),
                                          batch_size=b, num_frames=v)
        return tokens, frame_rope

    def _run_visual_route(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
        *,
        route: str,
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

        aggregator = self.experts[route]
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
                    # selected stage indirectly through its updated visual tokens.
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

    def forward_mixed_with_reconstruction(self, images, stage_ids, *, context, language_context_mask):
        validate_vggt_images(images, patch_size=int(self.aggregator.patch_size))
        if stage_ids.dtype != torch.long or stage_ids.shape != (len(images),) or stage_ids.device != images.device:
            raise ValueError("stage_ids must be int64 [B] on the image device.")
        if not len(images) or bool(((stage_ids < 0) | (stage_ids >= len(self.stage_names))).any()):
            raise ValueError("Invalid stage ids or empty batch.")
        if context.ndim != 3 or language_context_mask.shape != context.shape[:2] or len(context) != len(images):
            raise ValueError("Language context/mask must match [B,L,D]/[B,L].")
        results, indices = {}, {}
        anchor = images.reshape(-1)[:0].sum()
        grad_context = nullcontext() if self.trainable else torch.no_grad()
        with grad_context, self._autocast_context(images.device):
            patches = rope = None
            if self.share_patch_embedding:
                patches, rope = self._prepare_patch_tokens(images, self.aggregator)
            for i, route in enumerate(self.stage_names):
                index = (stage_ids == i).nonzero(as_tuple=False).flatten()
                indices[route] = index
                results[route] = None
                if not index.numel() and not (self.trainable and self.training):
                    continue
                chosen = index if index.numel() else stage_ids.new_zeros(1)
                selected_images = images.index_select(0, chosen)
                tokens, frame_rope = self.prepare_route_tokens(
                    selected_images, route, frame_rope=rope,
                    patch_tokens=None if patches is None else patches.index_select(0, chosen))
                result = self._run_visual_route(
                    tokens, selected_images, route=route, frame_rope=frame_rope,
                    context=context.index_select(0, chosen),
                    language_context_mask=language_context_mask.index_select(0, chosen),
                    use_language=self.stage_use_language_condition[i])
                if index.numel():
                    results[route] = result
                    # Keep the final language state connected even for sparse action capture.
                    if result.get("language_state") is not None:
                        anchor = anchor + result["language_state"].reshape(-1)[:0].sum()
                else:
                    anchor = anchor + self._zero_anchor(result, images)
        active = [name for name in self.stage_names if results[name] is not None]
        inverse = torch.argsort(torch.cat([indices[name] for name in active]))
        memory = {}
        for layer in self.callback_layer_indices:
            memory[layer] = {"layer_idx": layer, "stage_ids": stage_ids}
            for key in ("k", "v"):
                memory[layer][key] = torch.cat([results[name]["kv_by_layer"][layer][key] for name in active]).index_select(0, inverse)
        return {"kv_by_layer": memory, "results": results, "indices": indices, "inactive_anchor": anchor}


def build_vggt_multistage_recon_backbone(*, num_stages, weights_path=DEFAULT_VGGT_WEIGHTS,
        aggregator_checkpoint_path=None, callback_layer_indices, input_camera_keys,
        stage_use_language_condition, text_dim=4096, trainable=True,
        share_patch_embedding=True, share_camera_register_tokens=True, share_geometry_heads=True):
    from vggt_omega.models.aggregator import Aggregator
    from vggt_omega.models.heads import CameraHead, DenseHead
    first = Aggregator()
    load_aggregator_weights(first, aggregator_checkpoint_path or weights_path,
                           aggregator_only=aggregator_checkpoint_path is not None)
    # Memoized copies avoid allocating a temporary DINO for every shared expert.
    experts = [first]
    for _ in range(num_stages - 1):
        memo = {id(first.rope_embed): first.rope_embed}
        if share_patch_embedding:
            memo[id(first.patch_embed)] = first.patch_embed
        if share_camera_register_tokens:
            memo.update({id(first.camera_token): first.camera_token, id(first.register_token): first.register_token})
        experts.append(copy.deepcopy(first, memo))
    camera = CameraHead(dim_in=2048)
    dense = DenseHead(dim_in=2048, patch_size=int(first.patch_size))
    full = torch.load(weights_path, map_location="cpu", weights_only=True)
    full = full.get("state_dict", full)
    camera.load_state_dict({k[12:]: v for k, v in full.items() if k.startswith("camera_head.")}, strict=True)
    dense.load_state_dict({k[11:]: v for k, v in full.items() if k.startswith("dense_head.")}, strict=True)
    return VGGTMultiStageReconBackbone(experts, callback_layer_indices,
        input_camera_keys=input_camera_keys, text_dim=text_dim, camera_head=camera, dense_head=dense,
        stage_use_language_condition=stage_use_language_condition, trainable=trainable,
        share_patch_embedding=share_patch_embedding, share_camera_register_tokens=share_camera_register_tokens,
        share_geometry_heads=share_geometry_heads)
