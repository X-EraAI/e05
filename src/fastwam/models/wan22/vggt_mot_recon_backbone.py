from __future__ import annotations

import os
import copy
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vggt_mot import DEFAULT_VGGT_WEIGHTS, VGGT_PATCH_SIZE, load_aggregator_weights, validate_vggt_images
from .vggt_mot_backbone import VGGTMoTBackbone


class VGGTMoTReconBackbone(VGGTMoTBackbone):
    """MoT VGGT backbone with optional per-stream language and geometry heads.

    Image tokens have shape ``[B,V,S,D]`` and language has ``[B,L,D]``.
    Patch embedding, camera/register prefixes and decoders share independently.  The
    language sequence is never copied per camera and bypasses frame attention.
    Each enabled stream maintains its own language state, initially produced by
    the shared language projector. It bypasses frame attention and is updated
    by the owning inter-frame block, then split from visual tokens again. The
    bridge receives only visual K/V; the language state affects it indirectly
    through the updated visual tokens of the enabled streams.

    ``va2_to_va1`` limits va1 queries to their own visual/language K/V, while
    va2 queries read both streams plus their valid language tokens.
    This controls forward visibility; gradients through va1 K/V are retained.
    """

    VALID_ACTION_VGGT_SOURCES = frozenset(("both", "va1", "va2"))
    VALID_STREAM_ATTENTION = frozenset(("isolated", "coupled", "va2_to_va1"))

    def __init__(self, *args, text_dim: int, camera_head, dense_head,
                 va1_use_language_condition: bool = False,
                 va2_use_language_condition: bool = True,
                 vggt_stream_attention: str | None = None,
                 action_vggt_source: str = "both", share_geometry_heads: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        # An explicit mode overrides the legacy symmetric isolation flag.
        mode = ("isolated" if self.isolate_vggt_streams else "coupled") if vggt_stream_attention is None else str(vggt_stream_attention).lower()
        if mode not in self.VALID_STREAM_ATTENTION:
            raise ValueError(f"`vggt_stream_attention` must be one of {sorted(self.VALID_STREAM_ATTENTION)}, got {mode!r}.")
        self.vggt_stream_attention = mode
        self.isolate_vggt_streams = mode == "isolated"
        self.va1_use_language_condition = bool(va1_use_language_condition)
        self.va2_use_language_condition = bool(va2_use_language_condition)
        action_vggt_source = str(action_vggt_source).lower()
        if action_vggt_source not in self.VALID_ACTION_VGGT_SOURCES:
            raise ValueError(
                "`action_vggt_source` must be one of "
                f"{sorted(self.VALID_ACTION_VGGT_SOURCES)}, got {action_vggt_source!r}."
            )
        # `both` preserves the historical bridge input.  `va1`/`va2` select
        # exactly one stream for ActionDiT; the two VGGT streams still use
        # their own configured visibility during VGGT computation.
        self.action_vggt_source = action_vggt_source
        width = int(self.va1.patch_embed.embed_dim if hasattr(self.va1.patch_embed, "embed_dim") else self.va1.camera_token.shape[-1])
        self.language_projector = nn.Sequential(
            nn.Linear(int(text_dim), width), nn.GELU(), nn.Linear(width, width)
        )
        self.camera_head = camera_head
        self.dense_head = dense_head
        self.camera_head.float()
        self.dense_head.float()
        self.share_geometry_heads = bool(share_geometry_heads)
        if not self.share_geometry_heads:
            self.low_camera_head = copy.deepcopy(camera_head)
            self.low_dense_head = copy.deepcopy(dense_head)

    def geometry_heads(self, route):
        """VA1/high and VA2/low heads; aliases only when sharing is enabled."""
        if route not in ("high", "low"):
            raise ValueError(f"Unknown geometry route: {route}")
        if route == "low" and not self.share_geometry_heads:
            return self.low_camera_head, self.low_dense_head
        return self.camera_head, self.dense_head

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Both decoders consume FP32 features even after model.to(bfloat16).
        for route in ("high", "low"):
            for head in self.geometry_heads(route):
                head.float()
        return self

    def _language(self, context, mask, *, dtype, device):
        if context is None:
            raise ValueError("Language-conditioned VGGT requires `context`.")
        if context.ndim != 3 or mask is None or mask.ndim != 2 or context.shape[:2] != mask.shape:
            raise ValueError("`context` must be [B,L,text_dim] and `language_context_mask` [B,L].")
        valid = mask.to(device=device, dtype=torch.bool)
        state = self.language_projector(
            context.to(device=device, dtype=dtype).masked_fill(~valid[..., None], 0))
        return state.masked_fill(~valid[..., None], 0), valid

    @staticmethod
    def _sdpa(q, k, v, key_valid=None):
        if key_valid is None:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            # SDPA boolean masks use True for permitted key positions.
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_valid[:, None, None, :])
        return out.transpose(1, 2).contiguous().flatten(2)

    def _run_language_layer(self, va1_tokens, va2_tokens, language, language_valid, *, layer_idx, attention_type):
        """Run one layer and return visual states, visual K/V, and geometry features.

        ``va1_tokens/va2_tokens`` are ``[B,V,S,D]``; ``language`` is ``[B,L,D]``.
        Only the active camera/register prefix participates in register layers;
        patch tokens are restored unchanged after attention.
        """
        block1, block2 = self.va1.inter_frame_blocks[layer_idx], self.va2.inter_frame_blocks[layer_idx]
        active1, restore1 = self._active_inter_frame_tokens(va1_tokens, attention_type=attention_type, patch_token_start=int(self.va1.patch_token_start))
        active2, restore2 = self._active_inter_frame_tokens(va2_tokens, attention_type=attention_type, patch_token_start=int(self.va2.patch_token_start))
        q1, k1, v1 = self._attention_qkv(block1, active1)
        q2, k2, v2 = self._attention_qkv(block2, active2)
        if not self.va2_use_language_condition:
            # Configuration-disabled path: both streams remain visual-only.
            # This deliberately avoids constructing/projecting language tokens,
            # while retaining the configured visual stream visibility.
            if self.isolate_vggt_streams:
                mixed1 = self._sdpa(q1, k1, v1)
                mixed2 = self._sdpa(q2, k2, v2)
            else:
                k_joint = torch.cat([k1, k2], dim=2)
                v_joint = torch.cat([v1, v2], dim=2)
                mixed1 = (self._sdpa(q1, k1, v1) if self.vggt_stream_attention == "va2_to_va1"
                          else self._sdpa(q1, k_joint, v_joint))
                mixed2 = self._sdpa(q2, k_joint, v_joint)
            updated1 = self._post_inter_frame_attention(block1, active1, mixed1)
            updated2 = self._post_inter_frame_attention(block2, active2, mixed2)
            return restore1(updated1).contiguous(), restore2(updated2).contiguous(), None, (k1, v1), (k2, v2)
        ql, kl, vl = self._attention_qkv(block2, language)
        if self.isolate_vggt_streams:
            mixed1 = self._sdpa(q1, k1, v1)
            joined2 = torch.cat([active2, language], dim=1)
            q2_all, k2_all, v2_all = self._attention_qkv(block2, joined2)
            mixed2_all = self._sdpa(q2_all, k2_all, v2_all, torch.cat([torch.ones(k2.shape[0], k2.shape[2], dtype=torch.bool, device=k2.device), language_valid], dim=1))
            mixed2 = mixed2_all[:, :active2.shape[1]]
            mixed_lang = mixed2_all[:, active2.shape[1]:]
        else:
            k_joint = torch.cat([k1, k2, kl], dim=2)
            v_joint = torch.cat([v1, v2, vl], dim=2)
            valid_joint = torch.cat([torch.ones(k1.shape[0], k1.shape[2] + k2.shape[2], dtype=torch.bool, device=k1.device), language_valid], dim=1)
            # VA1 never reads VA2 or language in the directional mode.
            # VA2 and its language queries still read the complete joint K/V.
            mixed1 = (self._sdpa(q1, k1, v1) if self.vggt_stream_attention == "va2_to_va1"
                      else self._sdpa(q1, k_joint, v_joint, valid_joint))
            q2_joint = torch.cat([q2, ql], dim=2)
            mixed2_all = self._sdpa(q2_joint, k_joint, v_joint, valid_joint)
            mixed2, mixed_lang = mixed2_all[:, :active2.shape[1]], mixed2_all[:, active2.shape[1]:]
        updated1 = self._post_inter_frame_attention(block1, active1, mixed1)
        updated2_all = self._post_inter_frame_attention(block2, torch.cat([active2, language], dim=1), torch.cat([mixed2, mixed_lang], dim=1))
        updated2, language = updated2_all[:, :active2.shape[1]], updated2_all[:, active2.shape[1]:]
        language = language * language_valid[..., None].to(language.dtype)
        # Removing the language suffix leaves batch stride (V*S + L)*D.
        # For B>1 and V>1, restoring [B,V,S,D] does not remove that gap,
        # so the next frame block cannot view it as [B*V,S,D]. Compact the
        # visual storage here; contiguous() preserves the gradient path.
        return restore1(updated1).contiguous(), restore2(updated2).contiguous(), language, (k1, v1), (k2, v2)

    def _run_dual_language_layer(self, va1_tokens, va2_tokens, language1, language2,
                                language_valid, *, layer_idx, attention_type):
        """Update each stream's visual/language state using explicit visibility.

        The legacy VA1-language-disabled path remains in _run_language_layer.
        Only visual K/V is exposed to the action bridge, including in register
        layers where patches bypass inter-frame attention.
        """
        blocks = (self.va1.inter_frame_blocks[layer_idx], self.va2.inter_frame_blocks[layer_idx])
        sequences, restores, lengths, projected, valid = [], [], [], [], []
        for stream, tokens, language, block in zip(
                (self.va1, self.va2), (va1_tokens, va2_tokens),
                (language1, language2), blocks):
            visual, restore = self._active_inter_frame_tokens(
                tokens, attention_type=attention_type,
                patch_token_start=int(stream.patch_token_start))
            sequence = visual if language is None else torch.cat([visual, language], dim=1)
            key_valid = torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device)
            if language is not None:
                key_valid = torch.cat([key_valid, language_valid], dim=1)
            sequences.append(sequence)
            restores.append(restore)
            lengths.append(visual.shape[1])
            projected.append(self._attention_qkv(block, sequence))
            valid.append(key_valid)

        joint_k = joint_v = joint_valid = None
        if not self.isolate_vggt_streams:
            joint_k = torch.cat([entry[1] for entry in projected], dim=2)
            joint_v = torch.cat([entry[2] for entry in projected], dim=2)
            joint_valid = torch.cat(valid, dim=1)
        visual_outputs, languages, visual_kv = [], [], []
        for index, (block, sequence, (q, k, v), key_valid, restore, length) in enumerate(
                zip(blocks, sequences, projected, valid, restores, lengths)):
            local = self.isolate_vggt_streams or (index == 0 and self.vggt_stream_attention == "va2_to_va1")
            mixed = self._sdpa(q, k if local else joint_k,
                               v if local else joint_v, key_valid if local else joint_valid)
            updated = self._post_inter_frame_attention(block, sequence, mixed)
            visual_outputs.append(restore(updated[:, :length]).contiguous())
            language = updated[:, length:]
            languages.append(None if language.shape[1] == 0 else
                             language.masked_fill(~language_valid[..., None], 0).contiguous())
            visual_kv.append((k[:, :, :length], v[:, :, :length]))
        return (*visual_outputs, *languages, *visual_kv)

    def _forward_impl_reconstruction(self, images, context, language_context_mask):
        """RGB [B,V,3,H,W] -> two [B,Vi,N,D] streams -> visual K/V.

        Shared DINO runs once. Prefixes are attached per stream independently
        of DINO sharing, then camera routing preserves input-camera indices.
        Every layer passes frame output into inter-frame attention; native
        [frame || inter-frame] features [B,Vi,N,2D] feed the geometry heads.
        """
        validate_vggt_images(images, patch_size=int(self.va1.patch_size))
        batch_size, num_views, channels, height, width = images.shape
        if num_views != len(self.input_camera_keys):
            raise ValueError("VGGT input view count does not match configured cameras.")

        def tokenize(aggregator):
            normalized = (images - aggregator._resnet_mean) / aggregator._resnet_std
            patch_dtype = next(aggregator.patch_embed.parameters()).dtype
            patches = aggregator.patch_embed(normalized.to(patch_dtype).reshape(
                batch_size * num_views, channels, height, width))
            if isinstance(patches, Mapping):
                patches = patches["x_norm_patchtokens"]
            return patches.to(next(aggregator.frame_blocks[0].parameters()).dtype)

        patch_tokens = tokenize(self.va1)
        patch2 = patch_tokens if self.share_patch_embedding else tokenize(self.va2)
        tokens1 = self._expand_input_tokens(self.va1, patch_tokens, batch_size=batch_size, num_frames=num_views)
        tokens2 = self._expand_input_tokens(self.va2, patch2, batch_size=batch_size, num_frames=num_views)
        grid = (height // int(self.va1.patch_size), width // int(self.va1.patch_size))
        with torch.no_grad():
            sin, cos = self.va1.rope_embed(H=grid[0], W=grid[1])
        frame_rope = (sin.to(images.device, torch.float32), cos.to(images.device, torch.float32))
        i1 = torch.as_tensor(self.va1_view_indices, device=images.device)
        i2 = torch.as_tensor(self.va2_view_indices, device=images.device)
        va1 = tokens1.index_select(1, i1).contiguous()
        va2 = tokens2.index_select(1, i2).contiguous()
        if self.va1_use_language_condition or self.va2_use_language_condition:
            language, language_valid = self._language(context, language_context_mask, dtype=patch_tokens.dtype, device=images.device)
        else:
            language, language_valid = None, None
        language1 = language.clone() if self.va1_use_language_condition else None
        if not self.va2_use_language_condition:
            language = None
        features1, features2, cache = [None] * int(self.va1.depth), [None] * int(self.va1.depth), {}
        selected = set(self.callback_layer_indices)
        head_layers = {int(self.va1.depth) - 1}
        for route in ("high", "low"):
            head_layers.update(getattr(self.geometry_heads(route)[1], "intermediate_layer_idx", ()))
        for layer_idx in range(int(self.va1.depth)):
            _, frame1 = self.va1._run_frame_block(va1, batch_size, va1.shape[1], va1.shape[2], va1.shape[3], layer_idx, frame_rope)
            _, frame2 = self.va2._run_frame_block(va2, batch_size, va2.shape[1], va2.shape[2], va2.shape[3], layer_idx, frame_rope)
            attention_type = self.va1.inter_frame_attention_types[layer_idx]
            if self.va1_use_language_condition:
                va1, va2, language1, language, kv1, kv2 = self._run_dual_language_layer(
                    frame1, frame2, language1, language, language_valid,
                    layer_idx=layer_idx, attention_type=attention_type)
            else:
                va1, va2, language, kv1, kv2 = self._run_language_layer(frame1, frame2, language, language_valid, layer_idx=layer_idx, attention_type=attention_type)
            if layer_idx in head_layers or layer_idx in selected:
                features1[layer_idx] = torch.cat([frame1, va1], dim=-1)
                features2[layer_idx] = torch.cat([frame2, va2], dim=-1)
            if layer_idx in selected:
                k1, v1 = kv1; k2, v2 = kv2
                if self.action_vggt_source == "va1":
                    action_k, action_v = k1, v1
                elif self.action_vggt_source == "va2":
                    action_k, action_v = k2, v2
                else:
                    action_k, action_v = torch.cat([k1, k2], dim=2), torch.cat([v1, v2], dim=2)
                cache[layer_idx] = {
                    "layer_idx": layer_idx,
                    "attention_type": attention_type,
                    "k": action_k.to(torch.bfloat16 if self.fp32_storage_bf16_compute else images.dtype),
                    "v": action_v.to(torch.bfloat16 if self.fp32_storage_bf16_compute else images.dtype),
                    # Keep both lengths for diagnostics/checkpoint consumers;
                    # action_vggt_source identifies which slice is exposed.
                    "source_seq_lens": {"va1": int(k1.shape[2]), "va2": int(k2.shape[2])},
                    "action_vggt_source": self.action_vggt_source,
                }
        return {"kv_by_layer": cache, "va1_features": features1, "va2_features": features2,
                # Historical names identify branches, not the configured target.
                "current_features": features1, "future_features": features2,
                "patch_token_start": int(self.va1.patch_token_start), "images": images}

    def forward(self, images, *, context=None, language_context_mask=None):
        with (nullcontext() if self.trainable else torch.no_grad()):
            with self._autocast_context(images.device):
                return self._forward_impl_reconstruction(images, context, language_context_mask)["kv_by_layer"]

    def forward_with_reconstruction(self, images, *, context, language_context_mask):
        with (nullcontext() if self.trainable else torch.no_grad()):
            with self._autocast_context(images.device):
                result = self._forward_impl_reconstruction(images, context, language_context_mask)
        # Route RGB shapes along with features (DenseHead uses B, Vi, H, W).
        with torch.autocast(device_type=images.device.type, enabled=False):
            for stream, prefix, route, indices in (("va1", "current", "high", self.va1_view_indices),
                                                   ("va2", "future", "low", self.va2_view_indices)):
                camera_head, dense_head = self.geometry_heads(route)
                features = [None if x is None else x.float() for x in result[f"{stream}_features"]]
                routed_images = images[:, list(indices)].contiguous()
                pose = camera_head(features, patch_token_start=result["patch_token_start"])
                depth, conf = dense_head(features, images=routed_images.float(),
                                         patch_token_start=result["patch_token_start"])
                result.update({f"{prefix}_pose": pose, f"{prefix}_depth": depth,
                               f"{prefix}_depth_conf": conf, f"{stream}_pose": pose,
                               f"{stream}_depth": depth, f"{stream}_depth_conf": conf})
        result.pop("images", None)
        return result


def build_vggt_mot_recon_backbone(*, weights_path=DEFAULT_VGGT_WEIGHTS, aggregator_checkpoint_path=None, callback_layer_indices: Sequence[int], input_camera_keys, va1_camera_keys, va2_camera_keys, trainable=True, isolate_vggt_streams=True, vggt_stream_attention=None, text_dim=4096, va1_use_language_condition=False, va2_use_language_condition=True, action_vggt_source="both", share_patch_embedding=True, share_camera_register_tokens=True, share_geometry_heads=True, head_checkpoint_path=None, aggregator_factory: Callable[..., nn.Module] | None = None):
    """Load both pretrained streams, then apply input/head sharing choices."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not aggregator_checkpoint_path:
        raise ValueError("Multi-rank reconstruction VGGT requires aggregator_checkpoint_path.")
    if aggregator_factory is None:
        from vggt_omega.models.aggregator import Aggregator
        aggregator_factory = Aggregator
    va1, va2 = aggregator_factory(), aggregator_factory()
    checkpoint = aggregator_checkpoint_path or weights_path
    load_aggregator_weights(va1, checkpoint, aggregator_only=aggregator_checkpoint_path is not None)
    load_aggregator_weights(va2, checkpoint, aggregator_only=aggregator_checkpoint_path is not None)
    from vggt_omega.models.heads import CameraHead, DenseHead
    camera_head, dense_head = CameraHead(dim_in=2048), DenseHead(dim_in=2048, patch_size=int(va1.patch_size))
    full = torch.load(head_checkpoint_path or weights_path, map_location="cpu", weights_only=True)
    if isinstance(full, dict) and "state_dict" in full: full = full["state_dict"]
    camera_head.load_state_dict({k[12:]: v for k, v in full.items() if k.startswith("camera_head.")}, strict=True)
    dense_head.load_state_dict({k[11:]: v for k, v in full.items() if k.startswith("dense_head.")}, strict=True)
    return VGGTMoTReconBackbone(va1, va2, callback_layer_indices, input_camera_keys=input_camera_keys, va1_camera_keys=va1_camera_keys, va2_camera_keys=va2_camera_keys, trainable=trainable, isolate_vggt_streams=isolate_vggt_streams, vggt_stream_attention=vggt_stream_attention, text_dim=text_dim, va1_use_language_condition=va1_use_language_condition, va2_use_language_condition=va2_use_language_condition, action_vggt_source=action_vggt_source, share_patch_embedding=share_patch_embedding, share_camera_register_tokens=share_camera_register_tokens, share_geometry_heads=share_geometry_heads, camera_head=camera_head, dense_head=dense_head)


__all__ = ["VGGTMoTReconBackbone", "build_vggt_mot_recon_backbone"]
