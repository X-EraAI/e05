from __future__ import annotations

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
    load_aggregator_weights,
    validate_vggt_images,
)


class VGGTMoTBackbone(nn.Module):
    """Two pretrained VGGT Aggregators with configurable stream visibility.

    Image normalization, patch embedding, camera tokens, register tokens, and
    frame RoPE are computed once.  Configured camera views from the resulting
    ``[B,V,S,D]`` tensor seed two independent transformer streams.  At every
    depth their frame blocks remain independent.  By default, inter-frame
    attention preserves the original bidirectional MoT visibility:

    ``va1_q -> [va1_kv, va2_kv]`` and ``va2_q -> [va1_kv, va2_kv]``.

    With ``isolate_vggt_streams=True``, each query instead reads only its own
    stream's K/V.  Action tokens are deliberately absent from this module in
    either mode.  Selected K/V tensors from both streams are still concatenated
    after their VGGT computation and returned to the one-way ActionDiT bridge.
    """

    SOURCE_NAMES = ("va1", "va2")

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

    def __init__(
        self,
        va1: nn.Module,
        va2: nn.Module,
        callback_layer_indices: Sequence[int],
        *,
        input_camera_keys: Sequence[str],
        va1_camera_keys: Sequence[str],
        va2_camera_keys: Sequence[str],
        trainable: bool = True,
        isolate_vggt_streams: bool = False,
        share_patch_embedding: bool = True,
        share_camera_register_tokens: bool = True,
    ) -> None:
        super().__init__()
        selected = tuple(int(layer) for layer in callback_layer_indices)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("`callback_layer_indices` must be unique and non-empty.")
        if int(va1.depth) != int(va2.depth):
            raise ValueError("va1 and va2 must have the same transformer depth.")
        invalid = set(selected).difference(range(int(va1.depth)))
        if invalid:
            raise ValueError(
                f"callback layers {sorted(invalid)} are outside [0,{int(va1.depth)})."
            )
        for name in ("patch_size", "patch_token_start"):
            if int(getattr(va1, name)) != int(getattr(va2, name)):
                raise ValueError(f"va1 and va2 must share `{name}`.")
        if list(va1.inter_frame_attention_types) != list(va2.inter_frame_attention_types):
            raise ValueError("va1 and va2 inter-frame attention layouts must match.")

        self.va1 = va1
        self.va2 = va2
        self.callback_layer_indices = selected
        self.trainable = bool(trainable)
        # Standalone backbones and older callers retain the legacy dtype path.
        self.fp32_storage_bf16_compute = False
        # False is the compatibility default: configurations and checkpoints
        # created before this switch keep their original bidirectional MoT path.
        self.isolate_vggt_streams = bool(isolate_vggt_streams)

        self.input_camera_keys = self._validate_camera_keys(
            "input_camera_keys", input_camera_keys
        )
        self.va1_camera_keys = self._validate_camera_keys(
            "va1_camera_keys", va1_camera_keys
        )
        self.va2_camera_keys = self._validate_camera_keys(
            "va2_camera_keys", va2_camera_keys
        )
        camera_to_index = {
            camera_key: index
            for index, camera_key in enumerate(self.input_camera_keys)
        }
        for source_name, camera_keys in (
            ("va1", self.va1_camera_keys),
            ("va2", self.va2_camera_keys),
        ):
            missing = tuple(key for key in camera_keys if key not in camera_to_index)
            if missing:
                raise ValueError(
                    f"{source_name}_camera_keys contains cameras absent from "
                    f"input_camera_keys: {missing}."
                )
        self.va1_view_indices = tuple(
            camera_to_index[key] for key in self.va1_camera_keys
        )
        self.va2_view_indices = tuple(
            camera_to_index[key] for key in self.va2_camera_keys
        )

        # Both Aggregators are strictly loaded before construction.  Alias all
        # image/token input modules afterwards so preprocessing is both shared
        # in parameters and executed only once.  Transformer blocks stay fully
        # independent and therefore retain two pretrained expert streams.
        # Defaults preserve historical parameter names/order and aliasing.
        # MoE can opt out; its route-aware input path handles either topology.
        self.share_patch_embedding = bool(share_patch_embedding)
        self.share_camera_register_tokens = bool(share_camera_register_tokens)
        if self.share_patch_embedding:
            self.va2.patch_embed = self.va1.patch_embed
        self.va2.rope_embed = self.va1.rope_embed
        if self.share_camera_register_tokens:
            self.va2.camera_token = self.va1.camera_token
            self.va2.register_token = self.va1.register_token

        self.va1.requires_grad_(self.trainable)
        self.va2.requires_grad_(self.trainable)
        self.train(self.trainable)

    @staticmethod
    def _validate_camera_keys(
        name: str, camera_keys: Sequence[str]
    ) -> tuple[str, ...]:
        keys = tuple(str(key) for key in camera_keys)
        if not keys:
            raise ValueError(f"`{name}` must be non-empty.")
        if any(not key for key in keys):
            raise ValueError(f"`{name}` must not contain empty camera names.")
        if len(set(keys)) != len(keys):
            raise ValueError(f"`{name}` must not contain duplicate camera names.")
        return keys

    @property
    def aggregator(self) -> nn.Module:
        """Compatibility alias used by generic VGGT-aware trainer checks."""
        return self.va1

    def train(self, mode: bool = True):
        effective_mode = bool(mode) if self.trainable else False
        super().train(effective_mode)
        self.va1.train(effective_mode)
        self.va2.train(effective_mode)
        return self

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

    def _run_joint_inter_frame_layer(
        self,
        va1_tokens: torch.Tensor,
        va2_tokens: torch.Tensor,
        *,
        layer_idx: int,
        attention_type: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Run one coupled or isolated va1/va2 inter-frame layer.

        Tensor flow for either global or register-active sequences is:

        ``xi [B,Si,D] -> qi/ki/vi [B,H,Si,Dh]``
        Coupled mode concatenates both streams before SDPA.  Isolated mode runs
        ``SDPA(q1,k1,v1)`` and ``SDPA(q2,k2,v2)`` separately.  Both modes then
        use each expert's own output projection, residual path, and FFN.
        """
        block1 = self.va1.inter_frame_blocks[layer_idx]
        block2 = self.va2.inter_frame_blocks[layer_idx]
        if block1.sample_drop_ratio or block2.sample_drop_ratio:
            raise ValueError("VGGTMoTBackbone does not support non-zero inter-frame drop path.")

        active1, restore1 = self._active_inter_frame_tokens(
            va1_tokens,
            attention_type=attention_type,
            patch_token_start=int(self.va1.patch_token_start),
        )
        active2, restore2 = self._active_inter_frame_tokens(
            va2_tokens,
            attention_type=attention_type,
            patch_token_start=int(self.va2.patch_token_start),
        )
        q1, k1, v1 = self._attention_qkv(block1, active1)
        q2, k2, v2 = self._attention_qkv(block2, active2)
        if (
            q1.shape[0] != q2.shape[0]
            or q1.shape[1] != q2.shape[1]
            or q1.shape[3] != q2.shape[3]
        ):
            raise ValueError(
                "va1/va2 inter-frame Q/K/V batch, head, and head-dim shapes "
                "must match, got "
                f"{tuple(q1.shape)} and {tuple(q2.shape)}."
            )

        if self.isolate_vggt_streams:
            # Strict token-level isolation.  Keeping these as two SDPA calls
            # avoids materializing cross-stream attention scores at all.
            mixed1 = F.scaled_dot_product_attention(q1, k1, v1)
            mixed2 = F.scaled_dot_product_attention(q2, k2, v2)
            mixed1 = mixed1.transpose(1, 2).contiguous().flatten(2)
            mixed2 = mixed2.transpose(1, 2).contiguous().flatten(2)
        else:
            # Legacy MoT behavior: every query reads both visual streams.
            seq_len1 = int(active1.shape[1])
            q_joint = torch.cat([q1, q2], dim=2)
            k_joint = torch.cat([k1, k2], dim=2)
            v_joint = torch.cat([v1, v2], dim=2)
            mixed = F.scaled_dot_product_attention(q_joint, k_joint, v_joint)
            mixed = mixed.transpose(1, 2).contiguous()
            mixed1 = mixed[:, :seq_len1].flatten(2)
            mixed2 = mixed[:, seq_len1:].flatten(2)

        updated1 = self._post_inter_frame_attention(block1, active1, mixed1)
        updated2 = self._post_inter_frame_attention(block2, active2, mixed2)
        return restore1(updated1), restore2(updated2), (k1, v1), (k2, v2)

    def forward(self, images: torch.Tensor) -> dict[int, dict[str, object]]:
        validate_vggt_images(
            images,
            patch_size=int(getattr(self.va1, "patch_size", VGGT_PATCH_SIZE)),
        )
        grad_context = nullcontext() if self.trainable else torch.no_grad()
        with grad_context:
            return self._forward_impl(images)

    def _forward_impl(self, images: torch.Tensor) -> dict[int, dict[str, object]]:
        batch_size, num_frames, channels, height, width = images.shape
        expected_views = len(self.input_camera_keys)
        if num_frames != expected_views:
            raise ValueError(
                "VGGT input view count does not match configured input_camera_keys: "
                f"images has {num_frames} views, configured cameras are "
                f"{self.input_camera_keys}."
        )
        normalized = (images - self.va1._resnet_mean) / self.va1._resnet_std
        flat_images = normalized.reshape(
            batch_size * num_frames, channels, height, width
        )
        patch_tokens = self.va1.patch_embed(flat_images)
        if isinstance(patch_tokens, Mapping):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        shared_tokens = self._expand_input_tokens(
            self.va1,
            patch_tokens,
            batch_size=batch_size,
            num_frames=num_frames,
        )

        grid = (height // int(self.va1.patch_size), width // int(self.va1.patch_size))
        with torch.no_grad():
            rope_sin, rope_cos = self.va1.rope_embed(H=grid[0], W=grid[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        # Tokenize all configured cameras once so their original camera-token
        # positions are preserved, then route the selected views to each expert.
        va1_index = torch.as_tensor(
            self.va1_view_indices, device=shared_tokens.device, dtype=torch.long
        )
        va2_index = torch.as_tensor(
            self.va2_view_indices, device=shared_tokens.device, dtype=torch.long
        )
        va1_tokens = shared_tokens.index_select(1, va1_index)
        va2_tokens = shared_tokens.index_select(1, va2_index)
        va1_num_frames = len(self.va1_view_indices)
        va2_num_frames = len(self.va2_view_indices)
        num_tokens = int(shared_tokens.shape[2])
        embed_dim = int(shared_tokens.shape[3])
        selected = set(self.callback_layer_indices)
        cache: dict[int, dict[str, object]] = {}

        for layer_idx in range(int(self.va1.depth)):
            _, va1_tokens = self.va1._run_frame_block(
                va1_tokens,
                batch_size,
                va1_num_frames,
                num_tokens,
                embed_dim,
                layer_idx,
                frame_rope,
            )
            _, va2_tokens = self.va2._run_frame_block(
                va2_tokens,
                batch_size,
                va2_num_frames,
                num_tokens,
                embed_dim,
                layer_idx,
                frame_rope,
            )
            attention_type = self.va1.inter_frame_attention_types[layer_idx]
            va1_tokens, va2_tokens, va1_kv, va2_kv = self._run_joint_inter_frame_layer(
                va1_tokens,
                va2_tokens,
                layer_idx=layer_idx,
                attention_type=attention_type,
            )
            if layer_idx not in selected:
                continue
            k1, v1 = va1_kv
            k2, v2 = va2_kv
            if not self.trainable:
                k1, v1, k2, v2 = (tensor.detach() for tensor in (k1, v1, k2, v2))
            # The existing one-way bridge consumes one K/V tensor.  Concatenating
            # the source sequence dimension preserves source boundaries in
            # metadata while letting ActionDiT attend to both memories at once.
            cache[layer_idx] = {
                "layer_idx": layer_idx,
                "attention_type": attention_type,
                "k": torch.cat([k1, k2], dim=2),
                "v": torch.cat([v1, v2], dim=2),
                "source_seq_lens": {"va1": int(k1.shape[2]), "va2": int(k2.shape[2])},
            }

        missing = selected.difference(cache)
        if missing:
            raise RuntimeError(f"VGGT MoT backbone did not capture layers: {sorted(missing)}")
        return cache


def build_vggt_mot_backbone(
    *,
    weights_path: str = DEFAULT_VGGT_WEIGHTS,
    aggregator_checkpoint_path: str | None = None,
    callback_layer_indices: Sequence[int],
    input_camera_keys: Sequence[str],
    va1_camera_keys: Sequence[str],
    va2_camera_keys: Sequence[str],
    trainable: bool = True,
    isolate_vggt_streams: bool = False,
    aggregator_factory: Callable[..., nn.Module] | None = None,
) -> VGGTMoTBackbone:
    """Construct two strictly pretrained Aggregators with shared tokenization."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not aggregator_checkpoint_path:
        raise ValueError(
            "Multi-rank VGGT-MoT construction requires `aggregator_checkpoint_path`; "
            "full checkpoint fallback is forbidden."
        )
    if aggregator_factory is None:
        from vggt_omega.models.aggregator import Aggregator

        aggregator_factory = Aggregator
    va1 = aggregator_factory()
    va2 = aggregator_factory()
    checkpoint_path = aggregator_checkpoint_path or weights_path
    aggregator_only = aggregator_checkpoint_path is not None
    load_aggregator_weights(va1, checkpoint_path, aggregator_only=aggregator_only)
    load_aggregator_weights(va2, checkpoint_path, aggregator_only=aggregator_only)
    return VGGTMoTBackbone(
        va1,
        va2,
        callback_layer_indices,
        input_camera_keys=input_camera_keys,
        va1_camera_keys=va1_camera_keys,
        va2_camera_keys=va2_camera_keys,
        trainable=trainable,
        isolate_vggt_streams=isolate_vggt_streams,
    )


__all__ = ["VGGTMoTBackbone", "build_vggt_mot_backbone"]
