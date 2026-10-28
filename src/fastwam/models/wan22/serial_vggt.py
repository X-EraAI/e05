from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from .vggt_mot import VGGT_PATCH_SIZE, load_aggregator_weights, validate_vggt_images


class _BlockOnlyAggregator(nn.Module):
    """The block portion of VGGT Aggregator, without image/token input modules."""

    def __init__(self, template: nn.Module) -> None:
        super().__init__()
        # Constructing a tiny Aggregator and retaining only its transformer blocks
        # keeps the exact VGGT block classes and checkpoint parameter names.
        self.frame_blocks = template.frame_blocks
        self.inter_frame_blocks = template.inter_frame_blocks
        self.inter_frame_attention_types = list(template.inter_frame_attention_types)
        self.depth = int(template.depth)
        self.patch_size = int(template.patch_size)
        self.patch_token_start = int(template.patch_token_start)
        self.rope_embed = template.rope_embed

    def forward_blocks(
        self,
        tokens: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        frame_rope: tuple[torch.Tensor, torch.Tensor],
        callback,
        callback_layer_offset: int,
        detach_callback_kv: bool,
    ) -> dict[int, dict[str, torch.Tensor | int]]:
        """Run va2 frame/inter-frame blocks on `[B,V,S,D]` tokens."""
        _, _, num_tokens, embed_dim = tokens.shape
        cache: dict[int, dict[str, torch.Tensor | int]] = {}
        x = tokens
        # In mode ``block0_plus_final_tokens`` both token segments retain the
        # original camera/register prefix, so register attention must use the
        # doubled prefix length.
        patch_token_start = self.patch_token_start
        for block_idx in range(self.depth):
            x, _ = self._run_frame_block(x, batch_size, num_frames, num_tokens, embed_dim, block_idx, frame_rope)
            x = self._run_inter_frame_block(
                x,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
                callback=callback,
                callback_layer_idx=callback_layer_offset + block_idx,
                detach_callback_kv=detach_callback_kv,
                patch_token_start=patch_token_start,
                cache=cache,
            )
        return cache

    def _run_frame_block(self, tokens, batch_size, num_frames, num_tokens, embed_dim, block_idx, rope):
        flat = tokens.reshape(batch_size * num_frames, num_tokens, embed_dim)
        out = self.frame_blocks[block_idx](flat, rope)
        return out.reshape(batch_size, num_frames, num_tokens, embed_dim), out

    def _run_inter_frame_block(
        self, tokens, batch_size, num_frames, num_tokens, embed_dim, block_idx,
        attention_type, *, callback, callback_layer_idx, detach_callback_kv,
        patch_token_start, cache,
    ):
        tokens = tokens.reshape(batch_size, num_frames, num_tokens, embed_dim)
        local_callback = None
        if callback is not None:
            def local_callback(k, v):
                callback({
                    "layer_idx": callback_layer_idx,
                    "attention_type": attention_type,
                    "k": k,
                    "v": v,
                })
        if attention_type == "global":
            flat = tokens.reshape(batch_size, num_frames * num_tokens, embed_dim)
            out = self.inter_frame_blocks[block_idx](
                flat, None, kv_callback=local_callback,
                detach_callback_kv=detach_callback_kv,
            )
            return out.reshape(batch_size, num_frames, num_tokens, embed_dim)
        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")
        prefix = tokens[:, :, :patch_token_start].reshape(batch_size, num_frames * patch_token_start, embed_dim)
        patch = tokens[:, :, patch_token_start:].reshape(batch_size, num_frames * (num_tokens - patch_token_start), embed_dim)
        prefix = self.inter_frame_blocks[block_idx](
            prefix, None, kv_callback=local_callback,
            detach_callback_kv=detach_callback_kv,
        )
        merged = torch.cat([prefix, patch], dim=1)
        return merged.reshape(batch_size, num_frames, num_tokens, embed_dim)


class SerialVGGTBackbone(nn.Module):
    """Two serial VGGT Aggregators exposing 48 inter-frame K/V memories.

    va1 performs normal image preprocessing. Its initial and final complete
    token tensors feed va2, whose frame/inter-frame blocks run directly on
    those tokens. Returned callback keys 0..23 belong to va1 and 24..47 to va2.
    """

    VALID_INPUT_MODES = frozenset({"final_tokens", "block0_plus_final_tokens"})

    def __init__(self, va1: nn.Module, va2: _BlockOnlyAggregator, *, va2_input_mode: str, trainable: bool) -> None:
        super().__init__()
        if va2_input_mode not in self.VALID_INPUT_MODES:
            raise ValueError(f"Invalid va2_input_mode={va2_input_mode!r}; expected one of {sorted(self.VALID_INPUT_MODES)}")
        self.va1 = va1
        self.va2 = va2
        self.va2_input_mode = va2_input_mode
        self.trainable = bool(trainable)
        self.aggregator = self.va1  # trainer/checkpoint compatibility
        self.va1.requires_grad_(self.trainable)
        self.va2.requires_grad_(self.trainable)
        self.train(self.trainable)

    def train(self, mode: bool = True):
        super().train(bool(mode) if self.trainable else False)
        self.va1.train(bool(mode) if self.trainable else False)
        self.va2.train(bool(mode) if self.trainable else False)
        return self

    def forward(self, images: torch.Tensor) -> dict[int, dict[str, torch.Tensor | int]]:
        validate_vggt_images(images, patch_size=int(getattr(self.va1, "patch_size", VGGT_PATCH_SIZE)))
        grad_context = nullcontext() if self.trainable else torch.no_grad()
        with grad_context:
            return self._forward_impl(images)

    def _forward_impl(self, images: torch.Tensor):
        b, v, c, h, w = images.shape
        images_n = (images - self.va1._resnet_mean) / self.va1._resnet_std
        flat_images = images_n.reshape(b * v, c, h, w)
        patch_tokens = self.va1.patch_embed(flat_images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        camera = self.va1.camera_token[:, 0:1].expand(b, 1, *self.va1.camera_token.shape[2:])
        camera = torch.cat([camera, self.va1.camera_token[:, 1:].expand(b, v - 1, *self.va1.camera_token.shape[2:])], dim=1)
        camera = camera.reshape(b * v, 1, -1)
        register = self.va1.register_token[:, 0:1].expand(b, 1, *self.va1.register_token.shape[2:])
        register = torch.cat([register, self.va1.register_token[:, 1:].expand(b, v - 1, *self.va1.register_token.shape[2:])], dim=1)
        register = register.reshape(b * v, register.shape[2], register.shape[3])
        tokens = torch.cat([camera, register, patch_tokens], dim=1)
        _, num_tokens, dim = tokens.shape
        grid = (h // self.va1.patch_size, w // self.va1.patch_size)
        with torch.no_grad():
            sin, cos = self.va1.rope_embed(H=grid[0], W=grid[1])
            frame_rope = (sin.to(patch_tokens.device, torch.float32), cos.to(patch_tokens.device, torch.float32))
        x = tokens.reshape(b, v, num_tokens, dim)
        block0_input = x
        cache: dict[int, dict[str, torch.Tensor | int]] = {}
        def capture(event):
            idx = int(event["layer_idx"])
            if idx in cache:
                raise RuntimeError(f"Duplicate VGGT callback layer {idx}")
            cache[idx] = {"layer_idx": idx, "k": event["k"], "v": event["v"]}
        for idx in range(self.va1.depth):
            x, _ = self.va1._run_frame_block(x.reshape(b, v, num_tokens, dim), b, v, num_tokens, dim, idx, frame_rope)
            x = self.va1._run_inter_frame_attention_block(
                x, b, v, num_tokens, dim, idx,
                self.va1.inter_frame_attention_types[idx],
                kv_callback=capture, callback_layer_indices={idx},
                detach_callback_kv=not self.trainable,
            )
            x = x.reshape(b, v, num_tokens, dim)
        va2_input = x if self.va2_input_mode == "final_tokens" else torch.cat([block0_input, x], dim=2)
        va2 = self.va2
        va2.patch_token_start = int(self.va1.patch_token_start) * (1 if self.va2_input_mode == "final_tokens" else 2)
        va2_tokens = va2_input.shape[2]
        patch_count = va2_tokens - va2.patch_token_start
        va2_sin, va2_cos = va2.rope_embed(H=1, W=patch_count)
        va2_rope = (va2_sin.to(x.device, torch.float32), va2_cos.to(x.device, torch.float32))
        va2_cache = va2.forward_blocks(
            va2_input, batch_size=b, num_frames=v, frame_rope=va2_rope,
            callback=capture, callback_layer_offset=self.va1.depth,
            detach_callback_kv=not self.trainable,
        )
        cache.update(va2_cache)
        expected = set(range(int(self.va1.depth) * 2))
        missing = expected - set(cache)
        if missing:
            raise RuntimeError(f"Serial VGGT callback missing layers: {sorted(missing)}")
        return cache


def build_serial_vggt_backbone(
    *, weights_path: str, aggregator_checkpoint_path: str | None,
    va2_input_mode: str = "block0_plus_final_tokens",
    trainable: bool = True,
    aggregator_factory=None,
) -> SerialVGGTBackbone:
    if aggregator_factory is None:
        from vggt_omega.models.aggregator import Aggregator
        aggregator_factory = Aggregator
    va1 = aggregator_factory()
    load_aggregator_weights(va1, aggregator_checkpoint_path or weights_path, aggregator_only=aggregator_checkpoint_path is not None)
    # va2 gets independent block parameters initialized from the same pretrained
    # state; no patch/token modules are retained in its state dict.
    va2_template = aggregator_factory()
    va2 = _BlockOnlyAggregator(va2_template)
    va2.load_state_dict({k: v for k, v in va1.state_dict().items() if k.startswith(("frame_blocks.", "inter_frame_blocks.", "rope_embed."))}, strict=False)
    return SerialVGGTBackbone(va1, va2, va2_input_mode=va2_input_mode, trainable=trainable)


__all__ = ["SerialVGGTBackbone", "build_serial_vggt_backbone"]
