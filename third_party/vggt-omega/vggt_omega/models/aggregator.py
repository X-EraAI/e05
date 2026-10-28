# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, TypedDict

import torch
import torch.nn as nn

from vggt_omega.models.layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class AttentionKV(TypedDict):
    layer_idx: int
    attention_type: str
    k: torch.Tensor
    v: torch.Tensor


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

    kv_callback: Callable[[AttentionKV], None] | None

    def __setattr__(self, name: str, value: object) -> None:
        if name == "kv_callback":
            # Callbacks are observers, not model children, even when callable nn.Modules.
            self.__dict__[name] = value
            return
        super().__setattr__(name, value)

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
        kv_callback: Callable[[AttentionKV], None] | None = None,
        callback_layer_indices: tuple[int, ...] = (),
        patch_embed: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.patch_embed = (
            _build_patch_embed(patch_size=patch_size, embed_dim=embed_dim)
            if patch_embed is None
            else patch_embed
        )
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        self.frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )
        self.inter_frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.kv_callback = kv_callback
        self.callback_layer_indices = set(callback_layer_indices)
        invalid_callback_indices = self.callback_layer_indices.difference(range(depth))
        if invalid_callback_indices:
            raise ValueError(f"callback_layer_indices contains invalid block indices {sorted(invalid_callback_indices)}")
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
        self.patch_token_start = 1 + num_register_tokens

        self.inter_frame_attention_types = ["global"] * depth
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)

    def forward(
        self,
        images: torch.Tensor,
        capture_only: bool = False,
        kv_callback: Callable[[AttentionKV], None] | None = None,
        callback_layer_indices: tuple[int, ...] | list[int] | set[int] | None = None,
        detach_callback_kv: bool = True,
    ) -> tuple[list[torch.Tensor | None], int] | None:
        tokens, batch_size, num_frames, patch_grid_size = self._tokenize_images(images)
        outputs, _ = self._forward_blocks(
            tokens,
            batch_size=batch_size,
            num_frames=num_frames,
            patch_grid_size=patch_grid_size,
            capture_only=capture_only,
            kv_callback=kv_callback,
            callback_layer_indices=callback_layer_indices,
            detach_callback_kv=detach_callback_kv,
        )
        if capture_only:
            return None
        return outputs, self.patch_token_start

    def forward_with_state(
        self,
        images: torch.Tensor,
        kv_callback: Callable[[AttentionKV], None] | None = None,
        callback_layer_indices: tuple[int, ...] | list[int] | set[int] | None = None,
        detach_callback_kv: bool = True,
    ) -> tuple[list[torch.Tensor | None], int, torch.Tensor, tuple[int, int]]:
        """Run the normal image pass and also expose its recurrent token state.

        The legacy :meth:`forward` contract remains unchanged.  Loop consumers
        use this opt-in method to obtain the final inter-frame tokens with shape
        ``[B,V,N,D]`` and the patch grid needed to rebuild frame RoPE later.
        """
        tokens, batch_size, num_frames, patch_grid_size = self._tokenize_images(images)
        outputs, final_tokens = self._forward_blocks(
            tokens,
            batch_size=batch_size,
            num_frames=num_frames,
            patch_grid_size=patch_grid_size,
            capture_only=False,
            kv_callback=kv_callback,
            callback_layer_indices=callback_layer_indices,
            detach_callback_kv=detach_callback_kv,
        )
        assert outputs is not None
        return outputs, self.patch_token_start, final_tokens, patch_grid_size

    def forward_from_state(
        self,
        tokens: torch.Tensor,
        patch_grid_size: tuple[int, int],
        capture_only: bool = True,
        kv_callback: Callable[[AttentionKV], None] | None = None,
        callback_layer_indices: tuple[int, ...] | list[int] | set[int] | None = None,
        detach_callback_kv: bool = True,
    ) -> tuple[list[torch.Tensor | None] | None, torch.Tensor]:
        """Re-enter block 0 from a previous pass's final inter-frame tokens.

        ``tokens`` includes camera, register, and patch tokens.  Consequently
        this path deliberately skips image normalization, patch embedding, and
        special-token injection while applying all frame/inter-frame blocks.
        """
        if tokens.ndim != 4:
            raise ValueError(
                "Recurrent VGGT tokens must be [B,V,N,D], "
                f"got {tuple(tokens.shape)}."
            )
        batch_size, num_frames, num_tokens, embed_dim = tokens.shape
        grid_h, grid_w = (int(value) for value in patch_grid_size)
        expected_tokens = self.patch_token_start + grid_h * grid_w
        expected_dim = int(self.camera_token.shape[-1])
        if batch_size < 1 or num_frames < 1 or grid_h < 1 or grid_w < 1:
            raise ValueError(
                "Recurrent VGGT state requires positive batch/view/grid dimensions."
            )
        if num_tokens != expected_tokens or embed_dim != expected_dim:
            raise ValueError(
                "Recurrent VGGT state shape mismatch: expected token/dim "
                f"({expected_tokens},{expected_dim}), got ({num_tokens},{embed_dim})."
            )
        return self._forward_blocks(
            tokens,
            batch_size=batch_size,
            num_frames=num_frames,
            patch_grid_size=(grid_h, grid_w),
            capture_only=capture_only,
            kv_callback=kv_callback,
            callback_layer_indices=callback_layer_indices,
            detach_callback_kv=detach_callback_kv,
        )

    def _tokenize_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, tuple[int, int]]:
        if images.ndim != 5:
            raise ValueError(f"Expected images [B,V,3,H,W], got {tuple(images.shape)}")
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        normalized = (images - self._resnet_mean) / self._resnet_std
        normalized = normalized.view(batch_size * num_frames, num_channels, height, width)
        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)
        patch_tokens = self.patch_embed(normalized)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        tokens = tokens.view(batch_size, num_frames, tokens.shape[1], tokens.shape[2])
        return tokens, batch_size, num_frames, (height // self.patch_size, width // self.patch_size)

    def _forward_blocks(
        self,
        tokens: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        patch_grid_size: tuple[int, int],
        capture_only: bool,
        kv_callback: Callable[[AttentionKV], None] | None,
        callback_layer_indices: tuple[int, ...] | list[int] | set[int] | None,
        detach_callback_kv: bool,
    ) -> tuple[list[torch.Tensor | None] | None, torch.Tensor]:
        active_callback = self.kv_callback if kv_callback is None else kv_callback
        active_layer_indices = (
            self.callback_layer_indices
            if callback_layer_indices is None
            else set(callback_layer_indices)
        )
        invalid_callback_indices = set(active_layer_indices).difference(range(self.depth))
        if invalid_callback_indices:
            raise ValueError(
                f"callback_layer_indices contains invalid block indices {sorted(invalid_callback_indices)}"
            )

        num_tokens, embed_dim = int(tokens.shape[2]), int(tokens.shape[3])
        tokens = tokens.reshape(batch_size * num_frames, num_tokens, embed_dim)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=tokens.device, dtype=torch.float32),
                rope_cos.to(device=tokens.device, dtype=torch.float32),
            )

        outputs: list[torch.Tensor | None] = []
        for block_idx in range(self.depth):
            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
            )
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
                kv_callback=active_callback,
                callback_layer_indices=active_layer_indices,
                detach_callback_kv=detach_callback_kv,
            )
            if capture_only:
                continue
            if block_idx in self.cached_layer_indices:
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
            else:
                outputs.append(None)

        final_tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
        return (None if capture_only else outputs), final_tokens

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        tokens = self.frame_blocks[block_idx](tokens, rope_sincos)
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
        kv_callback: Callable[[AttentionKV], None] | None,
        callback_layer_indices: set[int],
        detach_callback_kv: bool,
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
        callback = None
        if kv_callback is not None and block_idx in callback_layer_indices:
            def callback(k: torch.Tensor, v: torch.Tensor) -> None:
                kv_callback(
                    {
                        "layer_idx": block_idx,
                        "attention_type": attention_type,
                        "k": k,
                        "v": v,
                    }
                )

        if attention_type == "global":
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            tokens = self.inter_frame_blocks[block_idx](
                tokens,
                None,
                kv_callback=callback,
                detach_callback_kv=detach_callback_kv,
            )
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:].reshape(
            batch_size,
            num_frames * (num_tokens - patch_token_start),
            embed_dim,
        )

        camera_and_register_tokens = self.inter_frame_blocks[block_idx](
            camera_and_register_tokens,
            None,
            kv_callback=callback,
            detach_callback_kv=detach_callback_kv,
        )
        tokens = torch.cat([camera_and_register_tokens, patch_tokens], dim=1)

        camera_and_register_tokens = tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, num_frames * patch_token_start :].view(
            batch_size,
            num_frames,
            num_tokens - patch_token_start,
            embed_dim,
        )
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2)


def _build_patch_embed(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model


def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])
