"""Layer-exact Gen3R GeometryAdapter with FastWAM/Wan2.2 I/O shapes.

The encoder/decoder topology and chunked causal execution mirror the vendored
Gen3R implementation in ``third_party/Gen3R``.  Only shape-bound resampling,
the latent width, and the Wan latent statistics are parameterized:

``[B,10240,1+4k,24,20] -> [B,48,k+1,24,20] -> [B,10240,1+4k,24,20]``.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import os
from pathlib import Path
from typing import Any
import warnings

from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

from .vggt_gen3r_geometry_adapter import DiagonalGaussianPosterior
from .vggt_gen3r_exact_features import VGGT_GEN3R_EXACT_GROUPING_CONTRACT


GEN3R_EXACT_CACHE_T = 2
GEN3R_EXACT_FORMAT_VERSION = 4
GEN3R_EXACT_NATIVE_SUPERVISION_CONTRACT = (
    "per_timestep_head_camera_per_timestep_mean_valid_point_scale_v1"
)
GEN3R_EXACT_LEGACY_EVAL_ENV = "FASTWAM_ALLOW_LEGACY_GEN3R_EVAL"
GEN3R_EXACT_DIM_MULT = (4, 2, 1, 1)
GEN3R_EXACT_TEMPORAL_DOWNSAMPLE = (False, True, True)
WAN22_VAE_LATENT_MEAN = (
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
)
WAN22_VAE_LATENT_STD = (
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
)


def _parse_pair(value: Sequence[int], *, name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"`{name}` must contain [H,W], got {value}.")
    parsed = tuple(map(int, value))
    if min(parsed) <= 0:
        raise ValueError(f"`{name}` must be positive, got {parsed}.")
    return parsed


class ExactCausalConv3d(nn.Conv3d):
    """Gen3R's left-padded Conv3d, including streaming cache semantics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._causal_padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(
        self,
        inputs: torch.Tensor,
        cache_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        padding = list(self._causal_padding)
        if cache_inputs is not None and padding[4] > 0:
            cache_inputs = cache_inputs.to(device=inputs.device, dtype=inputs.dtype)
            inputs = torch.cat([cache_inputs, inputs], dim=2)
            padding[4] -= int(cache_inputs.shape[2])
            if padding[4] < 0:
                raise ValueError("Causal cache is longer than the configured temporal padding.")
        return super().forward(F.pad(inputs, padding))


class ExactRMSNorm(nn.Module):
    """Gen3R RMS_norm without changing its normalization arithmetic."""

    def __init__(
        self,
        dim: int,
        *,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (int(dim), *broadcastable_dims) if channel_first else (int(dim),)
        self.channel_first = bool(channel_first)
        self.scale = int(dim) ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(inputs, dim=1 if self.channel_first else -1)
        return normalized * self.scale * self.gamma + self.bias


class ExactUpsample(nn.Upsample):
    """Gen3R's BF16-safe nearest-exact interpolation."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs.float()).to(dtype=inputs.dtype)


class ExactResample(nn.Module):
    """Gen3R Resample with only its shape-bound endpoint sizes parameterized."""

    VALID_MODES = {
        "none",
        "decode_block_2d",
        "decode_block_3d",
        "encode_block_2d",
        "encode_block_3d",
        "input_upsample2d",
        "output_downsample2d",
    }

    def __init__(
        self,
        dim: int,
        mode: str,
        *,
        channel_resample_scale: int,
        input_grid_size: tuple[int, int],
        latent_grid_size: tuple[int, int],
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported exact Gen3R resample mode: {mode}.")
        self.dim = int(dim)
        self.mode = str(mode)
        self.channel_resample_scale = int(channel_resample_scale)
        if self.channel_resample_scale <= 0:
            raise ValueError("`channel_resample_scale` must be positive.")

        if mode == "decode_block_2d":
            self.resample = nn.Sequential(nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == "decode_block_3d":
            self.resample = nn.Sequential(nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = ExactCausalConv3d(
                dim,
                dim * 2,
                (3, 1, 1),
                padding=(1, 0, 0),
            )
        elif mode == "encode_block_2d":
            self.resample = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1))
        elif mode == "encode_block_3d":
            self.resample = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1))
            self.time_conv = ExactCausalConv3d(
                dim,
                dim,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=(0, 0, 0),
            )
        elif mode == "input_upsample2d":
            if dim % self.channel_resample_scale != 0:
                raise ValueError(
                    f"Input channels {dim} must be divisible by "
                    f"channel_resample_scale={self.channel_resample_scale}."
                )
            self.resample = nn.Sequential(
                (
                    nn.Identity()
                    if input_grid_size == latent_grid_size
                    else ExactUpsample(size=latent_grid_size, mode="nearest-exact")
                ),
                nn.Conv2d(dim, dim // self.channel_resample_scale, 3, padding=1),
            )
        elif mode == "output_downsample2d":
            self.resample = nn.Sequential(
                ExactUpsample(
                    size=(2 * input_grid_size[0], 2 * input_grid_size[1]),
                    mode="nearest-exact",
                ),
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(
                    dim,
                    dim * self.channel_resample_scale,
                    3,
                    stride=(2, 2),
                ),
            )
        else:
            self.resample = nn.Identity()

    def forward(
        self,
        inputs: torch.Tensor,
        feature_cache: list[torch.Tensor | str | None] | None = None,
        feature_index: list[int] | None = None,
    ) -> torch.Tensor:
        batch_size, channels, num_steps, height, width = inputs.shape
        if feature_cache is not None and feature_index is None:
            raise ValueError("`feature_index` is required with `feature_cache`.")

        if self.mode == "decode_block_3d" and feature_cache is not None:
            cache_index = feature_index[0]
            cached = feature_cache[cache_index]
            if cached is None:
                feature_cache[cache_index] = "Rep"
                feature_index[0] += 1
            else:
                cache_inputs = inputs[:, :, -GEN3R_EXACT_CACHE_T:].clone()
                if int(cache_inputs.shape[2]) < 2 and cached not in (None, "Rep"):
                    if not isinstance(cached, torch.Tensor):
                        raise TypeError("Invalid tensor cache in exact temporal upsample.")
                    cache_inputs = torch.cat(
                        [cached[:, :, -1:].to(cache_inputs.device), cache_inputs],
                        dim=2,
                    )
                if int(cache_inputs.shape[2]) < 2 and cached == "Rep":
                    cache_inputs = torch.cat([torch.zeros_like(cache_inputs), cache_inputs], dim=2)
                if cached == "Rep":
                    inputs = self.time_conv(inputs)
                else:
                    if not isinstance(cached, torch.Tensor):
                        raise TypeError("Invalid tensor cache in exact temporal upsample.")
                    inputs = self.time_conv(inputs, cached)
                feature_cache[cache_index] = cache_inputs
                feature_index[0] += 1
                inputs = rearrange(
                    inputs.reshape(batch_size, 2, channels, num_steps, height, width),
                    "b r c t h w -> b c (t r) h w",
                ).contiguous()

        current_steps = int(inputs.shape[2])
        inputs = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
        inputs = self.resample(inputs)
        inputs = rearrange(
            inputs,
            "(b t) c h w -> b c t h w",
            b=batch_size,
            t=current_steps,
        ).contiguous()

        if self.mode == "encode_block_3d" and feature_cache is not None:
            cache_index = feature_index[0]
            cached = feature_cache[cache_index]
            if cached is None:
                feature_cache[cache_index] = inputs.clone()
                feature_index[0] += 1
            else:
                if not isinstance(cached, torch.Tensor):
                    raise TypeError("Invalid tensor cache in exact temporal downsample.")
                cache_inputs = inputs[:, :, -1:].clone()
                inputs = self.time_conv(
                    torch.cat([cached[:, :, -1:].to(inputs.device), inputs], dim=2)
                )
                feature_cache[cache_index] = cache_inputs
                feature_index[0] += 1
        return inputs


class ExactResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            ExactRMSNorm(input_dim, images=False),
            nn.SiLU(),
            ExactCausalConv3d(input_dim, output_dim, 3, padding=1),
            ExactRMSNorm(output_dim, images=False),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            ExactCausalConv3d(output_dim, output_dim, 3, padding=1),
        )
        self.shortcut = (
            ExactCausalConv3d(input_dim, output_dim, 1)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(
        self,
        inputs: torch.Tensor,
        feature_cache: list[torch.Tensor | str | None] | None = None,
        feature_index: list[int] | None = None,
    ) -> torch.Tensor:
        shortcut = self.shortcut(inputs)
        hidden = inputs
        for layer in self.residual:
            if isinstance(layer, ExactCausalConv3d) and feature_cache is not None:
                if feature_index is None:
                    raise ValueError("`feature_index` is required with `feature_cache`.")
                cache_index = feature_index[0]
                cached = feature_cache[cache_index]
                if cached is not None and not isinstance(cached, torch.Tensor):
                    raise TypeError("Residual causal cache must be a tensor or None.")
                cache_inputs = hidden[:, :, -GEN3R_EXACT_CACHE_T:].clone()
                if int(cache_inputs.shape[2]) < 2 and cached is not None:
                    cache_inputs = torch.cat(
                        [cached[:, :, -1:].to(cache_inputs.device), cache_inputs],
                        dim=2,
                    )
                hidden = layer(hidden, cached)
                feature_cache[cache_index] = cache_inputs
                feature_index[0] += 1
            else:
                hidden = layer(hidden)
        return hidden + shortcut


class ExactAttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = ExactRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        batch_size, channels, num_steps, height, width = inputs.shape
        hidden = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
        hidden = self.norm(hidden)
        query, key, value = (
            self.to_qkv(hidden)
            .reshape(batch_size * num_steps, 1, channels * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        hidden = F.scaled_dot_product_attention(query, key, value)
        hidden = hidden.squeeze(1).permute(0, 2, 1).reshape(
            batch_size * num_steps,
            channels,
            height,
            width,
        )
        hidden = self.proj(hidden)
        hidden = rearrange(
            hidden,
            "(b t) c h w -> b c t h w",
            b=batch_size,
            t=num_steps,
        ).contiguous()
        return hidden + identity


def _count_exact_causal_convs(module: nn.Module) -> int:
    return sum(isinstance(child, ExactCausalConv3d) for child in module.modules())


class ExactEncoder3d(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        hidden_dim: int,
        posterior_channels: int,
        channel_resample_scale: int,
        input_grid_size: tuple[int, int],
        latent_grid_size: tuple[int, int],
        dim_mult: tuple[int, ...],
        num_res_blocks: int,
        attention_scales: tuple[float, ...],
        temporal_downsample: tuple[bool, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        dims = [hidden_dim * multiplier for multiplier in (4, *dim_mult)]
        self.init_block = ExactResample(
            input_channels,
            "input_upsample2d",
            channel_resample_scale=channel_resample_scale,
            input_grid_size=input_grid_size,
            latent_grid_size=latent_grid_size,
        )
        self.conv1 = ExactCausalConv3d(
            input_channels // channel_resample_scale,
            dims[0],
            3,
            padding=1,
        )

        blocks: list[nn.Module] = []
        scale = 1.0
        output_dim = dims[0]
        for stage_index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                blocks.append(ExactResidualBlock(input_dim, output_dim, dropout))
                if scale in attention_scales:
                    blocks.append(ExactAttentionBlock(output_dim))
                input_dim = output_dim
            if stage_index != len(dim_mult) - 1:
                mode = (
                    "encode_block_3d"
                    if temporal_downsample[stage_index]
                    else "encode_block_2d"
                )
                blocks.append(
                    ExactResample(
                        output_dim,
                        mode,
                        channel_resample_scale=channel_resample_scale,
                        input_grid_size=input_grid_size,
                        latent_grid_size=latent_grid_size,
                    )
                )
                scale /= 2.0
        self.downsamples = nn.Sequential(*blocks)
        self.middle = nn.Sequential(
            ExactResidualBlock(output_dim, output_dim, dropout),
            ExactAttentionBlock(output_dim),
            ExactResidualBlock(output_dim, output_dim, dropout),
        )
        self.head = nn.Sequential(
            ExactRMSNorm(output_dim, images=False),
            nn.SiLU(),
            ExactCausalConv3d(output_dim, posterior_channels, 3, padding=1),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        feature_cache: list[torch.Tensor | str | None] | None = None,
        feature_index: list[int] | None = None,
    ) -> torch.Tensor:
        hidden = self.init_block(inputs)
        if feature_cache is not None:
            if feature_index is None:
                raise ValueError("`feature_index` is required with `feature_cache`.")
            cache_index = feature_index[0]
            cached = feature_cache[cache_index]
            if cached is not None and not isinstance(cached, torch.Tensor):
                raise TypeError("Encoder input causal cache must be a tensor or None.")
            cache_inputs = hidden[:, :, -GEN3R_EXACT_CACHE_T:].clone()
            if int(cache_inputs.shape[2]) < 2 and cached is not None:
                cache_inputs = torch.cat(
                    [cached[:, :, -1:].to(cache_inputs.device), cache_inputs],
                    dim=2,
                )
            hidden = self.conv1(hidden, cached)
            feature_cache[cache_index] = cache_inputs
            feature_index[0] += 1
        else:
            hidden = self.conv1(hidden)

        for layer in self.downsamples:
            if feature_cache is not None:
                hidden = layer(hidden, feature_cache, feature_index)
            else:
                hidden = layer(hidden)
        for layer in self.middle:
            if isinstance(layer, ExactResidualBlock) and feature_cache is not None:
                hidden = layer(hidden, feature_cache, feature_index)
            else:
                hidden = layer(hidden)
        for layer in self.head:
            if isinstance(layer, ExactCausalConv3d) and feature_cache is not None:
                cache_index = feature_index[0]
                cached = feature_cache[cache_index]
                if cached is not None and not isinstance(cached, torch.Tensor):
                    raise TypeError("Encoder head causal cache must be a tensor or None.")
                cache_inputs = hidden[:, :, -GEN3R_EXACT_CACHE_T:].clone()
                if int(cache_inputs.shape[2]) < 2 and cached is not None:
                    cache_inputs = torch.cat(
                        [cached[:, :, -1:].to(cache_inputs.device), cache_inputs],
                        dim=2,
                    )
                hidden = layer(hidden, cached)
                feature_cache[cache_index] = cache_inputs
                feature_index[0] += 1
            else:
                hidden = layer(hidden)
        return hidden


class ExactDecoder3d(nn.Module):
    def __init__(
        self,
        *,
        output_channels: int,
        hidden_dim: int,
        latent_channels: int,
        channel_resample_scale: int,
        input_grid_size: tuple[int, int],
        latent_grid_size: tuple[int, int],
        dim_mult: tuple[int, ...],
        num_res_blocks: int,
        attention_scales: tuple[float, ...],
        temporal_upsample: tuple[bool, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        dims = [hidden_dim * multiplier for multiplier in (dim_mult[-1], *dim_mult[::-1])]
        self.conv1 = ExactCausalConv3d(latent_channels, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ExactResidualBlock(dims[0], dims[0], dropout),
            ExactAttentionBlock(dims[0]),
            ExactResidualBlock(dims[0], dims[0], dropout),
        )

        blocks: list[nn.Module] = []
        scale = 1.0 / 2 ** (len(dim_mult) - 2)
        output_dim = dims[0]
        for stage_index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if stage_index in (1, 2, 3):
                input_dim //= 2
            for _ in range(num_res_blocks + 1):
                blocks.append(ExactResidualBlock(input_dim, output_dim, dropout))
                if scale in attention_scales:
                    blocks.append(ExactAttentionBlock(output_dim))
                input_dim = output_dim
            if stage_index != len(dim_mult) - 1:
                mode = (
                    "decode_block_3d"
                    if temporal_upsample[stage_index]
                    else "decode_block_2d"
                )
                blocks.append(
                    ExactResample(
                        output_dim,
                        mode,
                        channel_resample_scale=channel_resample_scale,
                        input_grid_size=input_grid_size,
                        latent_grid_size=latent_grid_size,
                    )
                )
                scale *= 2.0
        self.upsamples = nn.Sequential(*blocks)
        if output_channels % channel_resample_scale != 0:
            raise ValueError(
                f"Output channels {output_channels} must be divisible by "
                f"channel_resample_scale={channel_resample_scale}."
            )
        bottleneck_channels = output_channels // channel_resample_scale
        self.head = nn.Sequential(
            ExactRMSNorm(output_dim, images=False),
            nn.SiLU(),
            ExactCausalConv3d(output_dim, bottleneck_channels, 3, padding=1),
        )
        self.out_block = ExactResample(
            bottleneck_channels,
            "output_downsample2d",
            channel_resample_scale=channel_resample_scale,
            input_grid_size=input_grid_size,
            latent_grid_size=latent_grid_size,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        feature_cache: list[torch.Tensor | str | None] | None = None,
        feature_index: list[int] | None = None,
    ) -> torch.Tensor:
        hidden = inputs
        if feature_cache is not None:
            if feature_index is None:
                raise ValueError("`feature_index` is required with `feature_cache`.")
            cache_index = feature_index[0]
            cached = feature_cache[cache_index]
            if cached is not None and not isinstance(cached, torch.Tensor):
                raise TypeError("Decoder input causal cache must be a tensor or None.")
            cache_inputs = hidden[:, :, -GEN3R_EXACT_CACHE_T:].clone()
            if int(cache_inputs.shape[2]) < 2 and cached is not None:
                cache_inputs = torch.cat(
                    [cached[:, :, -1:].to(cache_inputs.device), cache_inputs],
                    dim=2,
                )
            hidden = self.conv1(hidden, cached)
            feature_cache[cache_index] = cache_inputs
            feature_index[0] += 1
        else:
            hidden = self.conv1(hidden)

        for layer in self.middle:
            if isinstance(layer, ExactResidualBlock) and feature_cache is not None:
                hidden = layer(hidden, feature_cache, feature_index)
            else:
                hidden = layer(hidden)
        for layer in self.upsamples:
            if feature_cache is not None:
                hidden = layer(hidden, feature_cache, feature_index)
            else:
                hidden = layer(hidden)
        for layer in self.head:
            if isinstance(layer, ExactCausalConv3d) and feature_cache is not None:
                cache_index = feature_index[0]
                cached = feature_cache[cache_index]
                if cached is not None and not isinstance(cached, torch.Tensor):
                    raise TypeError("Decoder head causal cache must be a tensor or None.")
                cache_inputs = hidden[:, :, -GEN3R_EXACT_CACHE_T:].clone()
                if int(cache_inputs.shape[2]) < 2 and cached is not None:
                    cache_inputs = torch.cat(
                        [cached[:, :, -1:].to(cache_inputs.device), cache_inputs],
                        dim=2,
                    )
                hidden = layer(hidden, cached)
                feature_cache[cache_index] = cache_inputs
                feature_index[0] += 1
            else:
                hidden = layer(hidden)
        return self.out_block(hidden)


class FastWAMExactGeometryAdapter(nn.Module):
    """Gen3R-layer-exact VAE adapted only at its FastWAM I/O boundaries."""

    def __init__(
        self,
        *,
        input_channels: int = 10240,
        latent_channels: int = 48,
        hidden_dim: int = 128,
        channel_resample_scale: int = 10,
        input_grid_size: Sequence[int] = (24, 20),
        latent_grid_size: Sequence[int] = (24, 20),
        dim_mult: Sequence[int] = GEN3R_EXACT_DIM_MULT,
        num_res_blocks: int = 2,
        attention_scales: Sequence[float] = (),
        temporal_downsample: Sequence[bool] = GEN3R_EXACT_TEMPORAL_DOWNSAMPLE,
        dropout: float = 0.0,
        latent_mean: Sequence[float] | None = None,
        latent_std: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.channel_resample_scale = int(channel_resample_scale)
        self.input_grid_size = _parse_pair(input_grid_size, name="input_grid_size")
        self.latent_grid_size = _parse_pair(latent_grid_size, name="latent_grid_size")
        self.dim_mult = tuple(map(int, dim_mult))
        self.num_res_blocks = int(num_res_blocks)
        self.attention_scales = tuple(map(float, attention_scales))
        self.temporal_downsample = tuple(bool(value) for value in temporal_downsample)
        self.temporal_upsample = self.temporal_downsample[::-1]
        self.dropout = float(dropout)
        if self.dim_mult != GEN3R_EXACT_DIM_MULT:
            raise ValueError(
                f"Exact Gen3R requires dim_mult={GEN3R_EXACT_DIM_MULT}, got {self.dim_mult}."
            )
        if self.num_res_blocks != 2 or self.attention_scales:
            raise ValueError("Exact Gen3R requires two residual blocks and no stage attention.")
        if self.temporal_downsample != GEN3R_EXACT_TEMPORAL_DOWNSAMPLE:
            raise ValueError(
                "Exact Gen3R requires temporal_downsample="
                f"{GEN3R_EXACT_TEMPORAL_DOWNSAMPLE}."
            )
        if self.dropout != 0.0:
            raise ValueError("Exact Gen3R uses dropout=0.0.")
        if min(
            self.input_channels,
            self.latent_channels,
            self.hidden_dim,
            self.channel_resample_scale,
        ) <= 0:
            raise ValueError("Exact adapter dimensions must be positive.")
        if self.input_channels % self.channel_resample_scale != 0:
            raise ValueError("Input channels must be divisible by channel_resample_scale.")

        if latent_mean is None:
            if self.latent_channels != len(WAN22_VAE_LATENT_MEAN):
                raise ValueError("Non-48D exact adapters must provide `latent_mean`.")
            latent_mean = WAN22_VAE_LATENT_MEAN
        if latent_std is None:
            if self.latent_channels != len(WAN22_VAE_LATENT_STD):
                raise ValueError("Non-48D exact adapters must provide `latent_std`.")
            latent_std = WAN22_VAE_LATENT_STD
        if len(latent_mean) != self.latent_channels or len(latent_std) != self.latent_channels:
            raise ValueError("Latent mean/std lengths must equal latent_channels.")
        mean_values = tuple(map(float, latent_mean))
        std_values = tuple(map(float, latent_std))
        if not all(math.isfinite(value) for value in (*mean_values, *std_values)):
            raise ValueError("Latent mean/std must be finite.")
        if not all(value > 0 for value in std_values):
            raise ValueError("Latent std values must be positive.")
        mean_tensor = torch.tensor(mean_values, dtype=torch.float32)
        std_tensor = torch.tensor(std_values, dtype=torch.float32)
        self.register_buffer("latent_mean", mean_tensor, persistent=True)
        self.register_buffer("latent_std", std_tensor, persistent=True)

        self.encoder = ExactEncoder3d(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            posterior_channels=2 * self.latent_channels,
            channel_resample_scale=self.channel_resample_scale,
            input_grid_size=self.input_grid_size,
            latent_grid_size=self.latent_grid_size,
            dim_mult=self.dim_mult,
            num_res_blocks=self.num_res_blocks,
            attention_scales=self.attention_scales,
            temporal_downsample=self.temporal_downsample,
            dropout=self.dropout,
        )
        self.posterior_quant_conv = ExactCausalConv3d(
            2 * self.latent_channels,
            2 * self.latent_channels,
            1,
        )
        self.post_quant_conv = ExactCausalConv3d(
            self.latent_channels,
            self.latent_channels,
            1,
        )
        self.decoder = ExactDecoder3d(
            output_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            latent_channels=self.latent_channels,
            channel_resample_scale=self.channel_resample_scale,
            input_grid_size=self.input_grid_size,
            latent_grid_size=self.latent_grid_size,
            dim_mult=self.dim_mult,
            num_res_blocks=self.num_res_blocks,
            attention_scales=self.attention_scales,
            temporal_upsample=self.temporal_upsample,
            dropout=self.dropout,
        )
        self._clear_cache()

    @property
    def temporal_compression_ratio(self) -> int:
        return 4

    def architecture(self) -> dict[str, Any]:
        return {
            "format_version": GEN3R_EXACT_FORMAT_VERSION,
            "model_class": type(self).__name__,
            "source": "Gen3R_GeometryAdapter_layer_exact_FastWAM_shapes",
            "input_channels": self.input_channels,
            "latent_channels": self.latent_channels,
            "hidden_dim": self.hidden_dim,
            "channel_resample_scale": self.channel_resample_scale,
            "input_grid_size": list(self.input_grid_size),
            "latent_grid_size": list(self.latent_grid_size),
            "dim_mult": list(self.dim_mult),
            "num_res_blocks": self.num_res_blocks,
            "attention_scales": list(self.attention_scales),
            "temporal_downsample": list(self.temporal_downsample),
            "dropout": self.dropout,
            "temporal_compression_ratio": self.temporal_compression_ratio,
            "latent_scale": "WanVideoVAE38_mean_std_mean_only_matches_Gen3R",
            "field_contract": (
                "robotwin_latent24x20_view_aligned_four_patch_fields_plus_"
                "local_index_camera_16_register_carrier_per_timestep_omega_v4"
            ),
            "omega_grouping_contract": VGGT_GEN3R_EXACT_GROUPING_CONTRACT,
            "native_supervision_contract": GEN3R_EXACT_NATIVE_SUPERVISION_CONTRACT,
            "depth_contract": "linear_z_depth_divided_by_per_timestep_scene_scale",
            "point_contract": (
                "per_timestep_head_camera_divided_by_per_timestep_scene_scale"
            ),
        }

    def legacy_full_clip_architecture(self) -> dict[str, Any]:
        """Return the exact v3 metadata accepted only for explicit legacy evaluation."""
        architecture = dict(self.architecture())
        architecture["format_version"] = 3
        architecture["field_contract"] = (
            "robotwin_latent24x20_view_aligned_four_patch_fields_plus_"
            "local_index_camera_16_register_carrier_v3"
        )
        for key in (
            "omega_grouping_contract",
            "native_supervision_contract",
            "depth_contract",
            "point_contract",
        ):
            architecture.pop(key)
        return architecture

    def _clear_cache(self) -> None:
        self._decoder_cache = [None] * _count_exact_causal_convs(self.decoder)
        self._encoder_cache = [None] * _count_exact_causal_convs(self.encoder)

    def _validate_input(self, geometry_tokens: torch.Tensor) -> None:
        if not isinstance(geometry_tokens, torch.Tensor) or not geometry_tokens.is_floating_point():
            raise TypeError("Geometry tokens must be a floating-point tensor.")
        expected_tail = (self.input_channels, *self.input_grid_size)
        if geometry_tokens.ndim != 5 or (
            int(geometry_tokens.shape[1]),
            int(geometry_tokens.shape[3]),
            int(geometry_tokens.shape[4]),
        ) != expected_tail:
            raise ValueError(
                f"Geometry input must be [B,{self.input_channels},T,"
                f"{self.input_grid_size[0]},{self.input_grid_size[1]}], "
                f"got {tuple(geometry_tokens.shape)}."
            )
        if (int(geometry_tokens.shape[2]) - 1) % self.temporal_compression_ratio != 0:
            raise ValueError("Exact Gen3R geometry time must satisfy T=1+4k.")
        if not bool(torch.isfinite(geometry_tokens).all()):
            raise ValueError("Geometry tokens contain non-finite values.")

    def encode(self, geometry_tokens: torch.Tensor) -> DiagonalGaussianPosterior:
        self._validate_input(geometry_tokens)
        self._clear_cache()
        num_steps = int(geometry_tokens.shape[2])
        num_chunks = 1 + (num_steps - 1) // 4
        encoded_chunks = []
        for chunk_index in range(num_chunks):
            feature_index = [0]
            if chunk_index == 0:
                chunk = geometry_tokens[:, :, :1]
            else:
                start = 1 + 4 * (chunk_index - 1)
                chunk = geometry_tokens[:, :, start:start + 4]
            encoded_chunks.append(
                self.encoder(
                    chunk,
                    feature_cache=self._encoder_cache,
                    feature_index=feature_index,
                )
            )
        moments = self.posterior_quant_conv(torch.cat(encoded_chunks, dim=2))
        mean, logvar = moments.chunk(2, dim=1)
        latent_mean = self.latent_mean.to(device=mean.device, dtype=mean.dtype).view(
            1,
            self.latent_channels,
            1,
            1,
            1,
        )
        inverse_std = self.latent_std.to(device=mean.device, dtype=mean.dtype).reciprocal().view(
            1,
            self.latent_channels,
            1,
            1,
            1,
        )
        mean = (mean - latent_mean) * inverse_std
        expected_shape = (
            int(geometry_tokens.shape[0]),
            self.latent_channels,
            num_chunks,
            *self.latent_grid_size,
        )
        if tuple(mean.shape) != expected_shape:
            raise AssertionError(
                f"Exact geometry posterior must be {expected_shape}, got {tuple(mean.shape)}."
            )
        self._clear_cache()
        return DiagonalGaussianPosterior(mean.contiguous(), logvar.contiguous())

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if not isinstance(latents, torch.Tensor) or not latents.is_floating_point():
            raise TypeError("Geometry latents must be a floating-point tensor.")
        expected_tail = (self.latent_channels, *self.latent_grid_size)
        if latents.ndim != 5 or (
            int(latents.shape[1]),
            int(latents.shape[3]),
            int(latents.shape[4]),
        ) != expected_tail:
            raise ValueError(
                f"Geometry latents must be [B,{self.latent_channels},F,"
                f"{self.latent_grid_size[0]},{self.latent_grid_size[1]}], "
                f"got {tuple(latents.shape)}."
            )
        if not bool(torch.isfinite(latents).all()):
            raise ValueError("Geometry latents contain non-finite values.")
        latent_mean = self.latent_mean.to(device=latents.device, dtype=latents.dtype).view(
            1,
            self.latent_channels,
            1,
            1,
            1,
        )
        inverse_std = self.latent_std.to(device=latents.device, dtype=latents.dtype).reciprocal().view(
            1,
            self.latent_channels,
            1,
            1,
            1,
        )
        hidden = latents / inverse_std + latent_mean
        hidden = self.post_quant_conv(hidden)
        self._clear_cache()
        decoded_chunks = []
        for latent_index in range(int(hidden.shape[2])):
            feature_index = [0]
            decoded_chunks.append(
                self.decoder(
                    hidden[:, :, latent_index:latent_index + 1],
                    feature_cache=self._decoder_cache,
                    feature_index=feature_index,
                )
            )
        reconstructed = torch.cat(decoded_chunks, dim=2)
        expected_shape = (
            int(latents.shape[0]),
            self.input_channels,
            4 * int(latents.shape[2]) - 3,
            *self.input_grid_size,
        )
        if tuple(reconstructed.shape) != expected_shape:
            raise AssertionError(
                f"Exact reconstruction must be {expected_shape}, got {tuple(reconstructed.shape)}."
            )
        self._clear_cache()
        return reconstructed.contiguous()

    def forward(
        self,
        geometry_tokens: torch.Tensor,
        *,
        sample_posterior: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, DiagonalGaussianPosterior]:
        posterior = self.encode(geometry_tokens)
        latents = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(latents), posterior

    def freeze(self) -> "FastWAMExactGeometryAdapter":
        self.eval()
        self.requires_grad_(False)
        return self

    def save_pretrained(self, path: str | Path) -> None:
        checkpoint_path = Path(path).expanduser()
        if checkpoint_path.exists():
            raise FileExistsError(f"Refusing to overwrite exact geometry adapter: {checkpoint_path}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"architecture": self.architecture(), "state_dict": self.state_dict()},
            checkpoint_path,
        )

    def load_pretrained(self, path: str | Path) -> None:
        checkpoint_path = Path(path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Exact geometry adapter checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        architecture = payload.get("architecture") if isinstance(payload, dict) else None
        allow_legacy_eval = os.environ.get(GEN3R_EXACT_LEGACY_EVAL_ENV) == "1"
        legacy_eval = (
            allow_legacy_eval
            and architecture == self.legacy_full_clip_architecture()
        )
        if architecture != self.architecture() and not legacy_eval:
            raise RuntimeError(f"Exact geometry adapter architecture mismatch: {checkpoint_path}")
        if legacy_eval:
            warnings.warn(
                "Loading legacy full-clip exact Geometry Adapter for explicit evaluation only.",
                RuntimeWarning,
                stacklevel=2,
            )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Exact geometry adapter checkpoint has no state_dict: {checkpoint_path}")
        self.load_state_dict(state_dict, strict=True)


__all__ = [
    "ExactAttentionBlock",
    "ExactCausalConv3d",
    "ExactDecoder3d",
    "ExactEncoder3d",
    "ExactRMSNorm",
    "ExactResample",
    "ExactResidualBlock",
    "FastWAMExactGeometryAdapter",
    "GEN3R_EXACT_DIM_MULT",
    "GEN3R_EXACT_TEMPORAL_DOWNSAMPLE",
    "WAN22_VAE_LATENT_MEAN",
    "WAN22_VAE_LATENT_STD",
]
