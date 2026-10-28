"""Gen3R-style causal geometry VAE adapted to FastWAM's Wan2.2 latent grid.

The upstream Gen3R implementation is vendored under ``third_party/Gen3R`` for
traceability.  This module intentionally has no runtime dependency on that
package: upstream couples its adapter to Diffusers and hard-codes the Wan2.1
``[B,16,F,70,70]`` latent shape.  FastWAM instead maps unnormalized VGGT-Omega
fields ``[B,10240,T,12,10]`` to Wan2.2-aligned latents
``[B,48,(T-1)//4+1,24,20]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F


VGGT_GEN3R_LAYER_INDICES = (4, 11, 17, 23)
VGGT_GEN3R_FEATURE_DIM = 2048
VGGT_GEN3R_NUM_FEATURE_FIELDS = 5
VGGT_GEN3R_INPUT_CHANNELS = VGGT_GEN3R_FEATURE_DIM * VGGT_GEN3R_NUM_FEATURE_FIELDS
VGGT_GEN3R_INPUT_GRID_SIZE = (12, 10)
VGGT_GEN3R_LATENT_CHANNELS = 48
VGGT_GEN3R_LATENT_GRID_SIZE = (24, 20)
VGGT_GEN3R_TEMPORAL_COMPRESSION = 4
VGGT_GEN3R_ADAPTER_FORMAT_VERSION = 1


def _validate_float_tensor(value: torch.Tensor, *, name: str, ndim: int) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"`{name}` must be a floating-point torch.Tensor.")
    if value.ndim != ndim:
        raise ValueError(f"`{name}` must have {ndim} dimensions, got {tuple(value.shape)}.")
    if not torch.compiler.is_compiling() and not bool(torch.isfinite(value.float()).all().item()):
        raise ValueError(f"`{name}` contains non-finite values.")


@dataclass(frozen=True)
class DiagonalGaussianPosterior:
    """Small pure-PyTorch diagonal Gaussian used by both adapter and tests."""

    mean: torch.Tensor
    logvar: torch.Tensor

    def __post_init__(self) -> None:
        _validate_float_tensor(self.mean, name="mean", ndim=5)
        _validate_float_tensor(self.logvar, name="logvar", ndim=5)
        if self.mean.shape != self.logvar.shape:
            raise ValueError(
                "Posterior mean/logvar shapes must match, got "
                f"{tuple(self.mean.shape)} and {tuple(self.logvar.shape)}."
            )

    @property
    def variance(self) -> torch.Tensor:
        return torch.exp(self.logvar.clamp(min=-30.0, max=20.0))

    @property
    def std(self) -> torch.Tensor:
        return torch.exp(0.5 * self.logvar.clamp(min=-30.0, max=20.0))

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape,
            dtype=self.mean.dtype,
            device=self.mean.device,
            generator=generator,
        )
        return self.mean + self.std.to(dtype=self.mean.dtype) * noise

    def kl_elements(self, other: "DiagonalGaussianPosterior | None" = None) -> torch.Tensor:
        """Return elementwise FP32 KL with shape ``[B,C,F,H,W]``."""
        mean = self.mean.float()
        logvar = self.logvar.float().clamp(min=-30.0, max=20.0)
        variance = torch.exp(logvar)
        if other is None:
            return 0.5 * (mean.square() + variance - 1.0 - logvar)
        if self.mean.shape != other.mean.shape:
            raise ValueError(
                "Aligned posteriors must have identical shapes, got "
                f"{tuple(self.mean.shape)} and {tuple(other.mean.shape)}."
            )
        other_mean = other.mean.float()
        other_logvar = other.logvar.float().clamp(min=-30.0, max=20.0)
        other_variance = torch.exp(other_logvar)
        return 0.5 * (
            other_logvar
            - logvar
            + (variance + (mean - other_mean).square()) / other_variance
            - 1.0
        )

    def kl(self, other: "DiagonalGaussianPosterior | None" = None) -> torch.Tensor:
        """Match Diffusers' Gen3R reduction, preserving batch and width."""
        return self.kl_elements(other).sum(dim=(1, 2, 3))


class CausalConv3d(nn.Conv3d):
    """Conv3d with left-only temporal padding and symmetric spatial padding."""

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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(inputs, self._causal_padding))


class ChannelRMSNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"`channels` must be positive, got {channels}.")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = inputs.float().square().mean(dim=1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance + self.eps).to(dtype=inputs.dtype)
        return normalized * self.weight.to(dtype=inputs.dtype)


class ResidualBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, eps: float) -> None:
        super().__init__()
        self.norm1 = ChannelRMSNorm(in_channels, eps=eps)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = ChannelRMSNorm(out_channels, eps=eps)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1)
        self.shortcut = (
            CausalConv3d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + residual


class SpatialAttentionBlock(nn.Module):
    """Single-head attention over HxW independently for each latent time step."""

    def __init__(self, channels: int, eps: float) -> None:
        super().__init__()
        self.norm = ChannelRMSNorm(channels, eps=eps)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_steps, height, width = inputs.shape
        hidden = rearrange(
            self.norm(inputs),
            "b c f h w -> (b f) c h w",
        ).contiguous()
        query, key, value = rearrange(
            self.qkv(hidden),
            "bf (qkv c) h w -> qkv bf 1 (h w) c",
            qkv=3,
        ).contiguous()
        hidden = F.scaled_dot_product_attention(query, key, value)
        hidden = rearrange(
            hidden,
            "(b f) 1 (h w) c -> (b f) c h w",
            b=batch_size,
            f=num_steps,
            h=height,
            w=width,
        ).contiguous()
        hidden = self.proj(hidden)
        hidden = rearrange(
            hidden,
            "(b f) c h w -> b c f h w",
            b=batch_size,
            f=num_steps,
        ).contiguous()
        if tuple(hidden.shape) != (batch_size, channels, num_steps, height, width):
            raise AssertionError("Spatial attention changed the tensor shape.")
        return inputs + hidden


class TemporalDownsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class TemporalUpsample(nn.Module):
    """Learned causal x2 expansion with first-frame-preserving ``2F-1`` output."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            channels,
            2 * channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expanded = rearrange(
            self.conv(inputs),
            "b (r c) f h w -> b c (f r) h w",
            r=2,
        ).contiguous()
        return torch.cat([expanded[:, :, 1:2], expanded[:, :, 2:]], dim=2).contiguous()


def _apply_per_frame(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    num_steps = int(inputs.shape[2])
    hidden = rearrange(inputs, "b c f h w -> (b f) c h w").contiguous()
    hidden = module(hidden)
    return rearrange(hidden, "(b f) c h w -> b c f h w", f=num_steps).contiguous()


class FastWAMGeometryAdapter(nn.Module):
    """Causal VAE from five aligned VGGT-Omega fields to Wan2.2 geometry latents."""

    def __init__(
        self,
        *,
        input_channels: int = VGGT_GEN3R_INPUT_CHANNELS,
        latent_channels: int = VGGT_GEN3R_LATENT_CHANNELS,
        hidden_dim: int = 128,
        input_grid_size: Sequence[int] = VGGT_GEN3R_INPUT_GRID_SIZE,
        latent_grid_size: Sequence[int] = VGGT_GEN3R_LATENT_GRID_SIZE,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.input_grid_size = self._parse_grid(input_grid_size, name="input_grid_size")
        self.latent_grid_size = self._parse_grid(latent_grid_size, name="latent_grid_size")
        self.num_res_blocks = int(num_res_blocks)
        self.dropout = float(dropout)
        self.eps = float(eps)
        if min(
            self.input_channels,
            self.latent_channels,
            self.hidden_dim,
            self.num_res_blocks,
        ) <= 0:
            raise ValueError("Adapter channels, hidden_dim and num_res_blocks must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0,1), got {self.dropout}.")
        if self.input_grid_size[0] * 2 != self.latent_grid_size[0] or self.input_grid_size[1] * 2 != self.latent_grid_size[1]:
            raise ValueError(
                "FastWAM geometry adapter requires a 2x spatial mapping, got "
                f"{self.input_grid_size} -> {self.latent_grid_size}."
            )

        encoder_dims = (
            4 * self.hidden_dim,
            4 * self.hidden_dim,
            2 * self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
        )
        self.encoder_input = nn.Conv2d(
            self.input_channels,
            encoder_dims[0],
            kernel_size=3,
            padding=1,
        )
        encoder_layers: list[nn.Module] = []
        temporal_downsample = (False, True, True)
        for stage_index, (in_channels, out_channels) in enumerate(
            zip(encoder_dims[:-1], encoder_dims[1:])
        ):
            for block_index in range(self.num_res_blocks):
                encoder_layers.append(
                    ResidualBlock3d(
                        in_channels if block_index == 0 else out_channels,
                        out_channels,
                        dropout=self.dropout,
                        eps=self.eps,
                    )
                )
            if stage_index < len(temporal_downsample) and temporal_downsample[stage_index]:
                encoder_layers.append(TemporalDownsample(out_channels))
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.encoder_middle = nn.Sequential(
            ResidualBlock3d(encoder_dims[-1], encoder_dims[-1], self.dropout, self.eps),
            SpatialAttentionBlock(encoder_dims[-1], self.eps),
            ResidualBlock3d(encoder_dims[-1], encoder_dims[-1], self.dropout, self.eps),
        )
        self.encoder_norm = ChannelRMSNorm(encoder_dims[-1], eps=self.eps)
        self.encoder_moments = CausalConv3d(
            encoder_dims[-1],
            2 * self.latent_channels,
            kernel_size=3,
            padding=1,
        )

        decoder_dims = (
            self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
            2 * self.hidden_dim,
            4 * self.hidden_dim,
        )
        self.decoder_input = CausalConv3d(
            self.latent_channels,
            decoder_dims[0],
            kernel_size=3,
            padding=1,
        )
        self.decoder_middle = nn.Sequential(
            ResidualBlock3d(decoder_dims[0], decoder_dims[0], self.dropout, self.eps),
            SpatialAttentionBlock(decoder_dims[0], self.eps),
            ResidualBlock3d(decoder_dims[0], decoder_dims[0], self.dropout, self.eps),
        )
        decoder_layers: list[nn.Module] = []
        temporal_upsample = (True, True, False)
        for stage_index, (in_channels, out_channels) in enumerate(
            zip(decoder_dims[:-1], decoder_dims[1:])
        ):
            for block_index in range(self.num_res_blocks + 1):
                decoder_layers.append(
                    ResidualBlock3d(
                        in_channels if block_index == 0 else out_channels,
                        out_channels,
                        dropout=self.dropout,
                        eps=self.eps,
                    )
                )
            if stage_index < len(temporal_upsample) and temporal_upsample[stage_index]:
                decoder_layers.append(TemporalUpsample(out_channels))
        self.decoder_layers = nn.Sequential(*decoder_layers)
        self.decoder_norm = ChannelRMSNorm(decoder_dims[-1], eps=self.eps)
        self.decoder_output = nn.Conv2d(
            decoder_dims[-1],
            self.input_channels,
            kernel_size=3,
            padding=1,
        )

    @staticmethod
    def _parse_grid(value: Sequence[int], *, name: str) -> tuple[int, int]:
        if len(value) != 2:
            raise ValueError(f"`{name}` must contain [H,W], got {value}.")
        grid = tuple(map(int, value))
        if min(grid) <= 0:
            raise ValueError(f"`{name}` must be positive, got {grid}.")
        return grid

    @property
    def temporal_compression_ratio(self) -> int:
        return VGGT_GEN3R_TEMPORAL_COMPRESSION

    def architecture(self) -> dict[str, Any]:
        return {
            "format_version": VGGT_GEN3R_ADAPTER_FORMAT_VERSION,
            "model_class": type(self).__name__,
            "source": "Gen3R_geometry_adapter_adapted_for_FastWAM_Wan2.2",
            "input_contract": "four_vggt_omega_layers_4_11_17_23_plus_camera_unnormalized",
            "input_channels": self.input_channels,
            "latent_channels": self.latent_channels,
            "hidden_dim": self.hidden_dim,
            "input_grid_size": list(self.input_grid_size),
            "latent_grid_size": list(self.latent_grid_size),
            "temporal_compression_ratio": self.temporal_compression_ratio,
            "num_res_blocks": self.num_res_blocks,
            "dropout": self.dropout,
            "eps": self.eps,
        }

    def _validate_geometry_tokens(self, geometry_tokens: torch.Tensor) -> None:
        _validate_float_tensor(geometry_tokens, name="geometry_tokens", ndim=5)
        if int(geometry_tokens.shape[1]) != self.input_channels:
            raise ValueError(
                f"Geometry input must have C={self.input_channels}, got {geometry_tokens.shape[1]}."
            )
        if tuple(geometry_tokens.shape[3:5]) != self.input_grid_size:
            raise ValueError(
                f"Geometry input grid must be {self.input_grid_size}, "
                f"got {tuple(geometry_tokens.shape[3:5])}."
            )
        num_frames = int(geometry_tokens.shape[2])
        if (num_frames - 1) % self.temporal_compression_ratio != 0:
            raise ValueError(
                "Geometry time must satisfy T=1+4k for causal alignment, "
                f"got T={num_frames}."
            )

    def encode(self, geometry_tokens: torch.Tensor) -> DiagonalGaussianPosterior:
        """Encode ``[B,Cg,T,12,10]`` to a normalized 48D geometry posterior."""
        self._validate_geometry_tokens(geometry_tokens)
        module_dtype = self.encoder_input.weight.dtype
        hidden = geometry_tokens.to(dtype=module_dtype)
        hidden = F.interpolate(
            hidden,
            size=(int(hidden.shape[2]), *self.latent_grid_size),
            mode="nearest",
        )
        hidden = _apply_per_frame(self.encoder_input, hidden)
        hidden = self.encoder_layers(hidden)
        hidden = self.encoder_middle(hidden)
        moments = self.encoder_moments(F.silu(self.encoder_norm(hidden)))
        mean, logvar = moments.chunk(2, dim=1)
        expected_steps = (int(geometry_tokens.shape[2]) - 1) // self.temporal_compression_ratio + 1
        expected_shape = (
            int(geometry_tokens.shape[0]),
            self.latent_channels,
            expected_steps,
            *self.latent_grid_size,
        )
        if tuple(mean.shape) != expected_shape:
            raise AssertionError(
                f"Geometry posterior must be {expected_shape}, got {tuple(mean.shape)}."
            )
        return DiagonalGaussianPosterior(mean.contiguous(), logvar.contiguous())

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode ``[B,48,F,24,20]`` to ``[B,Cg,4F-3,12,10]``."""
        _validate_float_tensor(latents, name="latents", ndim=5)
        if int(latents.shape[1]) != self.latent_channels:
            raise ValueError(
                f"Geometry latent must have C={self.latent_channels}, got {latents.shape[1]}."
            )
        if tuple(latents.shape[3:5]) != self.latent_grid_size:
            raise ValueError(
                f"Geometry latent grid must be {self.latent_grid_size}, "
                f"got {tuple(latents.shape[3:5])}."
            )
        module_dtype = self.decoder_input.weight.dtype
        hidden = self.decoder_input(latents.to(dtype=module_dtype))
        hidden = self.decoder_middle(hidden)
        hidden = self.decoder_layers(hidden)
        hidden = F.silu(self.decoder_norm(hidden))
        hidden = F.interpolate(
            hidden,
            size=(int(hidden.shape[2]), *self.input_grid_size),
            mode="nearest",
        )
        reconstructed = _apply_per_frame(self.decoder_output, hidden)
        expected_shape = (
            int(latents.shape[0]),
            self.input_channels,
            self.temporal_compression_ratio * int(latents.shape[2]) - 3,
            *self.input_grid_size,
        )
        if tuple(reconstructed.shape) != expected_shape:
            raise AssertionError(
                f"Decoded geometry must be {expected_shape}, got {tuple(reconstructed.shape)}."
            )
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

    def freeze(self) -> "FastWAMGeometryAdapter":
        self.eval()
        self.requires_grad_(False)
        return self

    def save_pretrained(self, path: str | Path) -> None:
        checkpoint_path = Path(path).expanduser()
        if checkpoint_path.exists():
            raise FileExistsError(f"Refusing to overwrite geometry adapter: {checkpoint_path}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": self.architecture(),
                "state_dict": self.state_dict(),
            },
            checkpoint_path,
        )

    def load_pretrained(self, path: str | Path) -> None:
        checkpoint_path = Path(path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Geometry adapter checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError(f"Geometry adapter checkpoint must be a dict: {checkpoint_path}")
        architecture = payload.get("architecture")
        if architecture != self.architecture():
            raise RuntimeError(
                "Geometry adapter architecture mismatch: "
                f"expected={self.architecture()}, checkpoint={architecture}."
            )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Geometry adapter checkpoint has no `state_dict`: {checkpoint_path}")
        self.load_state_dict(state_dict, strict=True)


def geometry_adapter_loss(
    *,
    reconstructed_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    posterior: DiagonalGaussianPosterior,
    video_posterior: DiagonalGaussianPosterior | None,
    frame_is_pad: torch.Tensor | None = None,
    smooth_l1_beta: float = 0.15,
    camera_smooth_l1_beta: float = 0.1,
    reconstruction_weight: float = 1.0,
    cosine_weight: float = 1.0,
    kl_weight: float = 0.1,
    view_regions: Sequence[Sequence[int]] | None = None,
    detach_recon_metrics: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Gen3R core objective with FP32 masked reductions.

    Dense VGGT head losses are deliberately outside this core function because
    FastWAM stores a compressed three-view collage rather than the native
    per-view 16x16 grids required by VGGT-Omega's dense head.

    ``detach_recon_metrics`` is only valid when all downstream recon/cos
    coefficients are zero. KL is always returned with gradients.
    """
    _validate_float_tensor(reconstructed_tokens, name="reconstructed_tokens", ndim=5)
    _validate_float_tensor(target_tokens, name="target_tokens", ndim=5)
    if reconstructed_tokens.shape != target_tokens.shape:
        raise ValueError(
            "Reconstruction/target shapes must match, got "
            f"{tuple(reconstructed_tokens.shape)} and {tuple(target_tokens.shape)}."
        )
    if (
        smooth_l1_beta <= 0
        or camera_smooth_l1_beta <= 0
        or reconstruction_weight < 0
        or cosine_weight < 0
        or kl_weight < 0
    ):
        raise ValueError("Loss beta must be positive and loss weights non-negative.")
    batch_size, input_channels, num_frames, _, _ = target_tokens.shape
    if input_channels % VGGT_GEN3R_NUM_FEATURE_FIELDS != 0:
        raise ValueError(
            "Geometry channels must contain four image fields plus one camera field, "
            f"got C={input_channels}."
        )
    feature_dim = input_channels // VGGT_GEN3R_NUM_FEATURE_FIELDS
    image_channels = (VGGT_GEN3R_NUM_FEATURE_FIELDS - 1) * feature_dim
    if frame_is_pad is None:
        frame_is_pad = torch.zeros(
            batch_size,
            num_frames,
            dtype=torch.bool,
            device=target_tokens.device,
        )
    if (
        not isinstance(frame_is_pad, torch.Tensor)
        or frame_is_pad.dtype != torch.bool
        or tuple(frame_is_pad.shape) != (batch_size, num_frames)
    ):
        raise ValueError(f"`frame_is_pad` must be bool {(batch_size, num_frames)}.")
    valid = (~frame_is_pad).to(device=target_tokens.device, dtype=torch.float32)
    valid_count = valid.sum().clamp_min(1.0)

    height, width = map(int, target_tokens.shape[3:])
    parsed_regions: tuple[tuple[int, int, int, int], ...] | None = None
    if view_regions is not None:
        parsed = []
        coverage = torch.zeros(height, width, dtype=torch.int64)
        for region_index, region in enumerate(view_regions):
            if len(region) != 4:
                raise ValueError(
                    f"View region {region_index} must be [r0,r1,c0,c1], got {region}."
                )
            row_start, row_end, col_start, col_end = map(int, region)
            if not (
                0 <= row_start < row_end <= height
                and 0 <= col_start < col_end <= width
            ):
                raise ValueError(
                    f"View region {region_index} is outside {(height, width)}: "
                    f"{(row_start, row_end, col_start, col_end)}."
                )
            coverage[row_start:row_end, col_start:col_end] += 1
            parsed.append((row_start, row_end, col_start, col_end))
        if not parsed or not bool(torch.equal(coverage, torch.ones_like(coverage))):
            raise ValueError("View regions must partition the complete geometry grid exactly once.")
        parsed_regions = tuple(parsed)

    reconstructed_image = reconstructed_tokens[:, :image_channels].float()
    target_image = target_tokens[:, :image_channels].float()
    reconstructed_camera_grid = reconstructed_tokens[:, image_channels:].float()
    target_camera_grid = target_tokens[:, image_channels:].float()

    image_reconstruction_elements = F.smooth_l1_loss(
        reconstructed_image,
        target_image,
        beta=float(smooth_l1_beta),
        reduction="none",
    )
    image_cosine_per_position = 1.0 - F.cosine_similarity(
        reconstructed_image,
        target_image,
        dim=1,
        eps=1e-6,
    )
    if parsed_regions is None:
        reconstructed_camera = reconstructed_camera_grid.mean(dim=(3, 4))
        target_camera = target_camera_grid.mean(dim=(3, 4))
        image_reconstruction_per_frame = image_reconstruction_elements.mean(
            dim=(1, 3, 4)
        )
        image_cosine_per_frame = image_cosine_per_position.mean(dim=(2, 3))
        camera_reconstruction_per_frame = F.smooth_l1_loss(
            reconstructed_camera,
            target_camera,
            beta=float(camera_smooth_l1_beta),
            reduction="none",
        ).mean(dim=1)
        camera_cosine_per_frame = 1.0 - F.cosine_similarity(
            reconstructed_camera,
            target_camera,
            dim=1,
            eps=1e-6,
        )
    else:
        image_reconstruction_regions = []
        image_cosine_regions = []
        camera_reconstruction_regions = []
        camera_cosine_regions = []
        for row_start, row_end, col_start, col_end in parsed_regions:
            image_reconstruction_regions.append(
                image_reconstruction_elements[
                    :, :, :, row_start:row_end, col_start:col_end
                ].mean(dim=(1, 3, 4))
            )
            image_cosine_regions.append(
                image_cosine_per_position[
                    :, :, row_start:row_end, col_start:col_end
                ].mean(dim=(2, 3))
            )
            reconstructed_camera = reconstructed_camera_grid[
                :, :, :, row_start:row_end, col_start:col_end
            ]
            target_camera = target_camera_grid[
                :, :, :, row_start:row_end, col_start:col_end
            ]
            camera_reconstruction_regions.append(
                F.smooth_l1_loss(
                    reconstructed_camera,
                    target_camera,
                    beta=float(camera_smooth_l1_beta),
                    reduction="none",
                ).mean(dim=(1, 3, 4))
            )
            camera_cosine_regions.append(
                1.0
                - F.cosine_similarity(
                    reconstructed_camera,
                    target_camera,
                    dim=1,
                    eps=1e-6,
                ).mean(dim=(2, 3))
            )
        image_reconstruction_per_frame = torch.stack(
            image_reconstruction_regions,
            dim=0,
        ).mean(dim=0)
        image_cosine_per_frame = torch.stack(image_cosine_regions, dim=0).mean(dim=0)
        camera_reconstruction_per_frame = torch.stack(
            camera_reconstruction_regions,
            dim=0,
        ).mean(dim=0)
        camera_cosine_per_frame = torch.stack(camera_cosine_regions, dim=0).mean(dim=0)

    image_reconstruction = (image_reconstruction_per_frame * valid).sum() / valid_count
    camera_reconstruction = (camera_reconstruction_per_frame * valid).sum() / valid_count
    reconstruction = image_reconstruction + camera_reconstruction
    image_cosine = (image_cosine_per_frame * valid).sum() / valid_count
    camera_cosine = (camera_cosine_per_frame * valid).sum() / valid_count
    cosine = image_cosine + camera_cosine

    kl_elements = posterior.kl_elements(video_posterior)
    if (num_frames - 1) % VGGT_GEN3R_TEMPORAL_COMPRESSION != 0:
        raise ValueError(
            "Adapter-loss time must satisfy T=1+4k, "
            f"got T={num_frames}."
        )
    latent_is_pad = torch.cat(
        [
            frame_is_pad[:, :1],
            frame_is_pad[:, 1:].reshape(
                batch_size,
                -1,
                VGGT_GEN3R_TEMPORAL_COMPRESSION,
            ).all(dim=2),
        ],
        dim=1,
    )
    # Diffusers sums C,F,H and leaves B,W before the caller's final mean.
    # Preserve that arithmetic while masking invalid latent timesteps.
    kl_per_step_width = kl_elements.sum(dim=(1, 3))
    if kl_per_step_width.shape[:2] != latent_is_pad.shape:
        raise ValueError(
            "Posterior/frame padding time mismatch: "
            f"KL={tuple(kl_per_step_width.shape)}, mask={tuple(latent_is_pad.shape)}."
        )
    latent_valid = (~latent_is_pad).to(
        device=kl_per_step_width.device,
        dtype=kl_per_step_width.dtype,
    )
    kl = (kl_per_step_width * latent_valid[..., None]).sum(dim=1).mean()
    if detach_recon_metrics:
        reconstruction = reconstruction.detach()
        image_reconstruction = image_reconstruction.detach()
        camera_reconstruction = camera_reconstruction.detach()
        cosine = cosine.detach()
        image_cosine = image_cosine.detach()
        camera_cosine = camera_cosine.detach()
    total = (
        float(reconstruction_weight) * reconstruction
        + float(cosine_weight) * cosine
        + float(kl_weight) * kl
    )
    return total, {
        "reconstruction": reconstruction,
        "image_reconstruction": image_reconstruction,
        "camera_reconstruction": camera_reconstruction,
        "cosine": cosine,
        "image_cosine": image_cosine,
        "camera_cosine": camera_cosine,
        "kl": kl,
    }


__all__ = [
    "DiagonalGaussianPosterior",
    "FastWAMGeometryAdapter",
    "VGGT_GEN3R_ADAPTER_FORMAT_VERSION",
    "VGGT_GEN3R_FEATURE_DIM",
    "VGGT_GEN3R_INPUT_CHANNELS",
    "VGGT_GEN3R_INPUT_GRID_SIZE",
    "VGGT_GEN3R_LATENT_CHANNELS",
    "VGGT_GEN3R_LATENT_GRID_SIZE",
    "VGGT_GEN3R_LAYER_INDICES",
    "VGGT_GEN3R_NUM_FEATURE_FIELDS",
    "VGGT_GEN3R_TEMPORAL_COMPRESSION",
    "geometry_adapter_loss",
]
