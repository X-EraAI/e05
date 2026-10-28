"""Frozen VGGT-Omega feature extraction for the FastWAM VGGT16 cache.

This module deliberately produces unnormalized 2048D x0 targets. A nine-frame
RoboTwin clip is encoded as three causal groups ``[0]``, ``[1:5]``, and
``[5:9]``. For every group, only its three-view endpoint is retained and the
official 16x16 patch grid is average-pooled in FP32 to 4x4 (16 tokens/view).
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
import sys
from typing import Any

from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F


VGGT_CAUSAL_GROUP_SLICES = (slice(0, 1), slice(1, 5), slice(5, 9))
VGGT_IMAGE_SIZE = 256
VGGT_RAW_GRID_SIZE = 16
VGGT_TARGET_GRID_SIZE = 4
VGGT_FEATURE_DIM = 2048
VGGT_NUM_VIEWS = 3
_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}

_IGNORED_CHECKPOINT_PREFIXES = (
    "camera_head.",
    "dense_head.",
    "text_alignment_head.",
)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    checkpoint_path = Path(path)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_aggregator_checkpoint(
    aggregator: nn.Module,
    checkpoint: Any,
    checkpoint_path: Path,
) -> None:
    if not isinstance(checkpoint, Mapping) or not all(isinstance(key, str) for key in checkpoint):
        raise TypeError(f"VGGT-Omega checkpoint must be a direct string-keyed state dict: {checkpoint_path}")
    known_prefixes = ("aggregator.",) + _IGNORED_CHECKPOINT_PREFIXES
    unexpected = sorted(key for key in checkpoint if not key.startswith(known_prefixes))
    if unexpected:
        raise RuntimeError(f"Unsupported VGGT-Omega checkpoint keys: {unexpected[:20]}")
    aggregator_state = {
        key.removeprefix("aggregator."): value
        for key, value in checkpoint.items()
        if key.startswith("aggregator.")
    }
    if not aggregator_state:
        raise RuntimeError(f"VGGT-Omega checkpoint has no `aggregator.*` weights: {checkpoint_path}")
    try:
        aggregator.load_state_dict(aggregator_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Invalid VGGT-Omega aggregator checkpoint {checkpoint_path}: {exc}") from exc


class FrozenVGGTOmegaFeatureEncoder:
    """Extract unnormalized causal endpoint features with shape [B,3,3,16,2048]."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        repo_path: str | Path,
        device: str | torch.device = "cuda",
        last_layers: int = 4,
        compile_backbone: bool = True,
        compile_mode: str = "default",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.device = torch.device(device)
        self.last_layers = int(last_layers)
        self.compile_backbone = bool(compile_backbone)
        self.compile_mode = str(compile_mode).strip().lower()
        if self.compile_mode not in _COMPILE_MODES:
            raise ValueError(
                f"compile_mode must be one of {sorted(_COMPILE_MODES)}, "
                f"got {compile_mode!r}."
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {self.checkpoint_path}")
        if not (self.repo_path / "vggt_omega").is_dir():
            raise FileNotFoundError(f"VGGT-Omega source package not found under: {self.repo_path}")
        if self.last_layers != 4:
            raise ValueError(
                "The immutable VGGT16 cache contract requires `last_layers=4`, "
                f"got {self.last_layers}."
            )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA VGGT extraction was requested but CUDA is not available.")
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self._model: nn.Module | None = None

    def _load_model(self) -> nn.Module:
        if self._model is not None:
            return self._model
        repo_path_str = str(self.repo_path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)
        try:
            from vggt_omega.models import VGGTOmega
        except ImportError as exc:
            raise ImportError(f"Cannot import VGGT-Omega from {self.repo_path}: {exc}") from exc

        # Only the aggregator is needed for the unnormalized feature contract.
        model = VGGTOmega(enable_camera=False, enable_depth=False, enable_alignment=False)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        _load_aggregator_checkpoint(model.aggregator, checkpoint, self.checkpoint_path)
        model.eval()
        model.requires_grad_(False)
        model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        model.to(device=self.device, dtype=model_dtype)
        cache_frame_rope_grids = getattr(model.aggregator, "cache_frame_rope_grids", None)
        if callable(cache_frame_rope_grids):
            cache_frame_rope_grids()
        if self.compile_backbone:
            model.aggregator = torch.compile(
                model.aggregator,
                mode=self.compile_mode,
            )
        self._model = model
        return model

    @staticmethod
    def _validate_images(images: torch.Tensor, expected_steps: int) -> None:
        if not torch.is_tensor(images) or not images.is_floating_point():
            raise TypeError("VGGT images must be a floating-point torch.Tensor in [0,1].")
        if images.ndim != 6:
            raise ValueError(f"VGGT images must be [B,T,V,3,H,W], got {tuple(images.shape)}.")
        if images.shape[1] != expected_steps or images.shape[2] != VGGT_NUM_VIEWS or images.shape[3] != 3:
            raise ValueError(
                f"VGGT images must be [B,{expected_steps},{VGGT_NUM_VIEWS},3,H,W], "
                f"got {tuple(images.shape)}."
            )
        if not bool(torch.isfinite(images).all().item()):
            raise ValueError("VGGT images contain non-finite values.")
        image_min = float(images.amin().item())
        image_max = float(images.amax().item())
        if image_min < -1e-4 or image_max > 1.0001:
            raise ValueError(f"VGGT images must be in [0,1], got range [{image_min}, {image_max}].")

    def _encode_flat_group(self, images: torch.Tensor) -> torch.Tensor:
        """Encode [B,S,3,H,W], returning all patch tokens [B,S,256,2048]."""
        model = self._load_model()
        batch_size, num_images, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"Expected RGB images, got C={channels}.")
        images = images.to(device=self.device, dtype=torch.float32, non_blocking=True)
        if (height, width) != (VGGT_IMAGE_SIZE, VGGT_IMAGE_SIZE):
            images = F.interpolate(
                images.reshape(batch_size * num_images, channels, height, width),
                size=(VGGT_IMAGE_SIZE, VGGT_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).reshape(batch_size, num_images, channels, VGGT_IMAGE_SIZE, VGGT_IMAGE_SIZE)
        model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            layer_outputs, patch_start = model.aggregator(
                images.to(dtype=model_dtype),
                last_layer=self.last_layers,
                return_last_layers=True,
            )
        if len(layer_outputs) != self.last_layers or any(output is None for output in layer_outputs):
            raise RuntimeError(
                f"Expected {self.last_layers} VGGT-Omega layer outputs, got {len(layer_outputs)}."
            )
        patch_layers = [output[:, :, int(patch_start):].float() for output in layer_outputs]
        patch_tokens = torch.stack(patch_layers, dim=0).mean(dim=0)
        expected_shape = (batch_size, num_images, VGGT_RAW_GRID_SIZE**2, VGGT_FEATURE_DIM)
        if tuple(patch_tokens.shape) != expected_shape:
            raise ValueError(
                f"VGGT-Omega patch output mismatch: got {tuple(patch_tokens.shape)}, expected {expected_shape}."
            )
        return patch_tokens

    @staticmethod
    def _pool_endpoint(patch_tokens: torch.Tensor) -> torch.Tensor:
        """Pool endpoint [B,V,256,2048] to [B,V,16,2048] in FP32."""
        if patch_tokens.ndim != 4 or tuple(patch_tokens.shape[1:]) != (
            VGGT_NUM_VIEWS,
            VGGT_RAW_GRID_SIZE**2,
            VGGT_FEATURE_DIM,
        ):
            raise ValueError(f"Unexpected VGGT endpoint shape: {tuple(patch_tokens.shape)}.")
        pooled = F.adaptive_avg_pool2d(
            rearrange(
                patch_tokens.float(),
                "b v (h w) d -> (b v) d h w",
                h=VGGT_RAW_GRID_SIZE,
                w=VGGT_RAW_GRID_SIZE,
            ),
            output_size=(VGGT_TARGET_GRID_SIZE, VGGT_TARGET_GRID_SIZE),
        )
        return rearrange(
            pooled,
            "(b v) d h w -> b v (h w) d",
            v=VGGT_NUM_VIEWS,
        ).contiguous()

    def encode_clip(self, images: torch.Tensor) -> torch.Tensor:
        """Encode [B,9,3,3,H,W] into unnormalized [B,3,3,16,2048] x0."""
        self._validate_images(images, expected_steps=9)
        current_group = rearrange(
            images[:, VGGT_CAUSAL_GROUP_SLICES[0]],
            "b t v c h w -> b (t v) c h w",
        ).contiguous()
        current_target = self._pool_endpoint(
            self._encode_flat_group(current_group)[:, -VGGT_NUM_VIEWS:]
        )

        # G1/G2 both contain four time steps, so merge them along batch for one
        # aggregator call. Together with G0 this is two VGGT forwards per batch.
        future_groups = torch.cat(
            [images[:, group_slice] for group_slice in VGGT_CAUSAL_GROUP_SLICES[1:]],
            dim=0,
        )
        future_groups = rearrange(future_groups, "b t v c h w -> b (t v) c h w").contiguous()
        future_targets = self._pool_endpoint(
            self._encode_flat_group(future_groups)[:, -VGGT_NUM_VIEWS:]
        )
        batch_size = images.shape[0]
        if future_targets.shape[0] != 2 * batch_size:
            raise AssertionError("Internal future VGGT batch merge produced an invalid shape.")
        first_future, last_future = future_targets.split(batch_size, dim=0)
        return torch.stack([current_target, first_future, last_future], dim=1).contiguous()

    def encode_current(self, images: torch.Tensor) -> torch.Tensor:
        """Encode current views [B,3,3,H,W] into [B,3,16,2048]."""
        if images.ndim != 5:
            raise ValueError(f"Current VGGT images must be [B,V,3,H,W], got {tuple(images.shape)}.")
        clip = images.unsqueeze(1)
        self._validate_images(clip, expected_steps=1)
        flat_group = rearrange(clip, "b t v c h w -> b (t v) c h w").contiguous()
        return self._pool_endpoint(self._encode_flat_group(flat_group))
