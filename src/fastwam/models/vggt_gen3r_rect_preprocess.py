"""Shared rectangular VGGT preprocess: one 480x640 -> 256x320 resize.

This module is the single implementation of the rectangular RGB contract:
raw RoboTwin ``uint8`` (or controlled FP32 in ``[0,1]``) cameras are resized
exactly once with bilinear interpolation, ``align_corners=False`` and
``antialias=True`` to ``VGGT_RECT_IMAGE_SIZE=(256,320)``.  Datasets, stage-1
training and RoboTwin deployment must all call
:func:`vggt_rect_preprocess_rgb` instead of duplicating the resize.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


VGGT_RECT_IMAGE_SIZE = (256, 320)
VGGT_RECT_NATIVE_GRID_SIZE = (16, 20)
VGGT_RECT_PATCH_SIZE = 16
VGGT_RECT_NUM_PATCHES_PER_VIEW = (
    VGGT_RECT_NATIVE_GRID_SIZE[0] * VGGT_RECT_NATIVE_GRID_SIZE[1]
)
RECT_SOURCE_CAMERA_SIZE = (480, 640)


def _resolve_source_size(height: int, width: int) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"Rectangular VGGT images must be non-degenerate, got {height}x{width}.")
    return (int(height), int(width))


def _validate_float_unit_interval(images: torch.Tensor) -> None:
    """Validate a floating input with one tensor reduction instead of three scans."""
    image_min, image_max = torch.aminmax(images)
    if not bool(torch.isfinite(image_min)) or not bool(torch.isfinite(image_max)):
        raise ValueError("Rectangular VGGT images contain non-finite values.")
    min_value = float(image_min.item())
    max_value = float(image_max.item())
    if min_value < -1e-4 or max_value > 1.0001:
        raise ValueError(
            f"Rectangular VGGT images must be in [0,1], got [{min_value}, {max_value}]."
        )


def vggt_rect_preprocess_rgb(images: torch.Tensor) -> torch.Tensor:
    """Resize raw RGB cameras once to ``[256,320]`` FP32 in ``[0,1]``.

    Args:
        images: ``[...,3,H,W]`` raw ``uint8`` cameras or controlled FP32
            values in ``[0,1]``.  Accepts the RoboTwin raw ``480x640``
            cameras and any leading batch/time/view dimensions.  Inputs that
            are already ``256x320`` are only normalized in dtype/range and
            are never interpolated again.

    Returns:
        Contiguous FP32 tensor with the leading dims preserved and the last
        two spatial dims exactly ``VGGT_RECT_IMAGE_SIZE``.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError("Rectangular VGGT images must be a torch.Tensor.")
    if images.ndim < 3 or int(images.shape[-3]) != 3:
        raise ValueError(
            f"Rectangular VGGT images must end with [3,H,W], got {tuple(images.shape)}."
        )
    height, width = _resolve_source_size(int(images.shape[-2]), int(images.shape[-1]))
    if images.dtype == torch.uint8:
        # uint8 is finite and bounded by construction. Exact division by 255
        # therefore needs no subsequent full-tensor finite/range/clamp scans.
        images = images.to(dtype=torch.float32).div_(255.0)
    elif images.dtype in (torch.float32, torch.float64):
        images = images.to(dtype=torch.float32)
        _validate_float_unit_interval(images)
        images = images.clamp_(0.0, 1.0)
    else:
        raise TypeError(
            "Rectangular VGGT images must be uint8 or float32/float64, got "
            f"{images.dtype}."
        )
    if (height, width) != VGGT_RECT_IMAGE_SIZE:
        images = F.interpolate(
            images.reshape(-1, 3, height, width),
            size=VGGT_RECT_IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(*images.shape[:-2], *VGGT_RECT_IMAGE_SIZE)
    return images.contiguous()


__all__ = [
    "RECT_SOURCE_CAMERA_SIZE",
    "VGGT_RECT_IMAGE_SIZE",
    "VGGT_RECT_NATIVE_GRID_SIZE",
    "VGGT_RECT_NUM_PATCHES_PER_VIEW",
    "VGGT_RECT_PATCH_SIZE",
    "vggt_rect_preprocess_rgb",
]
