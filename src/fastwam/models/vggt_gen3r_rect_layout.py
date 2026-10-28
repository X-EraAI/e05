"""Rectangular RoboTwin layout for native 16x20 VGGT-Omega token grids.

Mirrors :mod:`fastwam.models.vggt_gen3r_exact_layout` but starts from the
rectangular native per-view grid ``16x20``.  The composed FastWAM latent grid
stays ``24x20``: the head camera keeps its native ``16x20`` tokens via an
identity copy while each wrist camera is independently downsampled to
``8x10``.  Special-token carriers remain per-view and exactly restorable.
"""

from __future__ import annotations

import torch
from einops import rearrange
import torch.nn.functional as F


ROBOTWIN_RECT_VIEW_GRID_SIZE = (16, 20)
ROBOTWIN_RECT_NUM_VIEWS = 3
ROBOTWIN_RECT_NUM_SPECIAL_TOKENS = 17
ROBOTWIN_RECT_GRID_SIZE = (24, 20)
ROBOTWIN_RECT_VIEW_REGIONS = (
    (0, 16, 0, 20),
    (16, 24, 0, 10),
    (16, 24, 10, 20),
)
ROBOTWIN_RECT_VIEW_NATIVE_SIZES = (
    (16, 20),  # head camera: identity copy
    (8, 10),   # left wrist: independent downsample
    (8, 10),   # right wrist: independent downsample
)

_SUPPORTED_COMPOSE_DTYPES = (torch.float32, torch.bfloat16)


def _validate_view_regions() -> None:
    height, width = ROBOTWIN_RECT_GRID_SIZE
    covered = [[False] * width for _ in range(height)]
    for view_index, (row_start, row_end, col_start, col_end) in enumerate(
        ROBOTWIN_RECT_VIEW_REGIONS
    ):
        if not (0 <= row_start < row_end <= height and 0 <= col_start < col_end <= width):
            raise AssertionError("RoboTwin rect view region is outside the composed grid.")
        if any(covered[row][col] for row in range(row_start, row_end) for col in range(col_start, col_end)):
            raise AssertionError("RoboTwin rect view regions overlap.")
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                covered[row][col] = True
        expected_size = ROBOTWIN_RECT_VIEW_NATIVE_SIZES[view_index]
        if (row_end - row_start, col_end - col_start) != expected_size:
            raise AssertionError(
                "RoboTwin rect view region size must equal its native target, got "
                f"{(row_end - row_start, col_end - col_start)} != {expected_size}."
            )
    if not all(all(row) for row in covered):
        raise AssertionError("RoboTwin rect view regions do not cover the composed grid.")


_validate_view_regions()


def robotwin_rect_view_id_grid(*, device: torch.device | str | None = None) -> torch.Tensor:
    """Return the deterministic ``[24,20]`` camera-view assignment."""
    view_ids = torch.empty(ROBOTWIN_RECT_GRID_SIZE, dtype=torch.int64, device=device)
    for view_index, (row_start, row_end, col_start, col_end) in enumerate(
        ROBOTWIN_RECT_VIEW_REGIONS
    ):
        view_ids[row_start:row_end, col_start:col_end] = view_index
    return view_ids


def _resolve_compose_dtype(
    tokens: torch.Tensor,
    output_dtype: torch.dtype | None,
) -> torch.dtype:
    if output_dtype is None:
        return tokens.dtype
    if output_dtype not in _SUPPORTED_COMPOSE_DTYPES:
        raise ValueError(
            "Rect token composition only supports torch.float32 or torch.bfloat16, "
            f"got {output_dtype}."
        )
    return output_dtype


def _resize_token_grid(
    tokens: torch.Tensor,
    size: tuple[int, int],
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    batch_size, num_steps, height, width, feature_dim = map(int, tokens.shape)
    resized = F.interpolate(
        rearrange(tokens.float(), "b t h w d -> (b t) d h w").contiguous(),
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return rearrange(
        resized,
        "(b t) d h w -> b t h w d",
        b=batch_size,
        t=num_steps,
    ).to(dtype=output_dtype).contiguous()


def _copy_token_grid(
    tokens: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    # Identity head path: only normalize dtype, never interpolate.
    return tokens.to(dtype=output_dtype).contiguous()


def compose_robotwin_rect_token_views(
    per_view_tokens: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compose native rectangular views into the FastWAM latent layout.

    The head camera ``[B,T,16,20,D]`` is copied identity into rows ``0:16``;
    each wrist ``[B,T,16,20,D]`` is independently resized to ``8x10``.

    Args:
        per_view_tokens: ``[B,T,V=3,16,20,D]``.

    Returns:
        View-aligned tokens ``[B,T,H=24,W=20,D]``.
    """
    if not isinstance(per_view_tokens, torch.Tensor) or not per_view_tokens.is_floating_point():
        raise TypeError("`per_view_tokens` must be a floating-point tensor.")
    expected_tail = (ROBOTWIN_RECT_NUM_VIEWS, *ROBOTWIN_RECT_VIEW_GRID_SIZE)
    if per_view_tokens.ndim != 6 or tuple(map(int, per_view_tokens.shape[2:5])) != expected_tail:
        raise ValueError(
            "Per-view rect tokens must be [B,T,3,16,20,D], got "
            f"{tuple(per_view_tokens.shape)}."
        )
    if not bool(torch.isfinite(per_view_tokens).all()):
        raise ValueError("Per-view rect tokens contain non-finite values.")
    resolved_dtype = _resolve_compose_dtype(per_view_tokens, output_dtype)
    composed = torch.empty(
        int(per_view_tokens.shape[0]),
        int(per_view_tokens.shape[1]),
        *ROBOTWIN_RECT_GRID_SIZE,
        int(per_view_tokens.shape[-1]),
        dtype=resolved_dtype,
        device=per_view_tokens.device,
    )
    for view_index, (row_start, row_end, col_start, col_end) in enumerate(
        ROBOTWIN_RECT_VIEW_REGIONS
    ):
        target_size = (row_end - row_start, col_end - col_start)
        view_tokens = per_view_tokens[:, :, view_index]
        if target_size == ROBOTWIN_RECT_VIEW_GRID_SIZE:
            composed[:, :, row_start:row_end, col_start:col_end] = _copy_token_grid(
                view_tokens,
                output_dtype=resolved_dtype,
            )
        else:
            composed[:, :, row_start:row_end, col_start:col_end] = _resize_token_grid(
                view_tokens,
                target_size,
                output_dtype=resolved_dtype,
            )
    if composed.dtype != resolved_dtype or composed.device != per_view_tokens.device:
        raise AssertionError("Composed rect view tokens changed dtype or device unexpectedly.")
    return composed.contiguous()


def restore_robotwin_rect_token_views(composed_tokens: torch.Tensor) -> torch.Tensor:
    """Restore native ``16x20`` grids for frozen Omega supervision heads.

    The head camera region is copied back identity; each wrist ``8x10`` region
    is bilinearly upsampled to ``16x20``.

    Args:
        composed_tokens: ``[B,T,H=24,W=20,D]``.

    Returns:
        Restored view grids ``[B,T,V=3,16,20,D]``.
    """
    if not isinstance(composed_tokens, torch.Tensor) or not composed_tokens.is_floating_point():
        raise TypeError("`composed_tokens` must be a floating-point tensor.")
    if composed_tokens.ndim != 5 or tuple(map(int, composed_tokens.shape[2:4])) != ROBOTWIN_RECT_GRID_SIZE:
        raise ValueError(
            "Composed rect tokens must be [B,T,24,20,D], got "
            f"{tuple(composed_tokens.shape)}."
        )
    restored = [
        (
            _copy_token_grid(
                composed_tokens[:, :, row_start:row_end, col_start:col_end],
                output_dtype=composed_tokens.dtype,
            )
            if (row_end - row_start, col_end - col_start) == ROBOTWIN_RECT_VIEW_GRID_SIZE
            else _resize_token_grid(
                composed_tokens[:, :, row_start:row_end, col_start:col_end],
                ROBOTWIN_RECT_VIEW_GRID_SIZE,
                output_dtype=composed_tokens.dtype,
            )
        )
        for row_start, row_end, col_start, col_end in ROBOTWIN_RECT_VIEW_REGIONS
    ]
    return torch.stack(restored, dim=2).contiguous()


def compose_special_tokens_to_rect_grid(
    special_tokens: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Tile camera/register tokens over each final rect view region."""
    if (
        not isinstance(special_tokens, torch.Tensor)
        or not special_tokens.is_floating_point()
        or special_tokens.ndim != 5
        or int(special_tokens.shape[2]) != ROBOTWIN_RECT_NUM_VIEWS
        or int(special_tokens.shape[3]) != ROBOTWIN_RECT_NUM_SPECIAL_TOKENS
    ):
        raise ValueError("Rect special tokens must be floating [B,T,3,17,D].")
    if not bool(torch.isfinite(special_tokens).all()):
        raise ValueError("Rect special tokens contain non-finite values.")
    resolved_dtype = _resolve_compose_dtype(special_tokens, output_dtype)
    composed = torch.empty(
        int(special_tokens.shape[0]),
        int(special_tokens.shape[1]),
        *ROBOTWIN_RECT_GRID_SIZE,
        int(special_tokens.shape[-1]),
        dtype=resolved_dtype,
        device=special_tokens.device,
    )
    for view_index, (row_start, row_end, col_start, col_end) in enumerate(
        ROBOTWIN_RECT_VIEW_REGIONS
    ):
        region_height = row_end - row_start
        region_width = col_end - col_start
        token_indices = torch.arange(
            region_height * region_width,
            device=special_tokens.device,
        ) % ROBOTWIN_RECT_NUM_SPECIAL_TOKENS
        tiled = special_tokens[:, :, view_index, token_indices]
        composed[:, :, row_start:row_end, col_start:col_end] = rearrange(
            tiled,
            "b t (h w) d -> b t h w d",
            h=region_height,
            w=region_width,
        )
    if composed.dtype != resolved_dtype or composed.device != special_tokens.device:
        raise AssertionError("Composed rect special tokens changed dtype or device unexpectedly.")
    return composed.contiguous()


def restore_special_tokens_from_rect_grid(special_grid: torch.Tensor) -> torch.Tensor:
    """Recover each token by averaging its local-index carrier cells per view."""
    if (
        not isinstance(special_grid, torch.Tensor)
        or not special_grid.is_floating_point()
        or special_grid.ndim != 5
        or tuple(map(int, special_grid.shape[2:4])) != ROBOTWIN_RECT_GRID_SIZE
    ):
        raise ValueError("Rect special grid must be floating [B,T,24,20,D].")
    restored_views = []
    for row_start, row_end, col_start, col_end in ROBOTWIN_RECT_VIEW_REGIONS:
        flattened = rearrange(
            special_grid[:, :, row_start:row_end, col_start:col_end],
            "b t h w d -> b t (h w) d",
        )
        indices = torch.arange(flattened.shape[2], device=flattened.device)
        restored_views.append(
            torch.stack(
                [
                    flattened[:, :, indices % ROBOTWIN_RECT_NUM_SPECIAL_TOKENS == token_index].mean(dim=2)
                    for token_index in range(ROBOTWIN_RECT_NUM_SPECIAL_TOKENS)
                ],
                dim=2,
            )
        )
    return torch.stack(restored_views, dim=2).contiguous()


__all__ = [
    "ROBOTWIN_RECT_GRID_SIZE",
    "ROBOTWIN_RECT_NUM_SPECIAL_TOKENS",
    "ROBOTWIN_RECT_NUM_VIEWS",
    "ROBOTWIN_RECT_VIEW_GRID_SIZE",
    "ROBOTWIN_RECT_VIEW_NATIVE_SIZES",
    "ROBOTWIN_RECT_VIEW_REGIONS",
    "compose_robotwin_rect_token_views",
    "compose_special_tokens_to_rect_grid",
    "restore_robotwin_rect_token_views",
    "restore_special_tokens_from_rect_grid",
    "robotwin_rect_view_id_grid",
]
