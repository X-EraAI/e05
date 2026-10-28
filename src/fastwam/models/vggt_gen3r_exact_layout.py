"""View-aligned RoboTwin layout for native VGGT-Omega token grids."""

from __future__ import annotations

import torch
from einops import rearrange
import torch.nn.functional as F


ROBOTWIN_EXACT_VIEW_GRID_SIZE = (16, 16)
ROBOTWIN_EXACT_NUM_VIEWS = 3
ROBOTWIN_EXACT_NUM_SPECIAL_TOKENS = 17
ROBOTWIN_EXACT_GRID_SIZE = (24, 20)
ROBOTWIN_EXACT_VIEW_REGIONS = (
    (0, 16, 0, 20),
    (16, 24, 0, 10),
    (16, 24, 10, 20),
)

_SUPPORTED_COMPOSE_DTYPES = (torch.float32, torch.bfloat16)


def _validate_view_regions() -> None:
    height, width = ROBOTWIN_EXACT_GRID_SIZE
    covered = [[False] * width for _ in range(height)]
    for row_start, row_end, col_start, col_end in ROBOTWIN_EXACT_VIEW_REGIONS:
        if not (0 <= row_start < row_end <= height and 0 <= col_start < col_end <= width):
            raise AssertionError("RoboTwin exact view region is outside the composed grid.")
        if any(covered[row][col] for row in range(row_start, row_end) for col in range(col_start, col_end)):
            raise AssertionError("RoboTwin exact view regions overlap.")
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                covered[row][col] = True
    if not all(all(row) for row in covered):
        raise AssertionError("RoboTwin exact view regions do not cover the composed grid.")


_validate_view_regions()


def robotwin_exact_view_id_grid(*, device: torch.device | str | None = None) -> torch.Tensor:
    """Return the deterministic ``[24,20]`` camera-view assignment."""
    view_ids = torch.empty(ROBOTWIN_EXACT_GRID_SIZE, dtype=torch.int64, device=device)
    for view_index, (row_start, row_end, col_start, col_end) in enumerate(
        ROBOTWIN_EXACT_VIEW_REGIONS
    ):
        view_ids[row_start:row_end, col_start:col_end] = view_index
    return view_ids


def _resolve_compose_dtype(
    tokens: torch.Tensor,
    output_dtype: torch.dtype | None,
) -> torch.dtype:
    # Preserve the legacy public-layout behavior for callers that omit the
    # override. Geometry-only extraction always supplies an explicit FP32/BF16
    # dtype and therefore uses the stricter optimized-path contract below.
    if output_dtype is None:
        return tokens.dtype
    if output_dtype not in _SUPPORTED_COMPOSE_DTYPES:
        raise ValueError(
            "Exact token composition only supports torch.float32 or torch.bfloat16, "
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


def compose_robotwin_token_views(
    per_view_tokens: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Resize each native view independently into the FastWAM latent layout.

    Args:
        per_view_tokens: ``[B,T,V=3,H=16,W=16,D]``.

    Returns:
        View-aligned tokens ``[B,T,H=24,W=20,D]``.
    """
    if not isinstance(per_view_tokens, torch.Tensor) or not per_view_tokens.is_floating_point():
        raise TypeError("`per_view_tokens` must be a floating-point tensor.")
    expected_tail = (ROBOTWIN_EXACT_NUM_VIEWS, *ROBOTWIN_EXACT_VIEW_GRID_SIZE)
    if per_view_tokens.ndim != 6 or tuple(map(int, per_view_tokens.shape[2:5])) != expected_tail:
        raise ValueError(
            "Per-view tokens must be [B,T,3,16,16,D], got "
            f"{tuple(per_view_tokens.shape)}."
        )
    if not bool(torch.isfinite(per_view_tokens).all()):
        raise ValueError("Per-view tokens contain non-finite values.")
    resolved_dtype = _resolve_compose_dtype(per_view_tokens, output_dtype)
    composed = torch.empty(
        int(per_view_tokens.shape[0]),
        int(per_view_tokens.shape[1]),
        *ROBOTWIN_EXACT_GRID_SIZE,
        int(per_view_tokens.shape[-1]),
        dtype=resolved_dtype,
        device=per_view_tokens.device,
    )
    for view_index, (row_start, row_end, col_start, col_end) in enumerate(
        ROBOTWIN_EXACT_VIEW_REGIONS
    ):
        composed[:, :, row_start:row_end, col_start:col_end] = _resize_token_grid(
            per_view_tokens[:, :, view_index],
            (row_end - row_start, col_end - col_start),
            output_dtype=resolved_dtype,
        )
    if composed.dtype != resolved_dtype or composed.device != per_view_tokens.device:
        raise AssertionError("Composed view tokens changed dtype or device unexpectedly.")
    return composed.contiguous()


def restore_robotwin_token_views(composed_tokens: torch.Tensor) -> torch.Tensor:
    """Approximately restore native grids for frozen Omega supervision heads.

    Args:
        composed_tokens: ``[B,T,H=24,W=20,D]``.

    Returns:
        Independently resized view grids ``[B,T,V=3,H=16,W=16,D]``.
    """
    if not isinstance(composed_tokens, torch.Tensor) or not composed_tokens.is_floating_point():
        raise TypeError("`composed_tokens` must be a floating-point tensor.")
    if composed_tokens.ndim != 5 or tuple(map(int, composed_tokens.shape[2:4])) != ROBOTWIN_EXACT_GRID_SIZE:
        raise ValueError(
            "Composed tokens must be [B,T,24,20,D], got "
            f"{tuple(composed_tokens.shape)}."
        )
    restored = [
        _resize_token_grid(
            composed_tokens[:, :, row_start:row_end, col_start:col_end],
            ROBOTWIN_EXACT_VIEW_GRID_SIZE,
            output_dtype=composed_tokens.dtype,
        )
        for row_start, row_end, col_start, col_end in ROBOTWIN_EXACT_VIEW_REGIONS
    ]
    return torch.stack(restored, dim=2).contiguous()


def compose_special_tokens_to_grid(
    special_tokens: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Tile camera/register tokens directly over each final view region."""
    if (
        not isinstance(special_tokens, torch.Tensor)
        or not special_tokens.is_floating_point()
        or special_tokens.ndim != 5
        or int(special_tokens.shape[2]) != ROBOTWIN_EXACT_NUM_VIEWS
        or int(special_tokens.shape[3]) != ROBOTWIN_EXACT_NUM_SPECIAL_TOKENS
    ):
        raise ValueError("Special tokens must be floating [B,T,3,17,D].")
    if not bool(torch.isfinite(special_tokens).all()):
        raise ValueError("Special tokens contain non-finite values.")
    resolved_dtype = _resolve_compose_dtype(special_tokens, output_dtype)
    composed = torch.empty(
        int(special_tokens.shape[0]),
        int(special_tokens.shape[1]),
        *ROBOTWIN_EXACT_GRID_SIZE,
        int(special_tokens.shape[-1]),
        dtype=resolved_dtype,
        device=special_tokens.device,
    )
    for view_index, (row_start, row_end, col_start, col_end) in enumerate(
        ROBOTWIN_EXACT_VIEW_REGIONS
    ):
        region_height = row_end - row_start
        region_width = col_end - col_start
        token_indices = torch.arange(
            region_height * region_width,
            device=special_tokens.device,
        ) % ROBOTWIN_EXACT_NUM_SPECIAL_TOKENS
        tiled = special_tokens[:, :, view_index, token_indices]
        composed[:, :, row_start:row_end, col_start:col_end] = rearrange(
            tiled,
            "b t (h w) d -> b t h w d",
            h=region_height,
            w=region_width,
        )
    if composed.dtype != resolved_dtype or composed.device != special_tokens.device:
        raise AssertionError("Composed special tokens changed dtype or device unexpectedly.")
    return composed.contiguous()


def restore_special_tokens_from_grid(special_grid: torch.Tensor) -> torch.Tensor:
    """Recover each token by averaging its local-index carrier cells per view."""
    if (
        not isinstance(special_grid, torch.Tensor)
        or not special_grid.is_floating_point()
        or special_grid.ndim != 5
        or tuple(map(int, special_grid.shape[2:4])) != ROBOTWIN_EXACT_GRID_SIZE
    ):
        raise ValueError("Special grid must be floating [B,T,24,20,D].")
    restored_views = []
    for row_start, row_end, col_start, col_end in ROBOTWIN_EXACT_VIEW_REGIONS:
        flattened = rearrange(
            special_grid[:, :, row_start:row_end, col_start:col_end],
            "b t h w d -> b t (h w) d",
        )
        indices = torch.arange(flattened.shape[2], device=flattened.device)
        restored_views.append(
            torch.stack(
                [
                    flattened[:, :, indices % ROBOTWIN_EXACT_NUM_SPECIAL_TOKENS == token_index].mean(dim=2)
                    for token_index in range(ROBOTWIN_EXACT_NUM_SPECIAL_TOKENS)
                ],
                dim=2,
            )
        )
    return torch.stack(restored_views, dim=2).contiguous()


__all__ = [
    "ROBOTWIN_EXACT_GRID_SIZE",
    "ROBOTWIN_EXACT_NUM_VIEWS",
    "ROBOTWIN_EXACT_NUM_SPECIAL_TOKENS",
    "ROBOTWIN_EXACT_VIEW_GRID_SIZE",
    "ROBOTWIN_EXACT_VIEW_REGIONS",
    "compose_special_tokens_to_grid",
    "compose_robotwin_token_views",
    "robotwin_exact_view_id_grid",
    "restore_robotwin_token_views",
    "restore_special_tokens_from_grid",
]
