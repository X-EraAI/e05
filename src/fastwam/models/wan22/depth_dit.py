from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

from .action_dit import ActionDiT
from .wan_video_dit import sinusoidal_embedding_1d


class DepthDiT(ActionDiT):
    """ActionDiT backbone specialized for patchified multi-view future depth."""

    ACTION_BACKBONE_SKIP_PREFIXES = ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES + (
        "camera_embedding.",
        "row_embedding.",
        "column_embedding.",
    )

    def __init__(
        self,
        *,
        depth_patch_size: int,
        future_offsets: Sequence[int],
        num_views: int,
        patch_grid_height: int,
        patch_grid_width: int,
        **action_dit_kwargs,
    ) -> None:
        patch_size = int(depth_patch_size)
        offsets = tuple(int(value) for value in future_offsets)
        num_views = int(num_views)
        grid_height = int(patch_grid_height)
        grid_width = int(patch_grid_width)
        if patch_size <= 0 or num_views <= 0 or grid_height <= 0 or grid_width <= 0:
            raise ValueError("Depth patch/layout dimensions must be positive.")
        if (
            not offsets
            or any(value <= 0 for value in offsets)
            or any(left >= right for left, right in zip(offsets, offsets[1:]))
        ):
            raise ValueError("Depth future offsets must be strictly increasing positive integers.")
        expected_dim = patch_size * patch_size
        action_dim = int(action_dit_kwargs.pop("action_dim", expected_dim))
        if action_dim != expected_dim:
            raise ValueError(
                f"DepthDiT action_dim must equal patch_size^2={expected_dim}, got {action_dim}."
            )
        sequence_length = len(offsets) * num_views * grid_height * grid_width
        configured_max = int(action_dit_kwargs.get("max_seq_len", 1024))
        if configured_max < sequence_length:
            raise ValueError(
                f"Depth token length {sequence_length} exceeds max_seq_len={configured_max}."
            )
        super().__init__(action_dim=action_dim, **action_dit_kwargs)
        self.depth_patch_size = patch_size
        self.future_offsets = offsets
        self.num_views = num_views
        self.patch_grid_height = grid_height
        self.patch_grid_width = grid_width
        self.depth_sequence_length = sequence_length

        self.camera_embedding = nn.Embedding(num_views, self.hidden_dim)
        self.row_embedding = nn.Embedding(grid_height, self.hidden_dim)
        self.column_embedding = nn.Embedding(grid_width, self.hidden_dim)
        scale = self.hidden_dim**-0.5
        nn.init.normal_(self.camera_embedding.weight, std=scale)
        nn.init.normal_(self.row_embedding.weight, std=scale)
        nn.init.normal_(self.column_embedding.weight, std=scale)

        offsets_tensor = torch.tensor(offsets, dtype=torch.float32)
        offset_embedding = sinusoidal_embedding_1d(self.hidden_dim, offsets_tensor)
        self.register_buffer("offset_embedding", offset_embedding, persistent=False)

    @property
    def depth_height(self) -> int:
        return self.patch_grid_height * self.depth_patch_size

    @property
    def depth_width(self) -> int:
        return self.patch_grid_width * self.depth_patch_size

    def _position_embedding(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        horizon = self.offset_embedding[:, None, None, None, :]
        camera = self.camera_embedding.weight[None, :, None, None, :]
        row = self.row_embedding.weight[None, None, :, None, :]
        column = self.column_embedding.weight[None, None, None, :, :]
        position = horizon + camera + row + column
        return position.reshape(self.depth_sequence_length, self.hidden_dim).to(
            device=device, dtype=dtype
        )

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        if action_tokens.ndim != 3 or action_tokens.shape[1] != self.depth_sequence_length:
            raise ValueError(
                "Depth tokens must be [B,K*V*Gh*Gw,P^2] with sequence length "
                f"{self.depth_sequence_length}, got {tuple(action_tokens.shape)}."
            )
        state = super().pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        state["tokens"] = state["tokens"] + self._position_embedding(
            device=state["tokens"].device,
            dtype=state["tokens"].dtype,
        ).unsqueeze(0)
        state["meta"].update(
            {
                "future_offsets": self.future_offsets,
                "num_views": self.num_views,
                "patch_grid_height": self.patch_grid_height,
                "patch_grid_width": self.patch_grid_width,
            }
        )
        return state

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert ``[B,K*V*N,P^2]`` tokens to ``[B,K,V,H,W]`` maps."""
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            self.depth_sequence_length,
            self.action_dim,
        ):
            raise ValueError(
                "Depth tokens have incompatible shape for this layout: "
                f"{tuple(tokens.shape)}."
            )
        batch_size = tokens.shape[0]
        patch = self.depth_patch_size
        maps = tokens.view(
            batch_size,
            len(self.future_offsets),
            self.num_views,
            self.patch_grid_height,
            self.patch_grid_width,
            patch,
            patch,
        )
        return maps.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(
            batch_size,
            len(self.future_offsets),
            self.num_views,
            self.depth_height,
            self.depth_width,
        )


__all__ = ["DepthDiT"]
