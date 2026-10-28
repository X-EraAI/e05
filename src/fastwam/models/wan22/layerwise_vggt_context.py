from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from .vggt_bridge import LayerMapping


class LayerwiseVGGTContextAdapter(nn.Module):
    """Project full VGGT hidden tokens into layer-specific ActionDiT context.

    Each configured pair is ``(vggt_layer, action_layer)``. VGGT supplies
    post-block hidden features shaped ``[B, V, S, input_dim]``. The adapter
    preserves every view/token, applies an independent ``LayerNorm + Linear``
    for each pair, and flattens only the view/token axes:

    ``[B,V,S,input_dim] -> [B,V,S,output_dim] -> [B,V*S,output_dim]``.
    """

    def __init__(
        self,
        *,
        layer_pairs: Sequence[tuple[int, int]],
        input_dim: int,
        output_dim: int,
        vggt_num_layers: int = 24,
        action_num_layers: int = 30,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Context adapter dimensions must be positive.")
        pairs = tuple(tuple(pair) for pair in layer_pairs)
        self.layer_mapping = LayerMapping(
            pairs,
            vggt_num_layers=int(vggt_num_layers),
            fastwam_num_layers=int(action_num_layers),
        )
        self.layer_pairs = pairs
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.adapters = nn.ModuleDict(
            {
                str(action_layer): nn.Sequential(
                    nn.LayerNorm(self.input_dim),
                    nn.Linear(self.input_dim, self.output_dim),
                )
                for _, action_layer in pairs
            }
        )

    @property
    def selected_vggt_layers(self) -> tuple[int, ...]:
        return tuple(vggt_layer for vggt_layer, _ in self.layer_pairs)

    @property
    def selected_action_layers(self) -> tuple[int, ...]:
        return tuple(action_layer for _, action_layer in self.layer_pairs)

    def vggt_layer_for(self, action_layer: int) -> int | None:
        return self.layer_mapping.vggt_layer_for(action_layer)

    def forward(
        self, features_by_layer: Mapping[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        expected_layers = set(self.selected_vggt_layers)
        missing = expected_layers - set(features_by_layer)
        unexpected = set(features_by_layer) - expected_layers
        if missing or unexpected:
            raise ValueError(
                "Layerwise VGGT feature mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
            )

        reference = features_by_layer[self.selected_vggt_layers[0]]
        if reference.ndim != 4 or reference.shape[-1] != self.input_dim:
            raise ValueError(
                "VGGT features must be [B,V,S,input_dim], got "
                f"{tuple(reference.shape)} with input_dim={self.input_dim}."
            )
        batch_size, num_views = int(reference.shape[0]), int(reference.shape[1])
        projected_by_action_layer: dict[int, torch.Tensor] = {}

        for vggt_layer, action_layer in self.layer_pairs:
            feature = features_by_layer[vggt_layer]
            if (
                feature.ndim != 4
                or tuple(feature.shape[:2]) != (batch_size, num_views)
                or feature.shape[-1] != self.input_dim
            ):
                raise ValueError(
                    f"VGGT layer {vggt_layer} shape mismatch: "
                    f"{tuple(feature.shape)}."
                )
            if feature.dtype != reference.dtype or feature.device != reference.device:
                raise ValueError(
                    "All layerwise VGGT features must share dtype and device."
                )
            projected = self.adapters[str(action_layer)](feature)
            projected_by_action_layer[action_layer] = projected.reshape(
                batch_size,
                num_views * int(projected.shape[2]),
                self.output_dim,
            )
        return projected_by_action_layer


__all__ = ["LayerwiseVGGTContextAdapter"]
