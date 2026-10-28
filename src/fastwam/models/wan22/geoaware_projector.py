from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from .vggt_mot import load_aggregator_weights, validate_vggt_images


class FrozenVGGTFeatureBackbone(nn.Module):
    """VGGT Aggregator returning selected post-block hidden features.

    VGGT-Omega concatenates the per-frame and inter-frame hidden states at each
    cached layer, yielding tensors shaped ``[B, V, S, 2 * embed_dim]``. Unlike
    the K/V bridge, this wrapper uses the Aggregator's normal output path and
    never observes or modifies attention internals.
    """

    def __init__(
        self,
        aggregator: nn.Module,
        selected_layers: Sequence[int],
        *,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        selected = tuple(int(layer) for layer in selected_layers)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("`selected_layers` must be unique and non-empty.")
        depth = int(getattr(aggregator, "depth", 24))
        invalid = [layer for layer in selected if not 0 <= layer < depth]
        if invalid:
            raise ValueError(
                f"`selected_layers` contains indices outside [0,{depth}): {invalid}"
            )
        cached = set(getattr(aggregator, "cached_layer_indices", ()))
        missing = set(selected) - cached
        if missing:
            raise ValueError(
                "VGGT Aggregator must cache every selected hidden layer; "
                f"missing={sorted(missing)}."
            )
        self.aggregator = aggregator
        self.selected_layers = selected
        self.trainable = bool(trainable)
        self.aggregator.requires_grad_(self.trainable)
        self.train(self.trainable)

    def train(self, mode: bool = True):
        effective_mode = bool(mode) if self.trainable else False
        super().train(effective_mode)
        self.aggregator.train(effective_mode)
        return self

    def forward(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        validate_vggt_images(
            images,
            patch_size=int(getattr(self.aggregator, "patch_size", 16)),
        )
        grad_context = nullcontext() if self.trainable else torch.no_grad()
        with grad_context:
            output = self.aggregator(images)
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError(
                "VGGT Aggregator normal forward must return "
                "(aggregated_tokens_list, patch_token_start)."
            )
        features, _ = output
        if not isinstance(features, (tuple, list)):
            raise RuntimeError("VGGT aggregated feature output must be a sequence.")

        selected: dict[int, torch.Tensor] = {}
        for layer in self.selected_layers:
            feature = features[layer]
            if feature is None:
                raise RuntimeError(f"VGGT did not cache selected hidden layer {layer}.")
            if feature.ndim != 4:
                raise RuntimeError(
                    f"VGGT layer {layer} must be [B,V,S,D], got {tuple(feature.shape)}."
                )
            selected[layer] = feature if self.trainable else feature.detach()
        return selected


def build_vggt_feature_backbone(
    *,
    weights_path: str,
    aggregator_checkpoint_path: str | None,
    selected_layers: Sequence[int],
    trainable: bool = False,
    aggregator_factory: Callable[..., nn.Module] | None = None,
) -> FrozenVGGTFeatureBackbone:
    """Build and strictly load a frozen or trainable feature Aggregator."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not aggregator_checkpoint_path:
        raise ValueError(
            "Distributed VGGT construction requires `aggregator_checkpoint_path`."
        )
    if aggregator_factory is None:
        from vggt_omega.models.aggregator import Aggregator

        aggregator_factory = Aggregator
    selected = tuple(int(layer) for layer in selected_layers)
    aggregator = aggregator_factory(cached_layer_indices=selected)
    checkpoint_path = aggregator_checkpoint_path or weights_path
    load_aggregator_weights(
        aggregator,
        Path(checkpoint_path),
        aggregator_only=aggregator_checkpoint_path is not None,
    )
    return FrozenVGGTFeatureBackbone(
        aggregator,
        selected,
        trainable=trainable,
    )


class GeoAwareProjector(nn.Module):
    """GeoAware-VLA-style multi-layer, per-view feature projector."""

    def __init__(
        self,
        *,
        selected_layers: Sequence[int] = (4, 11, 17, 23),
        input_dim: int = 2048,
        layer_dim: int = 512,
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        selected = tuple(int(layer) for layer in selected_layers)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("`selected_layers` must be unique and non-empty.")
        if min(input_dim, layer_dim, output_dim) <= 0:
            raise ValueError("Projector dimensions must be positive.")

        self.selected_layers = selected
        self.input_dim = int(input_dim)
        self.layer_dim = int(layer_dim)
        self.output_dim = int(output_dim)
        self.layer_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.input_dim, 1024, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(1024, self.layer_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                for _ in selected
            ]
        )
        self.final_proj = nn.Sequential(
            nn.Linear(self.layer_dim * len(selected), 1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_dim),
        )

    def _validate_features(
        self, features_by_layer: Mapping[int, torch.Tensor]
    ) -> tuple[int, int]:
        missing = set(self.selected_layers) - set(features_by_layer)
        unexpected = set(features_by_layer) - set(self.selected_layers)
        if missing or unexpected:
            raise ValueError(
                "GeoAware feature layer mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
            )
        reference = features_by_layer[self.selected_layers[0]]
        if reference.ndim != 4 or reference.shape[-1] != self.input_dim:
            raise ValueError(
                "GeoAware features must be [B,V,S,input_dim], got "
                f"{tuple(reference.shape)} with input_dim={self.input_dim}."
            )
        batch_size, num_views = int(reference.shape[0]), int(reference.shape[1])
        for layer in self.selected_layers:
            feature = features_by_layer[layer]
            if (
                feature.ndim != 4
                or tuple(feature.shape[:2]) != (batch_size, num_views)
                or feature.shape[-1] != self.input_dim
            ):
                raise ValueError(
                    f"GeoAware layer {layer} shape mismatch: {tuple(feature.shape)}."
                )
            if feature.dtype != reference.dtype or feature.device != reference.device:
                raise ValueError("All GeoAware layer features must share dtype and device.")
        return batch_size, num_views

    def forward(self, features_by_layer: Mapping[int, torch.Tensor]) -> torch.Tensor:
        batch_size, num_views = self._validate_features(features_by_layer)
        per_layer = []
        for conv, layer in zip(self.layer_convs, self.selected_layers):
            feature = features_by_layer[layer]
            tokens = feature.reshape(
                batch_size * num_views, feature.shape[2], self.input_dim
            ).transpose(1, 2)
            per_layer.append(conv(tokens).squeeze(-1))
        projected = self.final_proj(torch.cat(per_layer, dim=-1))
        return projected.view(batch_size, num_views, self.output_dim)


class VisualContextAdapter(nn.Module):
    """Map compact GeoAware tokens into the ActionDiT context width."""

    def __init__(self, input_dim: int = 512, output_dim: int = 1024) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Visual adapter dimensions must be positive.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Visual tokens must be [B,V,{self.input_dim}], got {tuple(tokens.shape)}."
            )
        return self.net(tokens)


__all__ = [
    "FrozenVGGTFeatureBackbone",
    "GeoAwareProjector",
    "VisualContextAdapter",
    "build_vggt_feature_backbone",
]
