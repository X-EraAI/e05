"""Rectangular-256x320 exact Gen3R GeometryAdapter.

Inherits the full exact-adapter computation topology unchanged because both
its token input and latent grids stay ``24x20``.  Only the architecture
metadata contract differs so that rect checkpoints strictly refuse to load
into the old square exact adapter (and vice versa).
"""

from __future__ import annotations

from typing import Any

from .vggt_gen3r_geometry_adapter_exact import FastWAMExactGeometryAdapter
from .vggt_gen3r_rect_features import VGGT_GEN3R_RECT_GROUPING_CONTRACT


GEN3R_RECT_NATIVE_SUPERVISION_CONTRACT = (
    "per_timestep_head_camera_per_timestep_mean_valid_point_scale_rect256x320_v1"
)


class FastWAMExactRectGeometryAdapter(FastWAMExactGeometryAdapter):
    """Gen3R-layer-exact adapter fed by the rectangular VGGT-Omega encoder."""

    def architecture(self) -> dict[str, Any]:
        architecture = super().architecture()
        architecture["source"] = (
            "Gen3R_GeometryAdapter_layer_exact_FastWAM_shapes_rect256x320"
        )
        architecture["field_contract"] = (
            "robotwin_latent24x20_view_aligned_head_identity16x20_wrist8x10_"
            "local_index_camera_16_register_carrier_per_timestep_omega_rect_v1"
        )
        architecture["vggt_image_contract"] = "rect_256x320_native_16x20_single_resize_v1"
        architecture["omega_grouping_contract"] = VGGT_GEN3R_RECT_GROUPING_CONTRACT
        architecture["native_supervision_contract"] = GEN3R_RECT_NATIVE_SUPERVISION_CONTRACT
        return architecture


__all__ = [
    "FastWAMExactRectGeometryAdapter",
    "GEN3R_RECT_NATIVE_SUPERVISION_CONTRACT",
]
