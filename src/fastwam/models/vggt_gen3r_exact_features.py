"""View-aligned multi-view VGGT-Omega fields for exact Gen3R training."""

from __future__ import annotations

from typing import TYPE_CHECKING

from einops import rearrange
import torch
import torch.nn.functional as F

from .vggt_gen3r_exact_layout import (
    ROBOTWIN_EXACT_GRID_SIZE,
    compose_special_tokens_to_grid,
    compose_robotwin_token_views,
)
from .vggt_gen3r_geometry_adapter import (
    VGGT_GEN3R_FEATURE_DIM,
    VGGT_GEN3R_INPUT_CHANNELS,
    VGGT_GEN3R_LAYER_INDICES,
)
from .vggt_omega_features import (
    FrozenVGGTOmegaFeatureEncoder,
    VGGT_IMAGE_SIZE,
    VGGT_NUM_VIEWS,
    VGGT_RAW_GRID_SIZE,
)

if TYPE_CHECKING:
    from .vggt_gen3r_exact_supervision import FrozenVGGTOmegaGeometryHeads


VGGT_GEN3R_EXACT_INPUT_CHANNELS = VGGT_GEN3R_INPUT_CHANNELS
VGGT_GEN3R_EXACT_INPUT_GRID_SIZE = ROBOTWIN_EXACT_GRID_SIZE
VGGT_GEN3R_EXACT_GROUPING_CONTRACT = "omega_bt_batch_v_sequence_v1"
_SUPPORTED_GEOMETRY_DTYPES = (torch.float32, torch.bfloat16)


class FrozenVGGTOmegaGen3RExactFeatureEncoder(FrozenVGGTOmegaFeatureEncoder):
    """Encode three normal views, then align their native token grids.

    VGGT-Omega sees one three-view sequence per time step, with ``B*T`` grouped
    into the batch dimension for one aggregator call. Only after the frozen
    aggregator are the three complete 16x16 view grids independently resized
    into the exact GeoAdapter's FastWAM-aligned 24x20 grid.
    """

    @staticmethod
    def _validate_geometry_dtype(output_dtype: torch.dtype) -> None:
        if output_dtype not in _SUPPORTED_GEOMETRY_DTYPES:
            raise ValueError(
                "Exact geometry extraction only supports torch.float32 or "
                f"torch.bfloat16, got {output_dtype}."
            )

    def _is_encoder_device(self, tensor: torch.Tensor) -> bool:
        if tensor.device.type != self.device.type:
            return False
        if self.device.index is None:
            if self.device.type != "cuda":
                return True
            return tensor.device.index == torch.cuda.current_device()
        return tensor.device.index == self.device.index

    def _run_group_aggregator(
        self,
        images: torch.Tensor,
    ) -> tuple[list[torch.Tensor | None], int, int, int, int]:
        """Run one batched Omega pass as ``[B*T,V,3,H,W]``."""
        if not isinstance(images, torch.Tensor) or not images.is_floating_point():
            raise TypeError("VGGT group images must be a floating-point tensor.")
        if (
            images.ndim != 6
            or int(images.shape[2]) != VGGT_NUM_VIEWS
            or int(images.shape[3]) != 3
        ):
            raise ValueError(
                f"VGGT group must be [B,S,{VGGT_NUM_VIEWS},3,H,W], "
                f"got {tuple(images.shape)}."
            )
        batch_size, group_steps, num_views, channels, height, width = map(
            int,
            images.shape,
        )
        flat_images = rearrange(images, "b t v c h w -> (b t v) c h w").contiguous()
        if (height, width) != (VGGT_IMAGE_SIZE, VGGT_IMAGE_SIZE):
            flat_images = F.interpolate(
                flat_images.float(),
                size=(VGGT_IMAGE_SIZE, VGGT_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        flat_images = rearrange(
            flat_images,
            "(b t v) c h w -> (b t) v c h w",
            b=batch_size,
            t=group_steps,
            v=num_views,
        ).contiguous()

        model = self._load_model()
        flat_images = flat_images.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            layer_outputs, patch_start = model.aggregator(flat_images.to(dtype=model_dtype))
        if len(layer_outputs) != int(model.aggregator.depth):
            raise RuntimeError("VGGT-Omega returned an unexpected number of block outputs.")
        if int(patch_start) != int(model.aggregator.patch_token_start):
            raise RuntimeError("VGGT-Omega patch-token start changed during exact extraction.")
        for layer_index in VGGT_GEN3R_LAYER_INDICES:
            layer = layer_outputs[layer_index]
            if layer is None:
                raise RuntimeError(f"VGGT-Omega block {layer_index} is missing.")
            if not layer.is_floating_point() or not self._is_encoder_device(layer):
                raise RuntimeError(
                    f"VGGT-Omega block {layer_index} must be floating on {self.device}, "
                    f"got dtype={layer.dtype}, device={layer.device}."
                )
        return layer_outputs, int(patch_start), batch_size, group_steps, num_views

    def _encode_group(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode ``[B,S,V,3,H,W]`` including native supervision fields.

        Returns:
            - view-aligned FP32 fields ``[B,S,24,20,10240]``;
            - native FP32 image fields ``[B,S,V,4,16,16,2048]``;
            - native FP32 camera/register tokens ``[B,S,V,17,2048]``.
        """
        layer_outputs, patch_start, batch_size, group_steps, num_views = (
            self._run_group_aggregator(images)
        )

        per_view_image_fields = []
        composed_fields = []
        for layer_index in VGGT_GEN3R_LAYER_INDICES:
            layer = layer_outputs[layer_index]
            if layer is None:  # Defensive narrowing after aggregator validation.
                raise AssertionError(f"VGGT-Omega block {layer_index} disappeared.")
            patches = layer[:, :, int(patch_start):].float()
            expected = (
                batch_size * group_steps,
                num_views,
                VGGT_RAW_GRID_SIZE**2,
                VGGT_GEN3R_FEATURE_DIM,
            )
            if tuple(patches.shape) != expected:
                raise ValueError(
                    f"Omega block {layer_index} patches must be {expected}, "
                    f"got {tuple(patches.shape)}."
                )
            patch_grid = rearrange(
                patches,
                "(b t) v (h w) d -> b t v h w d",
                b=batch_size,
                t=group_steps,
                v=num_views,
                h=VGGT_RAW_GRID_SIZE,
                w=VGGT_RAW_GRID_SIZE,
            ).contiguous()
            per_view_image_fields.append(patch_grid)
            composed_fields.append(compose_robotwin_token_views(patch_grid))

        final_layer = layer_outputs[VGGT_GEN3R_LAYER_INDICES[-1]]
        if final_layer is None:
            raise RuntimeError("VGGT-Omega final block is missing.")
        special_tokens = rearrange(
            final_layer[:, :, : int(patch_start)].float(),
            "(b t) v n d -> b t v n d",
            b=batch_size,
            t=group_steps,
            v=num_views,
        ).contiguous()
        composed_fields.append(compose_special_tokens_to_grid(special_tokens))

        geometry = torch.cat(composed_fields, dim=-1)
        expected_geometry = (
            batch_size,
            group_steps,
            *VGGT_GEN3R_EXACT_INPUT_GRID_SIZE,
            VGGT_GEN3R_EXACT_INPUT_CHANNELS,
        )
        if tuple(geometry.shape) != expected_geometry:
            raise AssertionError(
                f"Exact composed fields must be {expected_geometry}, got {tuple(geometry.shape)}."
            )
        image_fields = torch.stack(per_view_image_fields, dim=3)
        expected_image_fields = (
            batch_size,
            group_steps,
            num_views,
            len(VGGT_GEN3R_LAYER_INDICES),
            VGGT_RAW_GRID_SIZE,
            VGGT_RAW_GRID_SIZE,
            VGGT_GEN3R_FEATURE_DIM,
        )
        if tuple(image_fields.shape) != expected_image_fields:
            raise AssertionError(
                f"Native view fields must be {expected_image_fields}, got {tuple(image_fields.shape)}."
            )
        if any(
            tensor.dtype != torch.float32 or not self._is_encoder_device(tensor)
            for tensor in (geometry, image_fields, special_tokens)
        ):
            raise AssertionError("Exact native extraction must return FP32 tensors on encoder.device.")
        return geometry.contiguous(), image_fields.contiguous(), special_tokens.contiguous()

    def _encode_group_geometry_only(
        self,
        images: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode only aligned geometry, without materializing native FP32 fields.

        Interpolation reductions stay FP32. Each completed field is cast directly
        to ``output_dtype`` before being copied into the final contiguous layout.
        This makes BF16 output numerically equivalent to casting the legacy FP32
        geometry result while avoiding the native-field list and stack.
        """
        self._validate_geometry_dtype(output_dtype)
        layer_outputs, patch_start, batch_size, group_steps, num_views = (
            self._run_group_aggregator(images)
        )
        expected_geometry = (
            batch_size,
            group_steps,
            *VGGT_GEN3R_EXACT_INPUT_GRID_SIZE,
            VGGT_GEN3R_EXACT_INPUT_CHANNELS,
        )
        geometry = torch.empty(
            expected_geometry,
            device=self.device,
            dtype=output_dtype,
        )

        final_layer_index = VGGT_GEN3R_LAYER_INDICES[-1]
        final_layer = layer_outputs[final_layer_index]
        if final_layer is None:  # Defensive narrowing after aggregator validation.
            raise AssertionError("VGGT-Omega final block disappeared.")
        special_tokens = rearrange(
            final_layer[:, :, :patch_start],
            "(b t) v n d -> b t v n d",
            b=batch_size,
            t=group_steps,
            v=num_views,
        ).contiguous()
        del final_layer
        selected_layer_indices = set(VGGT_GEN3R_LAYER_INDICES)
        for layer_index in range(len(layer_outputs)):
            if layer_index not in selected_layer_indices:
                layer_outputs[layer_index] = None

        for field_index, layer_index in enumerate(VGGT_GEN3R_LAYER_INDICES):
            layer = layer_outputs[layer_index]
            if layer is None:  # Defensive narrowing after aggregator validation.
                raise AssertionError(f"VGGT-Omega block {layer_index} disappeared.")
            patches = layer[:, :, patch_start:]
            expected_patches = (
                batch_size * group_steps,
                num_views,
                VGGT_RAW_GRID_SIZE**2,
                VGGT_GEN3R_FEATURE_DIM,
            )
            if tuple(patches.shape) != expected_patches:
                raise ValueError(
                    f"Omega block {layer_index} patches must be {expected_patches}, "
                    f"got {tuple(patches.shape)}."
                )
            patch_grid = rearrange(
                patches,
                "(b t) v (h w) d -> b t v h w d",
                b=batch_size,
                t=group_steps,
                v=num_views,
                h=VGGT_RAW_GRID_SIZE,
                w=VGGT_RAW_GRID_SIZE,
            )
            composed_field = compose_robotwin_token_views(
                patch_grid,
                output_dtype=output_dtype,
            )
            channel_start = field_index * VGGT_GEN3R_FEATURE_DIM
            channel_end = channel_start + VGGT_GEN3R_FEATURE_DIM
            geometry[..., channel_start:channel_end].copy_(composed_field)
            layer_outputs[layer_index] = None
            del layer, patches, patch_grid, composed_field

        special_field = compose_special_tokens_to_grid(
            special_tokens,
            output_dtype=output_dtype,
        )
        special_start = len(VGGT_GEN3R_LAYER_INDICES) * VGGT_GEN3R_FEATURE_DIM
        geometry[..., special_start:].copy_(special_field)
        del special_tokens, special_field

        if tuple(geometry.shape) != expected_geometry:
            raise AssertionError(
                f"Geometry-only fields must be {expected_geometry}, got {tuple(geometry.shape)}."
            )
        if geometry.dtype != output_dtype or not self._is_encoder_device(geometry):
            raise AssertionError("Geometry-only extraction changed dtype or device unexpectedly.")
        return geometry.contiguous()

    def _encode_all_groups(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # One batched pass, with each time step kept as an independent view group.
        return self._encode_group(images)

    def encode_clip(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``[B,9,V=3,3,H,W]`` to ``[B,10240,9,24,20]``."""
        return self.encode_clip_geometry_only(images, output_dtype=torch.float32)

    def encode_clip_geometry_only(
        self,
        images: torch.Tensor,
        *,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Encode a clip without native head fields, returning FP32 or BF16."""
        self._validate_images(images, expected_steps=9)
        geometry = self._encode_group_geometry_only(
            images,
            output_dtype=output_dtype,
        )
        expected = (
            int(images.shape[0]),
            9,
            *VGGT_GEN3R_EXACT_INPUT_GRID_SIZE,
            VGGT_GEN3R_EXACT_INPUT_CHANNELS,
        )
        if tuple(geometry.shape) != expected:
            raise AssertionError(f"Exact clip fields must be {expected}, got {tuple(geometry.shape)}.")
        return rearrange(geometry, "b t h w c -> b c t h w").contiguous()

    def encode_clip_targets(
        self,
        images: torch.Tensor,
        geometry_heads: "FrozenVGGTOmegaGeometryHeads",
    ) -> dict[str, torch.Tensor]:
        """Encode cache tokens plus frozen Omega camera/depth targets."""
        self._validate_images(images, expected_steps=9)
        geometry, image_fields, special_tokens = self._encode_all_groups(images)
        with torch.inference_mode():
            pose, depth, _ = geometry_heads.predict_from_per_view_fields(
                image_fields,
                special_tokens,
            )
        return {
            "geometry_tokens_x0": rearrange(
                geometry,
                "b t h w c -> b c t h w",
            ).contiguous(),
            "geometry_pose_x0": pose.contiguous(),
            "geometry_depth_x0": depth.contiguous(),
        }

    def encode_current(
        self,
        images: torch.Tensor,
        *,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Encode current three views to ``[B,10240,1,24,20]`` in the requested dtype."""
        if images.ndim != 5:
            raise ValueError(f"Current images must be [B,V,3,H,W], got {tuple(images.shape)}.")
        clip = images.unsqueeze(1)
        self._validate_images(clip, expected_steps=1)
        geometry = self._encode_group_geometry_only(
            clip,
            output_dtype=output_dtype,
        )
        return rearrange(geometry, "b t h w c -> b c t h w").contiguous()

    def encode_legacy_full_clip_geometry_only(
        self,
        images: torch.Tensor,
        *,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Evaluate the pre-per-timestep 27-frame Omega sequence contract.

        All ``T*V`` frames of every sample share one Omega sequence, matching
        the training-time encoder of legacy full-clip checkpoints. Only the
        composed 24x20 fields of every step are returned; no grouping is
        applied between time steps.
        """
        self._validate_geometry_dtype(output_dtype)
        self._validate_images(images, expected_steps=9)
        batch_size = int(images.shape[0])
        flat = rearrange(images, "b t v c h w -> b (t v) c h w").contiguous()
        if tuple(flat.shape[-2:]) != (VGGT_IMAGE_SIZE, VGGT_IMAGE_SIZE):
            flat = rearrange(
                F.interpolate(
                    rearrange(
                        flat.float(),
                        "b (t v) c h w -> (b t v) c h w",
                        t=9,
                        v=VGGT_NUM_VIEWS,
                    ).contiguous(),
                    size=(VGGT_IMAGE_SIZE, VGGT_IMAGE_SIZE),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ),
                "(b t v) c h w -> b (t v) c h w",
                b=batch_size,
                t=9,
                v=VGGT_NUM_VIEWS,
            ).contiguous()
        model = self._load_model()
        flat = flat.to(device=self.device, dtype=torch.float32, non_blocking=True)
        model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            layer_outputs, patch_start = model.aggregator(flat.to(dtype=model_dtype))
        if int(patch_start) != int(model.aggregator.patch_token_start):
            raise RuntimeError("VGGT-Omega patch-token start changed during legacy extraction.")
        composed_fields = []
        for layer_index in VGGT_GEN3R_LAYER_INDICES:
            layer = layer_outputs[layer_index]
            if layer is None:
                raise RuntimeError(f"VGGT-Omega block {layer_index} is missing.")
            patches = rearrange(
                layer[:, :, int(patch_start):].float(),
                "b (t v) (h w) d -> b t v h w d",
                t=9,
                v=VGGT_NUM_VIEWS,
                h=VGGT_RAW_GRID_SIZE,
            )
            composed_fields.append(
                compose_robotwin_token_views(patches, output_dtype=torch.float32)
            )
        special_tokens = rearrange(
            layer_outputs[VGGT_GEN3R_LAYER_INDICES[-1]][:, :, : int(patch_start)].float(),
            "b (t v) n d -> b t v n d",
            t=9,
            v=VGGT_NUM_VIEWS,
        )
        composed_fields.append(compose_special_tokens_to_grid(special_tokens))
        geometry = torch.cat(composed_fields, dim=-1)
        expected = (
            batch_size,
            9,
            *VGGT_GEN3R_EXACT_INPUT_GRID_SIZE,
            VGGT_GEN3R_EXACT_INPUT_CHANNELS,
        )
        if tuple(geometry.shape) != expected:
            raise AssertionError(
                f"Legacy full-clip fields must be {expected}, got {tuple(geometry.shape)}."
            )
        return rearrange(
            geometry.to(dtype=output_dtype),
            "b t h w c -> b c t h w",
        ).contiguous()


__all__ = [
    "FrozenVGGTOmegaGen3RExactFeatureEncoder",
    "VGGT_GEN3R_EXACT_GROUPING_CONTRACT",
    "VGGT_GEN3R_EXACT_INPUT_CHANNELS",
    "VGGT_GEN3R_EXACT_INPUT_GRID_SIZE",
]
