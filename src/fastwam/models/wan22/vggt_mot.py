from __future__ import annotations

import gc
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
DEFAULT_VGGT_WEIGHTS = "checkpoints/VGGT-Omega/vggt_omega_1b_256_text.pt"
VGGT_PATCH_SIZE = 16


def validate_vggt_images(
    images: torch.Tensor,
    *,
    batch_size: int | None = None,
    patch_size: int = VGGT_PATCH_SIZE,
) -> None:
    """Validate a batched, already-preprocessed VGGT image tensor."""
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"`vggt_images` must be a torch.Tensor, got {type(images)}")
    if images.ndim != 5:
        raise ValueError(
            "`vggt_images` must be [B,V,3,H,W], "
            f"got shape {tuple(images.shape)}"
        )
    actual_batch, num_views, channels, height, width = images.shape
    if batch_size is not None and actual_batch != batch_size:
        raise ValueError(
            "VGGT image batch mismatch: "
            f"vggt_images={actual_batch} vs expected={batch_size}"
        )
    if num_views < 1:
        raise ValueError("`vggt_images` must contain at least one view.")
    if channels != 3:
        raise ValueError(
            f"`vggt_images` must contain RGB views with 3 channels, got {channels}."
        )
    if height <= 0 or width <= 0:
        raise ValueError(
            f"`vggt_images` spatial dimensions must be positive, got {(height, width)}."
        )
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            "`vggt_images` height and width must be divisible by the VGGT "
            f"patch size {patch_size}, got {(height, width)}."
        )


def _unwrap_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"VGGT checkpoint must be a mapping, got {type(payload).__name__}.")
    tensor_values = payload and all(isinstance(value, torch.Tensor) for value in payload.values())
    if tensor_values:
        return payload
    wrappers = [key for key in ("state_dict", "model", "module") if key in payload]
    if len(wrappers) != 1:
        raise ValueError("VGGT checkpoint must be a direct state dict or contain exactly one of state_dict/model/module.")
    nested = payload[wrappers[0]]
    if not isinstance(nested, Mapping) or not all(isinstance(value, torch.Tensor) for value in nested.values()):
        raise TypeError(f"VGGT checkpoint wrapper `{wrappers[0]}` is not a tensor state dict.")
    return nested


def load_aggregator_weights(
    aggregator: nn.Module,
    weights_path: str | Path,
    *,
    aggregator_only: bool = False,
) -> None:
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"VGGT weights not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if aggregator_only and isinstance(payload, Mapping) and not payload:
        raise RuntimeError(f"Aggregator-only checkpoint is empty: {path}")
    state = _unwrap_state_dict(payload)
    prefix = "aggregator."
    if aggregator_only:
        aggregator_state = dict(state)
    else:
        aggregator_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not aggregator_state:
        del payload, state
        raise RuntimeError(f"VGGT checkpoint contains no `{prefix}*` parameters: {path}")
    try:
        aggregator.load_state_dict(aggregator_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Strict Aggregator load failed for {path}: {exc}") from exc
    finally:
        del aggregator_state, state, payload
        gc.collect()


class FrozenVGGTBackbone(nn.Module):
    """Aggregator-only VGGT runner returning selected inter-frame K/V.

    ``trainable=False`` preserves the historical frozen, detached behavior.
    """

    def __init__(
        self,
        aggregator: nn.Module,
        callback_layer_indices: Sequence[int],
        *,
        trainable: bool = False,
        enable_reconstruction_heads: bool = False,
        fp32_storage_bf16_compute: bool = False,
        camera_head: nn.Module | None = None,
        dense_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        selected = tuple(int(layer) for layer in callback_layer_indices)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("`callback_layer_indices` must be unique and non-empty.")
        self.aggregator = aggregator
        self.callback_layer_indices = selected
        self.trainable = bool(trainable)
        self.fp32_storage_bf16_compute = bool(fp32_storage_bf16_compute)
        self.camera_head = camera_head
        self.dense_head = dense_head
        if enable_reconstruction_heads and (camera_head is None or dense_head is None):
            from vggt_omega.models.heads import CameraHead, DenseHead
            self.camera_head = CameraHead(dim_in=2048)
            self.dense_head = DenseHead(dim_in=2048, patch_size=int(aggregator.patch_size))
        self.aggregator.requires_grad_(self.trainable)
        self.train(self.trainable)

    def train(self, mode: bool = True):
        effective_mode = bool(mode) if self.trainable else False
        super().train(effective_mode)
        self.aggregator.train(effective_mode)
        return self

    def _autocast_context(self, device: torch.device):
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.fp32_storage_bf16_compute and device.type == "cuda",
        )

    def forward(self, images: torch.Tensor) -> dict[int, dict[str, torch.Tensor | int]]:
        validate_vggt_images(
            images,
            patch_size=int(getattr(self.aggregator, "patch_size", VGGT_PATCH_SIZE)),
        )
        cache: dict[int, dict[str, torch.Tensor | int]] = {}

        def capture(event: Mapping[str, object]) -> None:
            layer = int(event["layer_idx"])
            if layer in cache:
                raise RuntimeError(f"VGGT callback emitted duplicate layer {layer}.")
            cache[layer] = {"layer_idx": layer, "k": event["k"], "v": event["v"]}

        grad_context = nullcontext() if self.trainable else torch.no_grad()
        with grad_context:
            aggregator_kwargs = {
                "capture_only": True,
                "kv_callback": capture,
                "callback_layer_indices": self.callback_layer_indices,
            }
            if self.trainable:
                aggregator_kwargs["detach_callback_kv"] = False
            with self._autocast_context(images.device):
                self.aggregator(images, **aggregator_kwargs)
        missing = set(self.callback_layer_indices) - set(cache)
        if missing:
            raise RuntimeError(f"VGGT Aggregator did not capture requested layers: {sorted(missing)}")
        return cache

    def forward_with_reconstruction(self, images: torch.Tensor) -> dict[str, object]:
        """Run Aggregator once and retain both FastWAM K/V and VGGT head features.

        Tensor flow: ``images [B,V,3,H,W]`` -> Aggregator ->
        ``aggregated_tokens[layer] [B,V,N,D]`` and callback K/V.  The caller
        can feed the K/V to the action bridge and the token list to the
        CameraHead/DenseHead without a second, expensive VGGT pass.
        """
        validate_vggt_images(
            images,
            patch_size=int(getattr(self.aggregator, "patch_size", VGGT_PATCH_SIZE)),
        )
        if self.camera_head is None or self.dense_head is None:
            raise RuntimeError("Reconstruction heads are not enabled on this VGGT backbone.")
        cache: dict[int, dict[str, torch.Tensor | int]] = {}

        def capture(event: Mapping[str, object]) -> None:
            layer = int(event["layer_idx"])
            # Reconstruction heads consume the Aggregator's native FP32
            # tokens, while the action bridge must match ActionDiT's model
            # dtype.  Casting captured K/V is differentiable, so action loss
            # still reaches the trainable Aggregator.
            cache[layer] = {
                "layer_idx": layer,
                "k": event["k"].to(dtype=images.dtype),
                "v": event["v"].to(dtype=images.dtype),
            }

        # DeepSpeed may keep the trainable Aggregator's master parameters in
        # FP32 even when the rest of the model consumes BF16 tensors.  Match
        # the explicit input dtype to the module at this boundary; CUDA
        # autocast can still execute eligible Aggregator kernels in BF16.
        aggregator_parameter = next(self.aggregator.parameters(), None)
        aggregator_images = (
            images
            if aggregator_parameter is None or images.dtype == aggregator_parameter.dtype
            else images.to(dtype=aggregator_parameter.dtype)
        )
        grad_context = nullcontext() if self.trainable else torch.no_grad()
        with grad_context:
            with self._autocast_context(images.device):
                tokens, patch_token_start = self.aggregator(
                    aggregator_images,
                    capture_only=False,
                    kv_callback=capture,
                    callback_layer_indices=self.callback_layer_indices,
                    detach_callback_kv=not self.trainable,
                )
        if tokens is None:
            raise RuntimeError("VGGT Aggregator returned no aggregated tokens.")
        with torch.autocast(device_type=images.device.type, enabled=False):
            tokens = [None if value is None else value.float() for value in tokens]
            pose_enc = self.camera_head(tokens, patch_token_start=patch_token_start)
            depth, depth_conf = self.dense_head(
                tokens,
                images=images.float(),
                patch_token_start=patch_token_start,
            )
        return {
            "kv_by_layer": cache,
            "aggregated_tokens": tokens,
            "patch_token_start": patch_token_start,
            "pose_enc": pose_enc,
            "depth": depth,
            "depth_conf": depth_conf,
        }


class LoopVGGTBackbone(FrozenVGGTBackbone):
    """Run one reconstruction pass followed by recurrent control passes.

    Tensor flow for ``N`` total passes::

        images [B,V,3,H,W]
          -> pass 1 image tokens -> final state [B,V,Ntok,1024]
          -> pass 2 state tokens -> ... -> pass N state tokens
          -> pass N inter-frame K/V only -> ActionDiT bridge

    Reconstruction heads consume the native 2048-wide cached features from
    pass 1.  The recurrent edge uses the 1024-wide final inter-frame state, so
    it needs neither an adapter nor another image/patch embedding operation.
    """

    def __init__(self, *args, num_vggt_forwards: int = 4, **kwargs) -> None:
        if isinstance(num_vggt_forwards, bool) or not isinstance(num_vggt_forwards, int):
            raise TypeError("`num_vggt_forwards` must be an integer.")
        if num_vggt_forwards < 2:
            raise ValueError("`num_vggt_forwards` must be at least 2.")
        super().__init__(*args, **kwargs)
        self.num_vggt_forwards = num_vggt_forwards

    def _run_loop(self, images: torch.Tensor, *, reconstruct: bool) -> dict[str, object]:
        validate_vggt_images(
            images,
            patch_size=int(getattr(self.aggregator, "patch_size", VGGT_PATCH_SIZE)),
        )
        if not hasattr(self.aggregator, "forward_with_state") or not hasattr(
            self.aggregator, "forward_from_state"
        ):
            raise RuntimeError(
                "LoopVGGTBackbone requires an Aggregator with recurrent state APIs."
            )
        if reconstruct and (self.camera_head is None or self.dense_head is None):
            raise RuntimeError("Reconstruction heads are not enabled on this VGGT backbone.")

        cache: dict[int, dict[str, torch.Tensor | int]] = {}

        def capture(event: Mapping[str, object]) -> None:
            layer = int(event["layer_idx"])
            if layer in cache:
                raise RuntimeError(f"Final VGGT pass emitted duplicate layer {layer}.")
            # The Aggregator/head path may be FP32 under DeepSpeed while the
            # shared-attention bridge follows ActionDiT's BF16/FP16 dtype.
            cache[layer] = {
                "layer_idx": layer,
                "k": event["k"].to(dtype=images.dtype),
                "v": event["v"].to(dtype=images.dtype),
            }

        aggregator_parameter = next(self.aggregator.parameters(), None)
        aggregator_images = (
            images
            if aggregator_parameter is None or images.dtype == aggregator_parameter.dtype
            else images.to(dtype=aggregator_parameter.dtype)
        )
        grad_context = nullcontext() if self.trainable else torch.no_grad()
        with grad_context:
            # Pass 1 intentionally disables every callback.  Its native cached
            # features supervise reconstruction, but its K/V must never leak
            # into ActionDiT shared attention.
            first_tokens, patch_token_start, state, patch_grid_size = (
                self.aggregator.forward_with_state(
                    aggregator_images,
                    kv_callback=None,
                    callback_layer_indices=(),
                    detach_callback_kv=not self.trainable,
                )
            )
            for pass_index in range(1, self.num_vggt_forwards):
                is_final = pass_index == self.num_vggt_forwards - 1
                _, state = self.aggregator.forward_from_state(
                    state,
                    patch_grid_size,
                    capture_only=True,
                    kv_callback=capture if is_final else None,
                    callback_layer_indices=(self.callback_layer_indices if is_final else ()),
                    detach_callback_kv=not self.trainable,
                )

        missing = set(self.callback_layer_indices) - set(cache)
        if missing:
            raise RuntimeError(
                "Final VGGT loop pass did not capture requested layers: "
                f"{sorted(missing)}"
            )
        result: dict[str, object] = {"kv_by_layer": cache}
        if reconstruct:
            # Heads remain FP32 by model contract.  They read pass-1 outputs,
            # while the action branch below reads only pass-N K/V.
            with torch.autocast(device_type=images.device.type, enabled=False):
                pose_enc = self.camera_head(
                    first_tokens,
                    patch_token_start=patch_token_start,
                )
                depth, depth_conf = self.dense_head(
                    first_tokens,
                    images=images,
                    patch_token_start=patch_token_start,
                )
            result.update(
                {
                    "aggregated_tokens": first_tokens,
                    "patch_token_start": patch_token_start,
                    "pose_enc": pose_enc,
                    "depth": depth,
                    "depth_conf": depth_conf,
                }
            )
        return result

    def forward(self, images: torch.Tensor) -> dict[int, dict[str, torch.Tensor | int]]:
        return self._run_loop(images, reconstruct=False)["kv_by_layer"]

    def forward_with_reconstruction(self, images: torch.Tensor) -> dict[str, object]:
        return self._run_loop(images, reconstruct=True)


def build_vggt_backbone(
    *,
    weights_path: str = DEFAULT_VGGT_WEIGHTS,
    aggregator_checkpoint_path: str | None = None,
    callback_layer_indices: Sequence[int],
    trainable: bool = False,
    skip_vggt_load_from_pretrain: bool = False,
    enable_reconstruction_heads: bool = False,
    aggregator_factory: Callable[..., nn.Module] | None = None,
) -> FrozenVGGTBackbone:
    if (
        not skip_vggt_load_from_pretrain
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
        and not aggregator_checkpoint_path
    ):
        raise ValueError(
            "Multi-rank VGGT construction requires `aggregator_checkpoint_path`; "
            "full checkpoint fallback is forbidden."
        )
    if aggregator_factory is None:
        from vggt_omega.models.aggregator import Aggregator
        aggregator_factory = Aggregator
    aggregator = aggregator_factory(callback_layer_indices=tuple(callback_layer_indices))
    if skip_vggt_load_from_pretrain:
        logger.info(
            "Skipping VGGT pretrained load (`skip_vggt_load_from_pretrain=True`); "
            "initializing the Aggregator randomly and expecting training or a "
            "model checkpoint to supply its weights."
        )
    else:
        checkpoint_path = aggregator_checkpoint_path or weights_path
        load_aggregator_weights(
            aggregator,
            checkpoint_path,
            aggregator_only=aggregator_checkpoint_path is not None,
        )
    camera_head = dense_head = None
    if enable_reconstruction_heads:
        from vggt_omega.models.heads import CameraHead, DenseHead
        camera_head = CameraHead(dim_in=2048)
        dense_head = DenseHead(dim_in=2048, patch_size=int(aggregator.patch_size))
        if not skip_vggt_load_from_pretrain:
            full_state = _unwrap_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
            camera_state = {
                k[len("camera_head."):]: v
                for k, v in full_state.items()
                if k.startswith("camera_head.")
            }
            dense_state = {
                k[len("dense_head."):]: v
                for k, v in full_state.items()
                if k.startswith("dense_head.")
            }
            if not camera_state or not dense_state:
                raise RuntimeError(
                    "Full VGGT checkpoint is required when reconstruction heads are enabled."
                )
            camera_head.load_state_dict(camera_state, strict=True)
            dense_head.load_state_dict(dense_state, strict=True)
        camera_head.float()
        dense_head.float()
    return FrozenVGGTBackbone(
        aggregator,
        callback_layer_indices,
        trainable=trainable,
        enable_reconstruction_heads=enable_reconstruction_heads,
        camera_head=camera_head,
        dense_head=dense_head,
    )


def build_loop_vggt_backbone(
    *,
    num_vggt_forwards: int = 4,
    **kwargs,
) -> LoopVGGTBackbone:
    """Build the reconstruction-capable recurrent VGGT wrapper."""
    kwargs["enable_reconstruction_heads"] = True
    backbone = build_vggt_backbone(**kwargs)
    return LoopVGGTBackbone(
        backbone.aggregator,
        backbone.callback_layer_indices,
        trainable=backbone.trainable,
        enable_reconstruction_heads=True,
        camera_head=backbone.camera_head,
        dense_head=backbone.dense_head,
        num_vggt_forwards=num_vggt_forwards,
    )


def export_aggregator_checkpoint(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    aggregator_factory: Callable[..., nn.Module] | None = None,
) -> Path:
    """Atomically convert a full checkpoint after strict Aggregator validation."""
    source = Path(weights_path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = _unwrap_state_dict(payload)
    prefix = "aggregator."
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise RuntimeError(f"VGGT checkpoint contains no `{prefix}*` parameters: {source}")
    if aggregator_factory is None:
        from vggt_omega.models.aggregator import Aggregator
        aggregator_factory = Aggregator
    validator = aggregator_factory()
    try:
        validator.load_state_dict(selected, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Exported Aggregator state failed strict validation: {exc}") from exc
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(selected, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    del validator, selected, state, payload
    gc.collect()
    return destination
