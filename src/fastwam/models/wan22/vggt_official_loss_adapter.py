"""FastWAM tensor boundary for the pinned, otherwise unmodified Omega losses."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

import torch
import yaml

from .vggt_omega_loss.losses.loss import MultitaskLoss
from .vggt_omega_loss.losses.unprojection import unproject_depth_to_points_torch_batch
from .vggt_omega_loss.train_utils.normalization import _normalize_scene_differentiable

OFFICIAL_COMMIT = "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6"


@lru_cache(maxsize=1)
def _default_config():
    with (Path(__file__).with_name("vggt_omega_loss") / "defaults.yaml").open() as f:
        return yaml.safe_load(f)


def official_config(overrides=None):
    """Recursively override official defaults; reject misspelled option names."""
    config = deepcopy(_default_config())

    def merge(target, source):
        for key, value in source.items():
            if key not in target:
                raise ValueError(f"Unknown official loss option: {key}")
            if isinstance(target[key], dict) and isinstance(value, dict):
                merge(target[key], value)
            else:
                target[key] = deepcopy(value)
    from omegaconf import OmegaConf
    if OmegaConf.is_config(overrides):
        overrides = OmegaConf.to_container(overrides, resolve=True)
    merge(config, dict(overrides or {}))
    # The Robotwin reconstruction cache contains no point tracks. Do not silently
    # enable a configured objective while dropping it because labels are absent.
    if config["track"] is not None:
        raise ValueError("Official tracking requires track labels, which FastWAM reconstruction caches do not provide; use official.track=null.")
    if config["camera"] is None or config["depth"] is None:
        raise ValueError("Official camera/depth configs must be dictionaries; disable a term using weight=0.")
    return config


@lru_cache(maxsize=16)
def _criterion(serialized_config):
    # Stateless loss modules have no trainable parameters/buffers. Caching avoids
    # reconstructing them every batch, without registering them in model weights.
    return MultitaskLoss(**json.loads(serialized_config))


def official_batch(gt_depth, gt_valid, gt_intrinsics, gt_extrinsics, is_synthetic=False):
    """Meter-space cache -> upstream batch, without modifying the cached tensors.

    Z depth [B,V,H,W] + K [B,V,3,3] + world-to-camera E [B,V,3,4]
    -> world XYZ [B,V,H,W,3]. Upstream subsequently expresses GT in camera 0
    coordinates and divides GT depth/translation/XYZ by one scale per sample.
    Predictions are deliberately NOT divided by this GT scale.
    """
    depth = gt_depth.float()
    mask = gt_valid.bool() & torch.isfinite(depth) & (depth > 0)
    depth = torch.where(mask, depth, 0.0)
    intrinsics, extrinsics = gt_intrinsics.float(), gt_extrinsics.float()
    b, v, h, w = depth.shape
    synthetic = torch.as_tensor(is_synthetic, device=depth.device, dtype=torch.bool)
    synthetic = synthetic.expand(b) if synthetic.ndim == 0 else synthetic
    if synthetic.shape != (b,):
        raise ValueError(f"is_synthetic must be scalar or [B], got {synthetic.shape}")
    batch = {
        # Only the image shape is read by official camera/point losses. An
        # expanded scalar avoids allocating another full RGB batch.
        "images": depth.new_zeros(()).expand(b, v, 3, h, w),
        "depths": depth, "point_masks": mask,
        "intrinsics": intrinsics, "extrinsics": extrinsics,
        "world_points": unproject_depth_to_points_torch_batch(depth, extrinsics, intrinsics),
        "valid_seq_mask": torch.ones(b, device=depth.device, dtype=torch.bool),
        "depth_train_mask": torch.ones(b, device=depth.device, dtype=torch.bool),
        "is_synthetic": synthetic,
    }
    return batch


def compute_official_loss(*, depth, depth_conf, pose_enc, gt_depth, gt_valid,
                          gt_intrinsics, gt_extrinsics, config=None,
                          schedule_progress=None, is_synthetic=False):
    """Run the official objective in FP32 even inside a BF16 training autocast.

    The final camera stage [B,V,9] becomes a one-element pose_enc_list. Views
    belong to one routed time branch; current/future branches normalize their GT
    independently. schedule_progress is successful optimizer updates / planned
    updates, shared across gradient accumulation microbatches and restored on resume.
    """
    if schedule_progress is None or not 0 <= float(schedule_progress) <= 1:
        raise ValueError("Official loss requires schedule_progress in [0,1].")
    options = official_config(config)
    with torch.autocast(device_type=depth.device.type, enabled=False):
        graph_zero = sum(x.reshape(-1)[:0].float().sum() for x in (depth, depth_conf, pose_enc))
        if depth.shape[0] == 0:
            return graph_zero, {key: graph_zero.detach() for key in (
                "loss_vggt_camera", "loss_vggt_depth", "loss_vggt_point",
                "vggt_scene_scale", "vggt_valid_depth_ratio")}
        batch = official_batch(gt_depth, gt_valid, gt_intrinsics, gt_extrinsics, is_synthetic)
        with torch.no_grad():
            scale, _ = _normalize_scene_differentiable(
                batch["extrinsics"], batch["world_points"], batch["depths"],
                batch["point_masks"], scale_only=True)
        predictions = {"depth": depth.float(), "depth_conf": depth_conf.float(),
                       "pose_enc_list": [pose_enc.float()]}
        result = _criterion(json.dumps(options, sort_keys=True))(
            predictions, batch, schedule_progress=float(schedule_progress))
        metrics = {
            f"loss_vggt_{term}": (result.get(f"loss_{term}", graph_zero)
                                  * (options[term] or {}).get("weight", 0)).detach()
            for term in ("camera", "depth", "point")
        }
        metrics.update(vggt_scene_scale=scale.mean().detach(),
                       vggt_valid_depth_ratio=batch["point_masks"].float().mean().detach())
        return result["loss_objective"] + graph_zero, metrics
