"""Backward-compatible legacy geometry loss and official VGGT-Omega dispatcher.

The adapter deliberately keeps the E05 trainer contract (``loss, loss_dict``)
while documenting the tensor flow.  Inputs use ``[B,V,...]``: views are the
three synchronized Robotwin cameras, not action timesteps.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vggt_omega.utils.pose_enc import encoding_to_camera


def _unproject(depth, intrinsics):
    """Convert optical-axis Z depth ``[B,V,H,W]`` to camera-frame XYZ points."""
    b, v, h, w = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=torch.float32) + 0.5,
        torch.arange(w, device=depth.device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    fx = intrinsics[..., 0, 0][..., None, None]
    fy = intrinsics[..., 1, 1][..., None, None]
    cx = intrinsics[..., 0, 2][..., None, None]
    cy = intrinsics[..., 1, 2][..., None, None]
    x = (xx - cx) * depth / fx.clamp_min(1e-6)
    y = (yy - cy) * depth / fy.clamp_min(1e-6)
    return torch.stack((x, y, depth), dim=-1)


def _to_world(points_cam, extrinsics):
    """Apply camera-from-world extrinsics inverse to ``[...,H,W,3]`` points."""
    r = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3]
    # X_cam = R X_world + t, hence X_world = R^T (X_cam - t).
    centered = points_cam - t[..., None, None, :]
    return torch.einsum("bvhwc,bvck->bvhwk", centered, r)


def _frame0_relative(extrinsics):
    """Return camera poses expressed relative to view 0 (same OpenCV convention)."""
    r0 = extrinsics[:, :1, :3, :3]
    t0 = extrinsics[:, :1, :3, 3]
    r = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3]
    rr = r @ r0.transpose(-1, -2)
    tt = t - (rr @ t0[..., None]).squeeze(-1)
    return torch.cat((rr, tt[..., None]), dim=-1)


def _scene_scale(depth, intrinsics, extrinsics, valid):
    points = _to_world(_unproject(depth, intrinsics), extrinsics)
    # Normalize in the first camera frame.  ``_to_world`` returns row vectors;
    # use an explicit einsum here to avoid the subtle broadcasting ambiguity of
    # ``matmul`` when B and V are both present.
    r0 = extrinsics[:, 0, :3, :3]
    t0 = extrinsics[:, 0, :3, 3]
    points0 = torch.einsum("bvhwc,bkc->bvhwk", points, r0) + t0[:, None, None, None, :]
    norm = points0.norm(dim=-1)
    denom = valid.sum(dim=(1, 2, 3)).clamp_min(1).float()
    scale = (norm * valid.float()).sum(dim=(1, 2, 3)) / denom
    return scale.clamp_min(1e-6).detach()


def compute_legacy_vggt_reconstruction_loss(
    *,
    depth,
    depth_conf,
    pose_enc,
    gt_depth,
    gt_valid,
    gt_intrinsics,
    gt_extrinsics,
    lambda_camera=1.0,
    lambda_depth=1.0,
    lambda_point=1.0,
    use_conf_loss=False,
    conf_alpha=0.2,
    gradient_weight=0.0,
):
    """Compute depth + camera + world-point losses with global scene scaling.

    ``depth`` is DenseHead output ``[B,V,H,W,1]``; labels are linear Z-depth
    ``[B,V,H,W]``.  GT scale is computed once per sample and shared by all
    three terms. Both predictions and GT are divided by that scale here, so
    legacy predictions retain metric units; this differs from official defaults.
    """
    if depth.ndim == 5:
        depth = depth[..., 0]
    if depth_conf.ndim == 5:
        depth_conf = depth_conf[..., 0]
    valid = gt_valid.bool() & torch.isfinite(gt_depth) & (gt_depth > 0)
    gt_depth = torch.where(valid, gt_depth.float(), torch.zeros_like(gt_depth.float()))
    scale = _scene_scale(gt_depth, gt_intrinsics, gt_extrinsics, valid)
    pred_depth = depth.float() / scale[:, None, None, None]
    target_depth = gt_depth / scale[:, None, None, None]

    # Depth regression is log-relative, with optional VGGT confidence term.
    safe_pred = pred_depth.clamp_min(1e-3)
    safe_target = target_depth.clamp_min(1e-3)
    residual = (safe_pred.log() - safe_target.log()).abs()
    denom = valid.float().sum().clamp_min(1.0)
    loss_depth = (residual * valid.float()).sum() / denom
    if use_conf_loss:
        conf = depth_conf.float().clamp_min(1e-3)
        loss_depth = loss_depth + (
            (conf * residual - conf_alpha * conf.log()) * valid.float()
        ).sum() / denom

    if gradient_weight:
        pred_log = safe_pred.log()
        dx = pred_log[..., :, 1:] - pred_log[..., :, :-1]
        dy = pred_log[..., 1:, :] - pred_log[..., :-1, :]
        tx = safe_target[..., :, 1:].log() - safe_target[..., :, :-1].log()
        ty = safe_target[..., 1:, :].log() - safe_target[..., :-1, :].log()
        mx = valid[..., :, 1:] & valid[..., :, :-1]
        my = valid[..., 1:, :] & valid[..., :-1, :]
        grad = ((dx - tx).abs() * mx).sum() + ((dy - ty).abs() * my).sum()
        grad /= (mx.sum() + my.sum()).clamp_min(1).float()
        loss_depth = loss_depth + float(gradient_weight) * grad

    # Decode prediction to camera matrices. CameraHead outputs [T, q, fov_h, fov_w].
    pred_extrinsics, pred_intrinsics = encoding_to_camera(
        pose_enc.float(), image_size_hw=gt_depth.shape[-2:]
    )
    pred_extrinsics = _frame0_relative(pred_extrinsics)
    target_extrinsics = _frame0_relative(gt_extrinsics.float())
    pred_extrinsics = pred_extrinsics.clone()
    pred_extrinsics[..., :3, 3] /= scale[:, None, None]
    target_extrinsics = target_extrinsics.clone()
    target_extrinsics[..., :3, 3] /= scale[:, None, None]
    rot_loss = F.mse_loss(pred_extrinsics[..., :3, :3], target_extrinsics[..., :3, :3])
    trans_loss = F.smooth_l1_loss(pred_extrinsics[..., :3, 3], target_extrinsics[..., :3, 3])
    focal_pred = pred_intrinsics[..., :2, :2].diagonal(dim1=-2, dim2=-1)
    focal_gt = gt_intrinsics.float()[..., :2, :2].diagonal(dim1=-2, dim2=-1)
    focal_loss = F.smooth_l1_loss(focal_pred / focal_pred.detach().abs().mean().clamp_min(1e-3), focal_gt / focal_gt.detach().abs().mean().clamp_min(1e-3))
    loss_camera = rot_loss + 2.0 * trans_loss + focal_loss

    pred_points = _to_world(_unproject(pred_depth, gt_intrinsics.float()), pred_extrinsics)
    gt_points = _to_world(_unproject(target_depth, gt_intrinsics.float()), target_extrinsics)
    point_error = (pred_points - gt_points).abs().mean(dim=-1)
    loss_point = (point_error * valid.float()).sum() / denom
    total = lambda_camera * loss_camera + lambda_depth * loss_depth + lambda_point * loss_point
    return total, {
        "loss_vggt_camera": (lambda_camera * loss_camera).detach(),
        "loss_vggt_depth": (lambda_depth * loss_depth).detach(),
        "loss_vggt_point": (lambda_point * loss_point).detach(),
        "vggt_scene_scale": scale.mean().detach(),
        "vggt_valid_depth_ratio": valid.float().mean().detach(),
    }


def canonical_reconstruction_config(config):
    """Keep historical legacy checkpoint metadata unchanged when adding a selector."""
    if config is None:
        return None
    from omegaconf import OmegaConf
    result = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    backend = result.get("backend", "legacy")
    if backend not in ("legacy", "official"):
        raise ValueError(f"Unknown reconstruction loss backend: {backend!r}")
    if backend == "legacy":
        result.pop("backend", None)
        result.pop("official", None)
    return result


def compute_vggt_reconstruction_loss(*, backend="legacy", official=None,
                                     schedule_progress=None, is_synthetic=False, **kwargs):
    """Select loss without changing historical formulas or their defaults.

    Official settings live only in ``official``; legacy lambda/confidence/gradient
    knobs deliberately do not override the upstream defaults. Predictions and GT
    use [B,V,H,W,(1)] depth, [B,V,9] pose, [B,V,3,3/4] camera matrices.
    """
    if backend == "legacy":
        return compute_legacy_vggt_reconstruction_loss(**kwargs)
    if backend != "official":
        raise ValueError(f"Unknown reconstruction loss backend: {backend!r}")
    from .vggt_official_loss_adapter import compute_official_loss
    inputs = {key: kwargs.pop(key) for key in (
        "depth", "depth_conf", "pose_enc", "gt_depth", "gt_valid",
        "gt_intrinsics", "gt_extrinsics")}
    legacy_options = {"lambda_camera", "lambda_depth", "lambda_point", "use_conf_loss",
                      "conf_alpha", "gradient_weight"}
    if set(kwargs) - legacy_options:
        raise TypeError(f"Unknown reconstruction options: {sorted(set(kwargs) - legacy_options)}")
    return compute_official_loss(**inputs, config=official,
                                schedule_progress=schedule_progress, is_synthetic=is_synthetic)
