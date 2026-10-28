from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
import torch

from experiments.robotwin.fastwam_policy.reconstruction_visualizer import (
    ReconstructionVisualizer,
)


def test_reconstruction_visualizer_writes_panel_npz_metrics_and_ply(
    tmp_path, monkeypatch
):
    # Load official dependencies before installing the narrow decoder stub.
    import fastwam.models.wan22.vggt_official_loss_adapter
    pose_module = types.ModuleType("vggt_omega.utils.pose_enc")

    def encoding_to_camera(value, image_size_hw):
        batch, views = value.shape[:2]
        extrinsics = torch.eye(4).repeat(batch, views, 1, 1)[:, :, :3]
        intrinsics = torch.eye(3).repeat(batch, views, 1, 1)
        intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = 12
        intrinsics[..., 0, 2] = image_size_hw[1] / 2
        intrinsics[..., 1, 2] = image_size_hw[0] / 2
        return extrinsics, intrinsics

    pose_module.encoding_to_camera = encoding_to_camera
    monkeypatch.setitem(sys.modules, "vggt_omega.utils.pose_enc", pose_module)

    visualizer = ReconstructionVisualizer(
        tmp_path,
        save_video=False,
        point_stride=2,
    )
    images = torch.rand(1, 3, 3, 16, 16)
    depth = torch.rand(1, 3, 16, 16, 1) + 0.5
    reconstruction = {
        "depth": depth,
        "depth_conf": torch.rand_like(depth),
        "pose_enc": torch.zeros(1, 3, 9),
    }
    ground_truth = {
        "depth": depth[..., 0] * 1.05,
        "valid": torch.ones(1, 3, 16, 16, dtype=torch.bool),
        "intrinsics": torch.eye(3).repeat(3, 1, 1),
        "extrinsics": torch.eye(4).repeat(3, 1, 1)[:, :3],
    }

    frame_dir = visualizer.save(
        episode_index=0,
        replan_index=0,
        vggt_images=images,
        reconstruction=reconstruction,
        ground_truth=ground_truth,
    )
    visualizer.close()

    assert (frame_dir / "panel.png").is_file()
    assert (frame_dir / "reconstruction.npz").is_file()
    assert (frame_dir / "fused_pointcloud.ply").is_file()
    metrics = json.loads((frame_dir / "metrics.json").read_text())
    assert metrics["point_count"] == 3 * 8 * 8
    assert metrics["depth_mae_m"] > 0
    arrays = np.load(frame_dir / "reconstruction.npz")
    assert arrays["pred_depth"].shape == (3, 16, 16)
    assert arrays["gt_depth"].shape == (3, 16, 16)


@pytest.mark.parametrize("convention", ("metric", "scene_normalized"))
def test_dualfuture_routes_gt_rgb_and_calibration_before_normalization(tmp_path, monkeypatch, convention):
    # Load official dependencies before installing the narrow decoder stub.
    import fastwam.models.wan22.vggt_official_loss_adapter
    pose_module = types.ModuleType("vggt_omega.utils.pose_enc")
    def encoding_to_camera(value, image_size_hw):
        batch, views = value.shape[:2]
        extrinsics = torch.eye(4).repeat(batch, views, 1, 1)[:, :, :3]
        intrinsics = torch.eye(3).repeat(batch, views, 1, 1)
        intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = 12
        return extrinsics, intrinsics
    pose_module.encoding_to_camera = encoding_to_camera
    monkeypatch.setitem(sys.modules, "vggt_omega.utils.pose_enc", pose_module)
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    images = torch.stack([torch.full((3, 16, 16), color) for color in (0., .5, 1.)])[None]
    def prediction(stream, keys):
        return dict(depth=torch.ones(1, len(keys), 16, 16, 1),
                    confidence=torch.ones(1, len(keys), 16, 16),
                    pose=torch.zeros(1, len(keys), 9), stream=stream, target="future",
                    camera_keys=keys, scale_convention=convention)
    reconstruction = {"va1_future": prediction("va1", cameras[2:]),
                      "va2_future": prediction("va2", cameras[:2])}
    depth = torch.stack([torch.full((16, 16), float(value)) for value in (1, 2, 3)])[None]
    intrinsics = torch.eye(3).repeat(1, 3, 1, 1)
    intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = 12
    gt = dict(depth=depth, valid=torch.ones_like(depth, dtype=torch.bool),
              intrinsics=intrinsics, extrinsics=torch.eye(4).repeat(1, 3, 1, 1)[:, :, :3])
    visualizer = ReconstructionVisualizer(tmp_path, save_video=False, point_stride=2)
    frame = visualizer.save(episode_index=0, replan_index=0, vggt_images=images,
        reconstruction=reconstruction, ground_truth={"future": gt}, input_camera_keys=cameras)
    arrays = np.load(frame / "reconstruction.npz")
    assert arrays["gt_va1_future_depth"].shape == (1, 16, 16)
    assert arrays["gt_va2_future_depth"].shape == (2, 16, 16)
    assert arrays["pred_va1_future_camera_keys"].tolist() == cameras[2:]
    metrics = json.loads((frame / "metrics.json").read_text())
    unit = "normalized" if convention == "scene_normalized" else "m"
    assert f"va1_future_cam_right_wrist_depth_mae_{unit}" in metrics
    if convention == "metric":
        assert metrics["va1_future_depth_mae_m"] == 2
        assert metrics["va2_future_depth_mae_m"] == .5
    else:
        assert metrics["va1_future_gt_scene_scale_m"] > 0
    # All VA1 points must use right-camera RGB rather than the first view.
    lines = (frame / "fused_pointcloud_va1_future.ply").read_text().splitlines()
    points = lines[lines.index("end_header") + 1:]
    assert points and all(line.split()[-3:] == ["255", "255", "255"] for line in points)
    # Current online GT must never be compared against future predictions.
    frame = visualizer.save(episode_index=0, replan_index=1, vggt_images=images,
        reconstruction=reconstruction, ground_truth={"current": gt}, input_camera_keys=cameras)
    metrics = json.loads((frame / "metrics.json").read_text())
    assert not any("mae" in key for key in metrics)
