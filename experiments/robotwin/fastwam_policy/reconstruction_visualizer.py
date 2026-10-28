"""Save Action-VGGT reconstruction outputs during RoboTwin evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw


CAMERA_NAMES = ("head", "left_wrist", "right_wrist")


def _to_rgb_uint8(images: torch.Tensor) -> np.ndarray:
    images = images.detach().float().cpu()
    if images.ndim == 5:
        images = images[0]
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected VGGT RGB [V,3,H,W], got {tuple(images.shape)}")
    array = images.permute(0, 2, 3, 1).numpy()
    if float(np.nanmin(array)) < -0.01:
        array = (array + 1.0) * 0.5
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def _head_map(value: Any, name: str) -> np.ndarray:
    tensor = torch.as_tensor(value).detach().float().cpu()
    if tensor.ndim == 5 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected reconstruction {name} [V,H,W], got {tuple(tensor.shape)}")
    return tensor.numpy()


def _reconstruction_field(reconstruction: Mapping[str, Any], *names: str) -> Any:
    """Return the first available field across old and new reconstruction ABIs."""
    for name in names:
        if name in reconstruction:
            return reconstruction[name]
    raise KeyError(f"Reconstruction is missing all of {names!r}.")


def _as_streams(reconstruction: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Normalize legacy single-stream and MoT dual-stream outputs."""
    # New stage outputs carry explicit target metadata. Preserve legacy parsing below.
    stages = {key: value for key, value in reconstruction.items()
              if key.startswith("coarse") and key[6:].isdigit() and isinstance(value, Mapping)
              and value.get("target") in ("current", "future")}
    if stages:
        return stages
    stream_names = ("current", "future", "high_current", "low_current", "high_future", "low_future", "va1_future", "va2_future")
    if any(name in reconstruction for name in stream_names):
        streams = {
            key: value
            for key, value in reconstruction.items()
            if key in stream_names and isinstance(value, Mapping)
        }
        if not streams:
            raise ValueError(
                "Reconstruction stream mappings must contain a known stream name."
            )
        return streams
    return {"current": reconstruction}


def _route_ground_truth(gt, camera_keys, input_camera_keys):
    """Select RGB-D/calibration views before scene-scale normalization."""
    if camera_keys is None:
        return gt
    source_keys = list(gt.get("camera_keys", input_camera_keys))
    indices = [source_keys.index(key) for key in camera_keys]
    result = dict(gt)
    for key in ("depth", "valid", "intrinsics", "extrinsics"):
        value = torch.as_tensor(gt[key])
        axis = 1 if value.ndim == 4 else 0
        result[key] = value.index_select(axis, torch.tensor(indices, device=value.device))
    result["camera_keys"] = list(camera_keys)
    return result


def _finite_percentiles(values: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    selected = values[valid & np.isfinite(values)]
    if selected.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(selected, (2.0, 98.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(selected))
        high = float(np.nanmax(selected))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _colorize(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = valid & np.isfinite(values)
    low, high = _finite_percentiles(values, valid)
    x = np.clip((values - low) / (high - low), 0.0, 1.0)
    x = np.where(np.isfinite(x), x, 0.0)
    # Compact blue -> cyan -> yellow -> red map without importing matplotlib.
    anchors = np.asarray(
        [[20, 20, 90], [20, 150, 220], [40, 220, 150], [245, 220, 50], [190, 20, 20]],
        dtype=np.float32,
    )
    position = x * (len(anchors) - 1)
    left = np.floor(position).astype(np.int64)
    right = np.minimum(left + 1, len(anchors) - 1)
    weight = (position - left)[..., None]
    rgb = anchors[left] * (1.0 - weight) + anchors[right] * weight
    rgb[~valid] = 0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _tile_row(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    labelled = []
    for image, label in zip(images, labels, strict=True):
        canvas = Image.new("RGB", (image.shape[1], image.shape[0] + 22), "black")
        canvas.paste(Image.fromarray(image), (0, 22))
        ImageDraw.Draw(canvas).text((5, 5), label, fill="white")
        labelled.append(np.asarray(canvas))
    return np.concatenate(labelled, axis=1)


def _write_ascii_ply(
    path: Path,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    rgb: np.ndarray,
    stride: int,
    scale_convention: str | None = None,
) -> int:
    points, colors = [], []
    for view in range(depth.shape[0]):
        d = depth[view, ::stride, ::stride]
        h, w = depth.shape[1:]
        yy, xx = np.meshgrid(
            np.arange(0, h, stride, dtype=np.float32) + 0.5,
            np.arange(0, w, stride, dtype=np.float32) + 0.5,
            indexing="ij",
        )
        k = intrinsics[view]
        xyz_camera = np.stack(
            (
                (xx - k[0, 2]) * d / max(float(k[0, 0]), 1e-6),
                (yy - k[1, 2]) * d / max(float(k[1, 1]), 1e-6),
                d,
            ),
            axis=-1,
        )
        rotation = extrinsics[view, :3, :3]
        translation = extrinsics[view, :3, 3]
        xyz_world = (xyz_camera - translation) @ rotation
        valid = np.isfinite(xyz_world).all(axis=-1) & np.isfinite(d) & (d > 0)
        points.append(xyz_world[valid])
        colors.append(rgb[view, ::stride, ::stride][valid])
    point_array = np.concatenate(points, axis=0) if points else np.empty((0, 3))
    color_array = np.concatenate(colors, axis=0) if colors else np.empty((0, 3))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        if scale_convention is not None:
            handle.write(f"comment scale_convention {scale_convention}\n")
        handle.write(f"element vertex {len(point_array)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(point_array, color_array, strict=True):
            handle.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
    return int(len(point_array))


class ReconstructionVisualizer:
    """Persist per-replan geometry and a compact diagnostic video."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        save_panel: bool = True,
        save_npz: bool = True,
        save_pointcloud: bool = True,
        save_video: bool = True,
        video_fps: int = 2,
        point_stride: int = 4,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_panel = bool(save_panel)
        self.save_npz = bool(save_npz)
        self.save_pointcloud = bool(save_pointcloud)
        self.save_video = bool(save_video)
        self.video_fps = max(1, int(video_fps))
        self.point_stride = max(1, int(point_stride))
        self._video_writer = None
        self._video_episode: int | None = None

    def close(self) -> None:
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
            self._video_episode = None

    def _append_video(self, episode_index: int, episode_dir: Path, panel: np.ndarray) -> None:
        if not self.save_video:
            return
        if self._video_episode != episode_index:
            self.close()
            import imageio.v2 as imageio

            self._video_writer = imageio.get_writer(
                episode_dir / "reconstruction.mp4",
                fps=self.video_fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=2,
            )
            self._video_episode = episode_index
        self._video_writer.append_data(panel)

    def save(
        self,
        *,
        episode_index: int,
        replan_index: int,
        vggt_images: torch.Tensor,
        reconstruction: Mapping[str, Any],
        ground_truth: Mapping[str, Any] | None = None,
        input_camera_keys: list[str] | tuple[str, ...] | None = None,
    ) -> Path:
        from vggt_omega.utils.pose_enc import encoding_to_camera

        rgb = _to_rgb_uint8(vggt_images)
        source_keys = list(input_camera_keys) if input_camera_keys is not None else list(
            ("cam_high", "cam_left_wrist", "cam_right_wrist")[:rgb.shape[0]])
        if len(source_keys) != rgb.shape[0]:
            raise ValueError("Input camera keys must match RGB view count.")
        streams = _as_streams(reconstruction)
        predictions: dict[str, dict[str, np.ndarray]] = {}
        stream_rgb, stream_camera_names = {}, {}
        for stream_name, stream in streams.items():
            depth = _head_map(_reconstruction_field(stream, "depth"), f"{stream_name}.depth")
            cameras = stream.get("camera_keys")
            indices = ([source_keys.index(key) for key in cameras] if cameras is not None
                       else list(range(depth.shape[0])))
            if len(indices) != depth.shape[0]:
                raise ValueError(f"{stream_name} camera keys must match predicted view count.")
            stream_rgb[stream_name] = rgb[indices]
            stream_camera_names[stream_name] = (list(cameras) if cameras is not None
                                              else list(CAMERA_NAMES[:depth.shape[0]]))
            confidence = _head_map(
                _reconstruction_field(stream, "confidence", "depth_conf"),
                f"{stream_name}.depth_conf",
            )
            pose = torch.as_tensor(
                _reconstruction_field(stream, "pose", "pose_enc")
            ).float()
            pred_extrinsics, pred_intrinsics = encoding_to_camera(
                pose,
                image_size_hw=depth.shape[-2:],
            )
            predictions[stream_name] = {
                "depth": depth,
                "confidence": confidence,
                "pose": pose.detach().cpu().numpy(),
                "intrinsics": pred_intrinsics[0].detach().cpu().numpy(),
                "extrinsics": pred_extrinsics[0].detach().cpu().numpy(),
            }

        episode_dir = self.output_dir / f"episode_{episode_index:03d}"
        frame_dir = episode_dir / f"replan_{replan_index:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        supplied_gt = _as_streams(ground_truth) if ground_truth is not None else {}
        gt_streams = {}
        for name, stream in streams.items():
            target = stream.get("target", name)
            gt = supplied_gt.get(name, supplied_gt.get(target))
            if gt is not None:
                gt_streams[name] = _route_ground_truth(gt, stream.get("camera_keys"), source_keys)
        metrics: dict[str, Any] = {
            "episode": int(episode_index),
            "replan": int(replan_index),
        }
        # Official defaults train dimensionless scene-normalized geometry. Record
        # units even when GT is unavailable (notably future predictions in rollout).
        for name, stream in streams.items():
            if "scale_convention" in stream:
                metrics[f"{name}_scale_convention"] = stream["scale_convention"]
            for field in ("stream", "route", "target", "camera_keys"):
                if field in stream:
                    metrics[f"{name}_{field}"] = stream[field]
        gt_arrays: dict[str, dict[str, np.ndarray]] = {}
        for stream_name, gt in gt_streams.items():
            gt_depth = _head_map(gt["depth"], f"GT {stream_name} depth")
            valid_tensor = torch.as_tensor(gt["valid"]).bool().cpu()
            if valid_tensor.ndim == 4:
                valid_tensor = valid_tensor[0]
            if valid_tensor.ndim != 3:
                raise ValueError(
                    f"Expected {stream_name} GT validity [V,H,W], got {tuple(valid_tensor.shape)}"
                )
            gt_arrays[stream_name] = {
                "depth": gt_depth,
                "valid": valid_tensor.numpy(),
                "intrinsics": torch.as_tensor(gt["intrinsics"]).float().cpu().numpy(),
                "extrinsics": torch.as_tensor(gt["extrinsics"]).float().cpu().numpy(),
            }
            prediction = predictions.get(stream_name)
            if prediction is None:
                continue
            convention = streams[stream_name].get("scale_convention", "metric")
            if convention == "arbitrary":
                # Optional upstream prediction normalization does not define metric
                # output scale. Do not report meter errors for these raw outputs.
                gt_arrays.pop(stream_name)
                continue
            unit = "m"
            if convention == "scene_normalized":
                from fastwam.models.wan22.vggt_official_loss_adapter import official_batch
                from fastwam.models.wan22.vggt_omega_loss.losses.gt_normalization import normalize_gt_batch
                from fastwam.models.wan22.vggt_omega_loss.train_utils.normalization import _normalize_scene_differentiable

                def batched_matrix(value):
                    tensor = torch.as_tensor(value).float()
                    return tensor.unsqueeze(0) if tensor.ndim == 3 else tensor

                batch = official_batch(torch.from_numpy(gt_depth)[None], valid_tensor[None],
                                       batched_matrix(gt["intrinsics"]), batched_matrix(gt["extrinsics"]))
                scale, valid_sequence = _normalize_scene_differentiable(
                    batch["extrinsics"], batch["world_points"], batch["depths"],
                    batch["point_masks"], scale_only=True)
                metrics[f"{stream_name}_gt_scene_scale_m"] = float(scale[0])
                batch = normalize_gt_batch(batch, "global")
                gt_depth = batch["depths"][0].numpy()
                gt_arrays[stream_name].update(depth=gt_depth,
                    extrinsics=batch["extrinsics"][0].numpy(),
                    intrinsics=batch["intrinsics"][0].numpy(),
                    valid=(batch["point_masks"][0] & valid_sequence[0]).numpy())
                unit = "normalized"
            valid = (
                gt_arrays[stream_name]["valid"]
                & np.isfinite(gt_depth)
                & (gt_depth > 0)
                & np.isfinite(prediction["depth"])
                & (prediction["depth"] > 0)
            )
            if valid.any():
                difference = np.abs(prediction["depth"] - gt_depth)
                prefix = "" if stream_name == "current" and len(streams) == 1 else f"{stream_name}_"
                metrics.update(
                    **{
                        f"{prefix}depth_mae_{unit}": float(difference[valid].mean()),
                        f"{prefix}depth_rmse_{unit}": float(np.sqrt(np.square(difference[valid]).mean())),
                        f"{prefix}depth_abs_rel": float((difference[valid] / gt_depth[valid]).mean()),
                        f"{prefix}valid_depth_ratio": float(valid.mean()),
                    }
                )
                for view_index, camera_name in enumerate(stream_camera_names[stream_name]):
                    view_valid = valid[view_index]
                    if not view_valid.any():
                        continue
                    view_difference = difference[view_index]
                    view_gt = gt_depth[view_index]
                    metrics.update(
                        **{
                            f"{prefix}{camera_name}_depth_mae_{unit}": float(
                                view_difference[view_valid].mean()
                            ),
                            f"{prefix}{camera_name}_depth_abs_rel": float(
                                (view_difference[view_valid] / view_gt[view_valid]).mean()
                            ),
                        }
                    )

        if self.save_npz:
            arrays = {"rgb": rgb}
            for stream_name, prediction in predictions.items():
                prefix = "" if stream_name == "current" and len(streams) == 1 else f"{stream_name}_"
                arrays.update(
                    **{
                        f"pred_{prefix}depth": prediction["depth"],
                        f"pred_{prefix}depth_conf": prediction["confidence"],
                        f"pred_{prefix}pose_enc": prediction["pose"],
                        f"pred_{prefix}intrinsics": prediction["intrinsics"],
                        f"pred_{prefix}extrinsics": prediction["extrinsics"],
                    }
                )
                if "scale_convention" in streams[stream_name]:
                    arrays[f"pred_{prefix}scale_convention"] = np.asarray(streams[stream_name]["scale_convention"])
                    scale_key = f"{stream_name}_gt_scene_scale_m"
                    if scale_key in metrics:
                        arrays[f"gt_{prefix}scene_scale_m"] = np.asarray(metrics[scale_key])
                for field in ("stream", "route", "target", "camera_keys"):
                    if field in streams[stream_name]:
                        arrays[f"pred_{prefix}{field}"] = np.asarray(streams[stream_name][field])
                if stream_name in gt_arrays:
                    gt = gt_arrays[stream_name]
                    arrays.update(
                        **{
                            f"gt_{prefix}depth": gt["depth"],
                            f"gt_{prefix}valid": gt["valid"],
                            f"gt_{prefix}intrinsics": gt["intrinsics"],
                            f"gt_{prefix}extrinsics": gt["extrinsics"],
                        }
                    )
            np.savez_compressed(frame_dir / "reconstruction.npz", **arrays)

        if self.save_pointcloud:
            for stream_name, prediction in predictions.items():
                suffix = "" if stream_name == "current" and len(streams) == 1 else f"_{stream_name}"
                metrics[f"{stream_name}_point_count" if len(streams) > 1 else "point_count"] = _write_ascii_ply(
                    frame_dir / f"fused_pointcloud{suffix}.ply",
                    prediction["depth"], prediction["intrinsics"], prediction["extrinsics"],
                    stream_rgb[stream_name], self.point_stride, streams[stream_name].get("scale_convention"),
                )

        camera_names = list(CAMERA_NAMES[:rgb.shape[0]]) if input_camera_keys is None else source_keys
        rows = [
            _tile_row([image for image in rgb], [f"{name}: RGB" for name in camera_names]),
        ]
        for stream_name, prediction in predictions.items():
            camera_names = stream_camera_names[stream_name]
            title = "predicted depth" if len(predictions) == 1 else f"{stream_name} predicted depth"
            rows.append(_tile_row(
                [_colorize(prediction["depth"][i], np.isfinite(prediction["depth"][i]) & (prediction["depth"][i] > 0)) for i in range(prediction["depth"].shape[0])],
                [f"{name}: {title}" for name in camera_names],
            ))
            rows.append(_tile_row(
                [_colorize(prediction["confidence"][i]) for i in range(prediction["confidence"].shape[0])],
                [f"{name}: {stream_name} confidence" if len(predictions) > 1 else f"{name}: confidence" for name in camera_names],
            ))
            if stream_name in gt_arrays:
                gt = gt_arrays[stream_name]
                rows.append(_tile_row(
                    [_colorize(gt["depth"][i], gt["valid"][i]) for i in range(gt["depth"].shape[0])],
                    [f"{name}: {stream_name} GT depth" if len(predictions) > 1 else f"{name}: GT depth" for name in camera_names],
                ))
                rows.append(_tile_row(
                    [_colorize(np.abs(prediction["depth"][i] - gt["depth"][i]), gt["valid"][i]) for i in range(gt["depth"].shape[0])],
                    [f"{name}: {stream_name} absolute error" if len(predictions) > 1 else f"{name}: absolute error" for name in camera_names],
                ))
        panel_width = max(row.shape[1] for row in rows)
        rows = [np.pad(row, ((0, 0), (0, panel_width - row.shape[1]), (0, 0))) for row in rows]
        panel = np.concatenate(rows, axis=0)
        if panel.shape[0] % 2:
            panel = np.pad(panel, ((0, 1), (0, 0), (0, 0)))
        if panel.shape[1] % 2:
            panel = np.pad(panel, ((0, 0), (0, 1), (0, 0)))
        if self.save_panel:
            Image.fromarray(panel).save(frame_dir / "panel.png")
        self._append_video(episode_index, episode_dir, panel)

        (frame_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return frame_dir


__all__ = ["ReconstructionVisualizer"]
