"""Utilities for replaying RoboTwin trajectories into an RGB-D-mask dataset.

This module intentionally does not import SAPIEN or RoboTwin at import time.  The
render entry point imports them lazily after a worker has selected its GPU.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import av
import cv2
import numpy as np
import pyarrow.parquet as pq


HF_REPO_ID = "TianxingChen/RoboTwin2.0"
HF_REVISION = "c15cc97be71e35244b6605d2d84c187f8565cc4d"
MAPPING_VERSION = 1
MODALITIES_VERSION = 3
FPS = 50
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
EPISODES_PER_TASK = 550
SETTING_COUNTS = {"clean": 50, "randomized": 500}

CAMERA_SPECS = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}

# This is the official Hugging Face task order and the order used by the
# existing 27,500-episode FastWAM LeRobot dataset.
OFFICIAL_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)


@dataclass(frozen=True)
class EpisodeMapping:
    task: str
    setting: str
    source_episode: int
    seed: int
    global_episode: int
    expected_frames: int
    action_sha256: str
    source_relpath: str

    @property
    def source_id(self) -> str:
        return f"{self.task}/{self.setting}/episode{self.source_episode}"


def episode_chunk(episode_index: int) -> int:
    return int(episode_index) // 1000


def parquet_path(root: str | Path, episode_index: int) -> Path:
    return (
        Path(root)
        / "data"
        / f"chunk-{episode_chunk(episode_index):03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def video_path(root: str | Path, episode_index: int, camera_key: str) -> Path:
    return (
        Path(root)
        / "videos"
        / f"chunk-{episode_chunk(episode_index):03d}"
        / f"observation.images.{camera_key}"
        / f"episode_{episode_index:06d}.mp4"
    )


def sidecar_path(root: str | Path, episode_index: int) -> Path:
    return (
        Path(root)
        / "modalities"
        / f"chunk-{episode_chunk(episode_index):03d}"
        / f"episode_{episode_index:06d}.zip"
    )


def marker_path(root: str | Path, episode_index: int) -> Path:
    return (
        Path(root)
        / "meta"
        / "render_status"
        / f"chunk-{episode_chunk(episode_index):03d}"
        / f"episode_{episode_index:06d}.json"
    )


def action_fingerprint(action: np.ndarray) -> str:
    """Return a stable digest for a [T, 14] float32 action sequence."""
    value = np.asarray(action, dtype="<f4", order="C")
    if value.ndim != 2 or value.shape[1] != 14:
        raise ValueError(f"Expected action shape [T, 14], got {value.shape}.")
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def read_parquet_action(path: str | Path) -> np.ndarray:
    table = pq.read_table(path, columns=["action"])
    return np.asarray(table["action"].to_pylist(), dtype=np.float32)


def sha256_file(path: str | Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def depth_to_uint16(position: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    """Convert a SAPIEN Position texture into millimetre uint16 depth."""
    position = np.asarray(position)
    if position.ndim != 3 or position.shape[-1] < 3:
        raise ValueError(f"Expected Position [H,W,>=3], got {position.shape}.")
    depth_mm = -position[..., 2].astype(np.float64) * 1000.0
    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    if alpha is not None:
        alpha = np.asarray(alpha)
        if alpha.shape != depth_mm.shape:
            raise ValueError(f"Alpha shape {alpha.shape} does not match depth {depth_mm.shape}.")
        valid &= alpha > 0
    result = np.zeros(depth_mm.shape, dtype=np.uint16)
    result[valid] = np.clip(np.rint(depth_mm[valid]), 1, np.iinfo(np.uint16).max).astype(
        np.uint16
    )
    return result


def actor_ids_to_robot_mask(actor_ids: np.ndarray, robot_actor_ids: Sequence[int]) -> np.ndarray:
    """Build a binary mask without truncating SAPIEN's uint32 actor IDs."""
    actor_ids = np.asarray(actor_ids)
    if actor_ids.ndim != 2:
        raise ValueError(f"Expected actor IDs [H,W], got {actor_ids.shape}.")
    ids = np.unique(np.asarray(robot_actor_ids, dtype=np.uint32))
    return np.isin(actor_ids.astype(np.uint32, copy=False), ids).astype(np.uint8)


def png_bytes(image: np.ndarray, compression: int = 3) -> bytes:
    ok, encoded = cv2.imencode(
        ".png",
        np.asarray(image),
        [cv2.IMWRITE_PNG_COMPRESSION, int(compression)],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode PNG.")
    return encoded.tobytes()


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_mapping_manifest(path: str | Path) -> tuple[dict[str, Any], list[EpisodeMapping]]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("version", -1)) != MAPPING_VERSION:
        raise ValueError(f"Unsupported mapping manifest version in {path}.")
    episodes = [EpisodeMapping(**item) for item in manifest["episodes"]]
    return manifest, episodes


def write_mapping_manifest(
    path: str | Path,
    episodes: Iterable[EpisodeMapping],
    *,
    base_lerobot_root: str | Path,
    source_root: str | Path,
    hf_repo: str,
    hf_revision: str,
) -> None:
    records = sorted(episodes, key=lambda item: item.global_episode)
    globals_ = [item.global_episode for item in records]
    if len(globals_) != len(set(globals_)):
        raise ValueError("Mapping contains duplicate global episode indices.")
    payload = {
        "version": MAPPING_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hf_repo": hf_repo,
        "hf_revision": hf_revision,
        "base_lerobot_root": str(Path(base_lerobot_root).resolve()),
        "source_root": str(Path(source_root).resolve()),
        "camera_mapping": CAMERA_SPECS,
        "drop_leading_frames": 1,
        "episodes": [asdict(item) for item in records],
    }
    atomic_write_json(path, payload)


class RGBStats:
    def __init__(self, max_frames: int = 100) -> None:
        self.max_frames = int(max_frames)
        self.frames = 0
        self.pixels = 0
        self.minimum = np.full(3, np.inf, dtype=np.float64)
        self.maximum = np.full(3, -np.inf, dtype=np.float64)
        self.total = np.zeros(3, dtype=np.float64)
        self.total_sq = np.zeros(3, dtype=np.float64)

    def update(self, rgb: np.ndarray) -> None:
        if self.frames >= self.max_frames:
            return
        value = np.asarray(rgb, dtype=np.float64) / 255.0
        flat = value.reshape(-1, 3)
        self.minimum = np.minimum(self.minimum, flat.min(axis=0))
        self.maximum = np.maximum(self.maximum, flat.max(axis=0))
        self.total += flat.sum(axis=0)
        self.total_sq += np.square(flat).sum(axis=0)
        self.pixels += flat.shape[0]
        self.frames += 1

    def as_lerobot_stats(self) -> dict[str, Any]:
        if self.frames == 0 or self.pixels == 0:
            raise RuntimeError("Cannot finalize empty RGB statistics.")
        mean = self.total / self.pixels
        variance = np.maximum(self.total_sq / self.pixels - np.square(mean), 0)
        wrap = lambda value: [[[float(item)]] for item in value]
        return {
            "min": wrap(self.minimum),
            "max": wrap(self.maximum),
            "mean": wrap(mean),
            "std": wrap(np.sqrt(variance)),
            "count": [self.frames],
        }


class AV1Writer:
    def __init__(
        self,
        path: str | Path,
        *,
        width: int = IMAGE_WIDTH,
        height: int = IMAGE_HEIGHT,
        fps: int = FPS,
        codec: str = "libsvtav1",
        crf: int = 30,
        gop: int = 2,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(self.path), mode="w", format="mp4")
        self.stream = self.container.add_stream(
            codec,
            rate=fps,
            options={"crf": str(crf), "g": str(gop)},
        )
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.frames = 0
        self.closed = False

    def write(self, rgb: np.ndarray) -> None:
        rgb = np.asarray(rgb)
        expected = (self.stream.height, self.stream.width, 3)
        if rgb.shape != expected or rgb.dtype != np.uint8:
            raise ValueError(f"RGB must be uint8 {expected}, got {rgb.dtype} {rgb.shape}.")
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        if self.closed:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.closed = True


class EpisodeWriter:
    """Atomically write three RGB videos and one per-episode modality ZIP."""

    def __init__(
        self,
        output_root: str | Path,
        episode_index: int,
        *,
        fps: int = FPS,
        codec: str = "libsvtav1",
        crf: int = 30,
        gop: int = 2,
        overwrite: bool = False,
    ) -> None:
        self.root = Path(output_root)
        self.episode_index = int(episode_index)
        self.final_videos = {
            key: video_path(self.root, self.episode_index, key) for key in CAMERA_SPECS
        }
        self.final_sidecar = sidecar_path(self.root, self.episode_index)
        self.final_marker = marker_path(self.root, self.episode_index)
        finals = [*self.final_videos.values(), self.final_sidecar, self.final_marker]
        if not overwrite and any(path.exists() for path in finals):
            raise FileExistsError(f"Episode {episode_index} already has output files.")
        if overwrite:
            for path in finals:
                if path.exists():
                    path.unlink()

        self.temp_videos = {
            key: path.with_name(path.name + ".partial") for key, path in self.final_videos.items()
        }
        self.temp_sidecar = self.final_sidecar.with_name(self.final_sidecar.name + ".partial")
        for path in [*self.temp_videos.values(), self.temp_sidecar]:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()

        self.video_writers = {
            key: AV1Writer(path, fps=fps, codec=codec, crf=crf, gop=gop)
            for key, path in self.temp_videos.items()
        }
        self.archive = zipfile.ZipFile(
            self.temp_sidecar, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
        )
        self.calibration: dict[str, dict[str, list[np.ndarray]]] = {
            key: {"intrinsic_cv": [], "extrinsic_cv": [], "cam2world_gl": []}
            for key in CAMERA_SPECS
        }
        self.base_pose_world: list[np.ndarray] = []
        self.state_action: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.generated_trajectory: Mapping[str, Any] | None = None
        self.rgb_stats = {key: RGBStats() for key in CAMERA_SPECS}
        self.frames = 0
        self.closed = False

    def write_frame(self, observations: Mapping[str, Mapping[str, np.ndarray]]) -> None:
        if set(observations) != set(CAMERA_SPECS):
            raise ValueError(f"Expected cameras {list(CAMERA_SPECS)}, got {list(observations)}.")
        frame_name = f"frame_{self.frames:06d}.png"
        for camera_key in CAMERA_SPECS:
            data = observations[camera_key]
            rgb = np.asarray(data["rgb"])
            depth = np.asarray(data["depth"])
            mask = np.asarray(data["robot_mask"])
            task_mask = np.asarray(data.get("task_object_mask", np.zeros_like(mask)))
            if depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or depth.dtype != np.uint16:
                raise ValueError(f"Invalid depth for {camera_key}: {depth.dtype} {depth.shape}.")
            if mask.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or mask.dtype != np.uint8:
                raise ValueError(f"Invalid mask for {camera_key}: {mask.dtype} {mask.shape}.")
            if not np.isin(mask, [0, 1]).all():
                raise ValueError(f"Mask for {camera_key} contains values outside 0/1.")
            if task_mask.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or task_mask.dtype != np.uint8:
                raise ValueError(f"Invalid task object mask for {camera_key}: {task_mask.dtype} {task_mask.shape}.")
            if not np.isin(task_mask, [0, 1]).all():
                raise ValueError(f"Task object mask for {camera_key} contains values outside 0/1.")

            self.video_writers[camera_key].write(rgb)
            self.rgb_stats[camera_key].update(rgb)
            self.archive.writestr(
                f"observation.depth.{camera_key}/{frame_name}", png_bytes(depth)
            )
            self.archive.writestr(
                f"observation.robot_mask.{camera_key}/{frame_name}", png_bytes(mask)
            )
            self.archive.writestr(
                f"observation.task_object_mask.{camera_key}/{frame_name}", png_bytes(task_mask)
            )
            for calibration_key in self.calibration[camera_key]:
                value = np.asarray(data[calibration_key], dtype=np.float32)
                allowed_shape = (3, 3) if calibration_key == "intrinsic_cv" else ((3, 4) if calibration_key == "extrinsic_cv" else (4, 4))
                if value.shape != allowed_shape or not np.isfinite(value).all():
                    raise ValueError(
                        f"Invalid {calibration_key} for {camera_key}: {value.shape}"
                    )
                self.calibration[camera_key][calibration_key].append(value)
            if "base_pose_world" not in data:
                raise ValueError("Each camera observation must include base_pose_world.")
            base_pose = np.asarray(data["base_pose_world"], dtype=np.float32)
            if base_pose.shape != (4, 4) or not np.isfinite(base_pose).all():
                raise ValueError(f"Invalid base_pose_world: {base_pose.shape}")
            if camera_key == next(iter(CAMERA_SPECS)):
                self.base_pose_world.append(base_pose)
            elif not np.allclose(base_pose, self.base_pose_world[-1], atol=1e-6):
                raise ValueError("base_pose_world differs between synchronized cameras.")
        self.frames += 1

    def set_state_action(
        self,
        state: np.ndarray,
        action: np.ndarray,
        ee_state: np.ndarray | None = None,
        ee_action: np.ndarray | None = None,
    ) -> None:
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if state.ndim != 2 or action.ndim != 2 or state.shape != action.shape:
            raise ValueError(f"state/action must have equal rank-2 shapes, got {state.shape}/{action.shape}")
        if state.shape[0] != self.frames or state.shape[1] != 14:
            raise ValueError(f"state/action shape {state.shape} does not match frames={self.frames}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("state/action contains non-finite values")
        if ee_state is None:
            ee_state = np.zeros((self.frames, 14), dtype=np.float32)
        if ee_action is None:
            ee_action = np.zeros((self.frames, 14), dtype=np.float32)
        ee_state = np.asarray(ee_state, dtype=np.float32)
        ee_action = np.asarray(ee_action, dtype=np.float32)
        if ee_state.shape != (self.frames, 14) or ee_action.shape != (self.frames, 14):
            raise ValueError(f"EE state/action must have shape ({self.frames}, 14)")
        if not np.isfinite(ee_state).all() or not np.isfinite(ee_action).all():
            raise ValueError("EE state/action contains non-finite values")
        self.state_action = (
            np.ascontiguousarray(state), np.ascontiguousarray(action),
            np.ascontiguousarray(ee_state), np.ascontiguousarray(ee_action),
        )

    def set_generated_trajectory(self, trajectory: Mapping[str, Any]) -> None:
        self.generated_trajectory = dict(trajectory)

    def write_final_rgb_diagnostics(self, images: Mapping[str, np.ndarray]) -> None:
        """Store final RGB frames for episodes that need replay review."""
        if self.closed:
            raise RuntimeError("Cannot add diagnostics after the episode payload is sealed.")
        if set(images) != set(CAMERA_SPECS):
            raise ValueError(f"Expected diagnostic cameras {list(CAMERA_SPECS)}.")
        for camera_key, image in images.items():
            rgb = np.asarray(image)
            expected = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
            if rgb.dtype != np.uint8 or rgb.shape != expected:
                raise ValueError(
                    f"Diagnostic RGB must be uint8 {expected}, got {rgb.dtype} {rgb.shape}."
                )
            self.archive.writestr(f"diagnostics/final_rgb.{camera_key}.png", png_bytes(rgb))

    def seal(self) -> None:
        """Close encoded payloads so they can be inspected before atomic commit."""
        self._close_payloads()

    def _close_payloads(self) -> None:
        if self.closed:
            return
        for writer in self.video_writers.values():
            writer.close()
        calibration_buffer = io.BytesIO()
        arrays = {
            f"{camera_key}.{matrix_key}": np.stack(values)
            for camera_key, matrices in self.calibration.items()
            for matrix_key, values in matrices.items()
        }
        if len(self.base_pose_world) != self.frames:
            raise ValueError(
                f"base pose count {len(self.base_pose_world)} != frames {self.frames}"
            )
        arrays["base_pose_world"] = np.stack(self.base_pose_world).astype(np.float32)
        if self.state_action is None:
            raise ValueError("state/action must be set before sealing an episode")
        state_action_buffer = io.BytesIO()
        np.savez_compressed(
            state_action_buffer,
            state=self.state_action[0],
            action=self.state_action[1],
            ee_state=self.state_action[2],
            ee_action=self.state_action[3],
        )
        self.archive.writestr("state_action.npz", state_action_buffer.getvalue())
        if self.generated_trajectory is not None:
            import pickle
            self.archive.writestr(
                "generated_trajectory.pkl",
                pickle.dumps(self.generated_trajectory, protocol=pickle.HIGHEST_PROTOCOL),
            )
        np.savez_compressed(calibration_buffer, **arrays)
        self.archive.writestr("calibration.npz", calibration_buffer.getvalue())
        self.archive.close()
        self.closed = True

    def commit(self, provenance: Mapping[str, Any]) -> dict[str, Any]:
        self._close_payloads()
        for key, temporary in self.temp_videos.items():
            os.replace(temporary, self.final_videos[key])
        os.replace(self.temp_sidecar, self.final_sidecar)
        files = {
            f"rgb.{key}": {
                "path": str(path.relative_to(self.root)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for key, path in self.final_videos.items()
        }
        files["modalities"] = {
            "path": str(self.final_sidecar.relative_to(self.root)),
            "size": self.final_sidecar.stat().st_size,
            "sha256": sha256_file(self.final_sidecar),
        }
        marker = {
            "version": MODALITIES_VERSION,
            "episode_index": self.episode_index,
            "frames": self.frames,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": files,
            "rgb_stats": {
                f"observation.images.{key}": stats.as_lerobot_stats()
                for key, stats in self.rgb_stats.items()
            },
            **dict(provenance),
        }
        atomic_write_json(self.final_marker, marker)
        return marker

    def abort(self) -> None:
        try:
            for writer in self.video_writers.values():
                writer.close()
        except Exception:
            pass
        try:
            self.archive.close()
        except Exception:
            pass
        for path in [*self.temp_videos.values(), self.temp_sidecar]:
            if path.exists():
                path.unlink()


def marker_is_complete(
    output_root: str | Path,
    mapping: EpisodeMapping,
    *,
    check: str = "size",
) -> bool:
    path = marker_path(output_root, mapping.global_episode)
    if not path.is_file():
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
        if int(marker.get("version", -1)) != MODALITIES_VERSION:
            return False
        if marker["source_id"] != mapping.source_id:
            return False
        frames = int(marker["frames"])
        if frames < 1 or (not marker.get("replacement", False) and frames != mapping.expected_frames):
            return False
        for metadata in marker["files"].values():
            target = Path(output_root) / metadata["path"]
            if not target.is_file() or target.stat().st_size != int(metadata["size"]):
                return False
            if check == "sha256" and sha256_file(target) != metadata["sha256"]:
                return False
        return True
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def cleanup_episode_outputs(output_root: str | Path, episode_index: int) -> None:
    paths = [
        *(video_path(output_root, episode_index, key) for key in CAMERA_SPECS),
        sidecar_path(output_root, episode_index),
        marker_path(output_root, episode_index),
    ]
    for path in paths:
        partial = path.with_name(path.name + ".partial")
        if partial.exists():
            partial.unlink()
        if path.exists():
            path.unlink()


def build_modalities_manifest() -> dict[str, Any]:
    entries = {}
    for camera_key in CAMERA_SPECS:
        entries[f"observation.depth.{camera_key}"] = {
            "dtype": "uint16",
            "shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
            "unit": "millimeter",
            "invalid_value": 0,
            "encoding": "png16",
            "member_path": f"observation.depth.{camera_key}/frame_{{frame_index:06d}}.png",
        }
        entries[f"observation.robot_mask.{camera_key}"] = {
            "dtype": "uint8",
            "shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
            "values": {"0": "background", "1": "robot"},
            "encoding": "png",
            "member_path": f"observation.robot_mask.{camera_key}/frame_{{frame_index:06d}}.png",
        }
        entries[f"observation.task_object_mask.{camera_key}"] = {
            "dtype": "uint8",
            "shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
            "values": {"0": "background", "1": "task_object_or_container"},
            "encoding": "png",
            "member_path": f"observation.task_object_mask.{camera_key}/frame_{{frame_index:06d}}.png",
        }
    return {
        "version": MODALITIES_VERSION,
        "sidecar_path": "modalities/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.zip",
        "camera_mapping": CAMERA_SPECS,
        "calibration_member": "calibration.npz",
        "calibration_arrays": {
            "camera_intrinsic": "{camera}.intrinsic_cv [T,3,3]",
            "camera_extrinsic": "{camera}.extrinsic_cv [T,3,4]",
            "camera_cam2world": "{camera}.cam2world_gl [T,4,4]",
            "robot_base_pose_world": "base_pose_world [T,4,4], frame=aloha-agilex/footprint",
        },
        "state_action_member": "state_action.npz",
        "physics_warning_diagnostics": {
            "marker_field": "replay_validation",
            "final_rgb_member": "diagnostics/final_rgb.{camera_key}.png",
            "status": "physics_warning",
            "hard_consistency_required": True,
        },
        "features": entries,
    }


def inspect_sidecar(
    path: str | Path,
    expected_frames: int,
    *,
    deep: bool = True,
    verify_crc: bool = True,
) -> None:
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        if verify_crc:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"CRC failure in {path}: {bad}")
        names = set(archive.namelist())
        expected = {"calibration.npz", "state_action.npz"}
        for camera_key in CAMERA_SPECS:
            for frame_index in range(expected_frames):
                frame = f"frame_{frame_index:06d}.png"
                expected.add(f"observation.depth.{camera_key}/{frame}")
                expected.add(f"observation.robot_mask.{camera_key}/{frame}")
                # Task-object masks are optional for legacy sidecars.
        allowed_diagnostics = {
            f"diagnostics/final_rgb.{camera_key}.png" for camera_key in CAMERA_SPECS
        }
        allowed_optional = {"generated_trajectory.pkl"}
        unexpected = names - expected - allowed_diagnostics - allowed_optional
        task_mask_names = {
            f"observation.task_object_mask.{camera_key}/frame_{frame_index:06d}.png"
            for camera_key in CAMERA_SPECS
            for frame_index in range(expected_frames)
        }
        present_task_masks = names & task_mask_names
        if present_task_masks and present_task_masks != task_mask_names:
            raise ValueError("Sidecar contains an incomplete task-object mask set.")
        unexpected = names - expected - task_mask_names - allowed_diagnostics - allowed_optional
        if not expected.issubset(names) or unexpected:
            raise ValueError(
                f"Sidecar members mismatch: missing={len(expected - names)} "
                f"unexpected={len(unexpected)}"
            )
        diagnostic_names = names & allowed_diagnostics
        if diagnostic_names and diagnostic_names != allowed_diagnostics:
            raise ValueError("Sidecar contains an incomplete final RGB diagnostic set.")
        with np.load(io.BytesIO(archive.read("calibration.npz"))) as calibration:
            for camera_key in CAMERA_SPECS:
                shapes = {
                    "intrinsic_cv": (expected_frames, 3, 3),
                    "extrinsic_cv": (expected_frames, 3, 4),
                    "cam2world_gl": (expected_frames, 4, 4),
                }
                for key, shape in shapes.items():
                    if calibration[f"{camera_key}.{key}"].shape != shape:
                        raise ValueError(f"Invalid calibration shape for {camera_key}.{key}.")
            if calibration["base_pose_world"].shape != (expected_frames, 4, 4):
                raise ValueError("Invalid calibration shape for base_pose_world.")
        with np.load(io.BytesIO(archive.read("state_action.npz"))) as state_action:
            for key in ("state", "action"):
                value = state_action[key]
                if value.shape != (expected_frames, 14) or value.dtype != np.float32:
                    raise ValueError(f"Invalid state/action shape or dtype for {key}.")
            ee_keys = {"ee_state", "ee_action"}.intersection(state_action.files)
            if ee_keys and ee_keys != {"ee_state", "ee_action"}:
                raise ValueError("Sidecar contains an incomplete EE state/action pair.")
            for key in ee_keys:
                value = state_action[key]
                if value.shape != (expected_frames, 14) or value.dtype != np.float32:
                    raise ValueError(f"Invalid EE state/action shape or dtype for {key}.")
        if deep and expected_frames:
            indices = sorted({0, expected_frames // 2, expected_frames - 1})
            for camera_key in CAMERA_SPECS:
                for frame_index in indices:
                    frame = f"frame_{frame_index:06d}.png"
                    depth = cv2.imdecode(
                        np.frombuffer(
                            archive.read(f"observation.depth.{camera_key}/{frame}"), dtype=np.uint8
                        ),
                        cv2.IMREAD_UNCHANGED,
                    )
                    mask = cv2.imdecode(
                        np.frombuffer(
                            archive.read(f"observation.robot_mask.{camera_key}/{frame}"),
                            dtype=np.uint8,
                        ),
                        cv2.IMREAD_UNCHANGED,
                    )
                    if depth.dtype != np.uint16 or depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                        raise ValueError(f"Invalid decoded depth in {path}.")
                    if mask.dtype != np.uint8 or mask.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                        raise ValueError(f"Invalid decoded mask in {path}.")
                    if not np.isin(mask, [0, 1]).all():
                        raise ValueError(f"Non-binary mask in {path}.")
                    task_name = f"observation.task_object_mask.{camera_key}/{frame}"
                    if task_name in names:
                        task_mask = cv2.imdecode(
                            np.frombuffer(archive.read(task_name), dtype=np.uint8),
                            cv2.IMREAD_UNCHANGED,
                        )
                        if task_mask is None or task_mask.dtype != np.uint8 or task_mask.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                            raise ValueError(f"Invalid task-object mask in {path}.")
                        if not np.isin(task_mask, [0, 1]).all():
                            raise ValueError(f"Non-binary task-object mask in {path}.")


def inspect_video(path: str | Path, expected_frames: int, *, decode: bool = True) -> None:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.width != IMAGE_WIDTH or stream.height != IMAGE_HEIGHT:
            raise ValueError(f"Unexpected video dimensions in {path}.")
        if decode:
            frames = sum(1 for _ in container.decode(stream))
            if frames != expected_frames:
                raise ValueError(f"Video {path} has {frames} frames, expected {expected_frames}.")
        elif stream.frames not in (0, expected_frames):
            raise ValueError(f"Video {path} reports {stream.frames} frames, expected {expected_frames}.")


def copy_or_link_file(source: Path, destination: Path, mode: str = "auto") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode not in {"auto", "copy", "reflink"}:
        raise ValueError(f"Unsupported link mode {mode!r}.")
    if mode != "copy":
        # Linux FICLONE creates an independent copy-on-write file.  Unlike a
        # hardlink, deleting or replacing the base dataset cannot invalidate
        # the finalized dataset.  Network filesystems commonly reject this;
        # auto mode then falls back to a normal copy.
        ficlone = 0x40049409
        try:
            with source.open("rb") as input_file, destination.open("xb") as output_file:
                fcntl.ioctl(output_file.fileno(), ficlone, input_file.fileno())
            return
        except OSError:
            if destination.exists():
                destination.unlink()
            if mode == "reflink":
                raise
    shutil.copy2(source, destination)


def temporary_directory(parent: str | Path | None = None) -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix="robotwin-rgbdm-", dir=parent)
