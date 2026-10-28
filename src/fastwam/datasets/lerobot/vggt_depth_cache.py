from __future__ import annotations

import bisect
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .vggt_teacher_cache import (
    DEFAULT_SHARD_SIZE,
    assigned_shard_indices,
    atomic_write_json,
    descriptor_fingerprint,
    num_shards,
    sha256_file,
    shard_frame_count,
)


CACHE_FORMAT = "fastwam_vggt_depth_targets"
CACHE_VERSION = 1
TRUE_DEPTH_CACHE_VERSION = 2
SUPPORTED_CACHE_VERSIONS = (CACHE_VERSION, TRUE_DEPTH_CACHE_VERSION)
NORMALIZATION_VERSION = 1
TARGET_DTYPE = "bfloat16"
WEIGHT_DTYPE = "bfloat16"
VALIDITY_DTYPE = "bitpack_little"
STATS_DTYPE = "float32"


def build_depth_cache_descriptor(
    lerobot_dataset,
    *,
    camera_keys: Sequence[str],
    image_size: int,
    teacher_checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Describe every input that changes a per-timestep VGGT depth target."""
    logical_to_lerobot = {
        str(meta["key"]): str(meta["lerobot_key"])
        for meta in lerobot_dataset.image_meta
    }
    requested = tuple(str(key) for key in camera_keys)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("`camera_keys` must be non-empty and contain no duplicates.")
    missing = sorted(set(requested) - set(logical_to_lerobot))
    if missing:
        raise ValueError(f"VGGT depth cache cameras are absent from shape_meta: {missing}")

    sources = []
    for dataset in lerobot_dataset.multi_dataset._datasets:
        episodes = (
            list(range(int(dataset.meta.total_episodes)))
            if dataset.episodes is None
            else [int(value) for value in dataset.episodes]
        )
        sources.append(
            {
                "dataset_path": str(Path(dataset.root).expanduser().resolve()),
                "episodes": episodes,
                "num_frames": int(dataset.num_frames),
            }
        )

    checkpoint = Path(teacher_checkpoint_path).expanduser().resolve()
    return {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "sources": sources,
        "total_frames": int(len(lerobot_dataset)),
        "camera_keys": list(requested),
        "lerobot_camera_keys": [logical_to_lerobot[key] for key in requested],
        "preprocessing": {
            "mode": "balanced",
            "image_resolution": int(image_size),
            "patch_size": 16,
        },
        "teacher": {
            "checkpoint_sha256": sha256_file(str(checkpoint)),
            "head": "dense_head",
        },
        "normalization": {
            "version": NORMALIZATION_VERSION,
            "space": "log_depth",
            "center": "median_across_views",
            "scale": "1.4826_mad_across_views",
            "target_clip": [-5.0, 5.0],
            "confidence": "valid_mean_normalized_clip_0_5",
        },
        "dtype": {
            "target": TARGET_DTYPE,
            "weight": WEIGHT_DTYPE,
            "stats": STATS_DTYPE,
        },
    }


def build_true_depth_cache_descriptor(
    lerobot_dataset,
    *,
    dataset_name: str | None = None,
    camera_keys: Sequence[str],
    target_height: int,
    target_width: int,
) -> dict[str, Any]:
    """Describe RoboTwin simulator depth targets without changing v1 fingerprints."""
    logical_to_lerobot = {
        str(meta["key"]): str(meta["lerobot_key"])
        for meta in lerobot_dataset.image_meta
    }
    requested = tuple(str(key) for key in camera_keys)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("`camera_keys` must be non-empty and contain no duplicates.")
    missing = sorted(set(requested) - set(logical_to_lerobot))
    if missing:
        raise ValueError(f"True depth cache cameras are absent from shape_meta: {missing}")
    if target_height <= 0 or target_width <= 0:
        raise ValueError("True depth target height and width must be positive.")

    sources = []
    modality_contract = None
    depth_sources = set()
    for dataset in lerobot_dataset.multi_dataset._datasets:
        root = Path(dataset.root).expanduser().resolve()
        manifest_path = root / "meta" / "modalities.json"
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            features = manifest.get("features")
            if not isinstance(features, dict):
                raise ValueError(f"Invalid modalities feature map in {manifest_path}.")
            source = "robotwin_sidecar"
        else:
            # clean_xera stores depth as regular LeRobot video features.
            features = getattr(dataset.meta, "features", {})
            if not isinstance(features, dict):
                raise ValueError(f"Invalid LeRobot feature map in {root / 'meta' / 'info.json'}.")
            source = "lerobot_depth_video"
        depth_sources.add(source)
        camera_features = {}
        for camera_key in requested:
            feature_key = (
                f"observation.depth.{camera_key}"
                if source == "robotwin_sidecar"
                else f"observation.images.{camera_key}_depth"
            )
            feature = features.get(feature_key)
            if not isinstance(feature, dict):
                raise ValueError(f"Missing `{feature_key}` in {manifest_path}.")
            if source == "robotwin_sidecar":
                valid_contract = (
                    feature.get("dtype") == "uint16"
                    and feature.get("unit") == "millimeter"
                    and int(feature.get("invalid_value", -1)) == 0
                )
            else:
                info = feature.get("info", {})
                valid_contract = (
                    feature.get("dtype") == "video"
                    and bool(info.get("is_depth_map", False))
                    and str(info.get("depth_unit", "mm")).lower() in {"mm", "millimeter"}
                )
            if not valid_contract:
                raise ValueError(f"Unsupported true depth contract for `{feature_key}` in {root}.")
            shape = [int(value) for value in feature.get("shape", [])]
            if source == "lerobot_depth_video" and len(shape) == 3 and shape[-1] == 1:
                shape = shape[:2]
            if len(shape) != 2 or any(value <= 0 for value in shape):
                raise ValueError(f"Invalid true depth shape for `{feature_key}`: {shape}.")
            camera_features[camera_key] = {
                "shape": shape,
                "member_path": str(feature.get("member_path", feature_key)),
            }
            if source == "lerobot_depth_video":
                camera_features[camera_key]["lerobot_key"] = feature_key
        if source == "robotwin_sidecar":
            current_contract = {
                "version": int(manifest.get("version", -1)),
                "sidecar_path": str(manifest["sidecar_path"]),
                "features": camera_features,
            }
        else:
            current_contract = {"source": source, "features": camera_features}
        if modality_contract is None:
            modality_contract = current_contract
        elif current_contract != modality_contract:
            raise ValueError("All true depth datasets must share one modalities contract.")

        episodes = (
            list(range(int(dataset.meta.total_episodes)))
            if dataset.episodes is None
            else [int(value) for value in dataset.episodes]
        )
        sources.append(
            {
                "dataset_path": str(root),
                "episodes": episodes,
                "num_frames": int(dataset.num_frames),
                "modalities_manifest_sha256": (
                    sha256_file(str(manifest_path)) if manifest_path.is_file() else None
                ),
            }
        )

    descriptor = {
        "format": CACHE_FORMAT,
        "version": TRUE_DEPTH_CACHE_VERSION,
        "target_source": "robotwin_true_depth",
        "sources": sources,
        "total_frames": int(len(lerobot_dataset)),
        "camera_keys": list(requested),
        "lerobot_camera_keys": [logical_to_lerobot[key] for key in requested],
        "modalities": modality_contract,
        "preprocessing": {
            "source_unit": "millimeter",
            "target_unit": "meter",
            "invalid_value": 0,
            "resize": "nearest-exact",
            "target_shape": [int(target_height), int(target_width)],
            "crop": "none_for_supported_4_by_3_aspect_ratio",
        },
        "normalization": {
            "version": NORMALIZATION_VERSION,
            "space": "log_depth",
            "center": "median_across_views",
            "scale": "1.4826_mad_across_views",
            "target_clip": [-5.0, 5.0],
            "confidence": "binary_validity",
        },
        "dtype": {
            "target": TARGET_DTYPE,
            "validity": VALIDITY_DTYPE,
            "stats": STATS_DTYPE,
        },
    }
    if dataset_name:
        descriptor["dataset_name"] = str(dataset_name)
    if len(depth_sources) != 1 or next(iter(depth_sources)) != "robotwin_sidecar":
        descriptor["depth_source"] = (
            next(iter(depth_sources)) if len(depth_sources) == 1 else "mixed"
        )
    return descriptor


def resolve_depth_cache_dir(
    cache_root: str | Path, descriptor: Mapping[str, Any]
) -> Path:
    return Path(cache_root).expanduser().resolve() / descriptor_fingerprint(descriptor)


def normalize_vggt_depth(
    depth: torch.Tensor,
    confidence: torch.Tensor,
    *,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize each timestep jointly across views.

    Args:
        depth: ``[B,V,H,W]`` or ``[B,V,H,W,1]`` positive depth.
        confidence: ``[B,V,H,W]`` VGGT confidence.

    Returns:
        normalized target and confidence weights as ``[B,V,H,W]`` fp32, plus
        per-timestep ``[median, robust_scale]`` stats as ``[B,2]`` fp32.
    """
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if depth.ndim != 4 or confidence.ndim != 4 or depth.shape != confidence.shape:
        raise ValueError(
            "VGGT depth/confidence must match as [B,V,H,W], got "
            f"{tuple(depth.shape)} and {tuple(confidence.shape)}."
        )
    depth = depth.float()
    confidence = confidence.float()
    target = torch.zeros_like(depth)
    weight = torch.zeros_like(confidence)
    stats = torch.empty((depth.shape[0], 2), dtype=torch.float32, device=depth.device)

    for batch_index in range(depth.shape[0]):
        cur_depth = depth[batch_index]
        cur_conf = confidence[batch_index]
        valid = torch.isfinite(cur_depth) & (cur_depth > 0)
        valid_conf = valid & torch.isfinite(cur_conf) & (cur_conf > 0)
        if not bool(valid.any().item()):
            raise ValueError(f"VGGT depth timestep {batch_index} contains no valid pixels.")

        log_depth = torch.log(cur_depth.clamp_min(eps))
        valid_log = log_depth[valid]
        center = valid_log.median()
        mad = (valid_log - center).abs().median()
        scale = (1.4826 * mad).clamp_min(eps)
        normalized = ((log_depth - center) / scale).clamp(-5.0, 5.0)
        target[batch_index] = torch.where(valid, normalized, torch.zeros_like(normalized))

        if bool(valid_conf.any().item()):
            mean_conf = cur_conf[valid_conf].mean().clamp_min(eps)
            normalized_conf = (cur_conf / mean_conf).clamp(0.0, 5.0)
            weight[batch_index] = torch.where(
                valid_conf, normalized_conf, torch.zeros_like(normalized_conf)
            )
        stats[batch_index, 0] = center
        stats[batch_index, 1] = scale
    return target, weight, stats


def depth_shard_filename(shard_index: int, field: str) -> str:
    suffix = {
        "target": "depth.bf16",
        "weight": "weight.bf16",
        "validity": "validity.bitpack",
        "stats": "stats.f32",
    }
    if field not in suffix:
        raise ValueError(f"Unknown depth cache shard field: {field!r}.")
    return f"{int(shard_index):06d}.{suffix[field]}"


def depth_shard_done_filename(shard_index: int) -> str:
    return f"{int(shard_index):06d}.done.json"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_field_bytes(
    frame_count: int, depth_shape: Sequence[int], field: str
) -> int:
    if field == "stats":
        elements = int(frame_count) * 2
        dtype = torch.float32
    elif field == "validity":
        frame_elements = 1
        for value in depth_shape:
            frame_elements *= int(value)
        return int(frame_count) * ((frame_elements + 7) // 8)
    else:
        elements = int(frame_count)
        for value in depth_shape:
            elements *= int(value)
        dtype = torch.bfloat16
    return elements * torch.tensor([], dtype=dtype).element_size()


def _atomic_write_tensor(tensor: torch.Tensor, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.partial.{uuid.uuid4().hex}"
    if tensor.dtype == torch.bfloat16:
        tensor.contiguous().view(torch.uint16).numpy().tofile(temporary)
    elif tensor.dtype == torch.float32:
        tensor.contiguous().numpy().tofile(temporary)
    elif tensor.dtype == torch.uint8:
        tensor.contiguous().numpy().tofile(temporary)
    else:
        raise ValueError(f"Unsupported depth cache dtype: {tensor.dtype}.")
    digest = _sha256_path(temporary)
    os.replace(temporary, output)
    return {
        "file": output.name,
        "bytes": int(output.stat().st_size),
        "sha256": digest,
    }


def write_depth_cache_shard(
    target: torch.Tensor,
    weight: torch.Tensor,
    stats: torch.Tensor,
    *,
    shards_dir: str | Path,
    done_path: str | Path,
    shard_index: int,
) -> dict[str, Any]:
    if target.device.type != "cpu" or weight.device.type != "cpu" or stats.device.type != "cpu":
        raise ValueError("Depth cache shards must be CPU tensors.")
    if target.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("Depth target and weight cache tensors must be bfloat16.")
    if stats.dtype != torch.float32:
        raise ValueError("Depth cache stats must be float32.")
    if target.ndim != 4 or target.shape != weight.shape or target.shape[0] <= 0:
        raise ValueError(
            "Depth target/weight shards must match as [frames,views,H,W], got "
            f"{tuple(target.shape)} and {tuple(weight.shape)}."
        )
    if tuple(stats.shape) != (target.shape[0], 2):
        raise ValueError(
            f"Depth cache stats must be [{target.shape[0]},2], got {tuple(stats.shape)}."
        )
    directory = Path(shards_dir)
    fields = {
        "target": _atomic_write_tensor(
            target, directory / depth_shard_filename(shard_index, "target")
        ),
        "weight": _atomic_write_tensor(
            weight, directory / depth_shard_filename(shard_index, "weight")
        ),
        "stats": _atomic_write_tensor(
            stats, directory / depth_shard_filename(shard_index, "stats")
        ),
    }
    record = {
        "index": int(shard_index),
        "frames": int(target.shape[0]),
        "fields": fields,
    }
    atomic_write_json(record, done_path)
    return record


def pack_depth_validity(validity: torch.Tensor) -> torch.Tensor:
    """Pack one independent little-endian bit row per frame."""
    if validity.device.type != "cpu" or validity.ndim != 4:
        raise ValueError("Depth validity must be a CPU [frames,views,H,W] tensor.")
    rows = validity.to(dtype=torch.uint8).contiguous().view(validity.shape[0], -1)
    packed = np.packbits(rows.numpy(), axis=1, bitorder="little")
    return torch.from_numpy(packed.copy())


def unpack_depth_validity(
    packed: torch.Tensor, depth_shape: Sequence[int]
) -> torch.Tensor:
    """Expand packed little-endian validity rows to BF16 0/1 weights."""
    shape = tuple(int(value) for value in depth_shape)
    frame_elements = int(np.prod(shape, dtype=np.int64))
    expected_bytes = (frame_elements + 7) // 8
    if packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[1] != expected_bytes:
        raise ValueError(
            f"Packed depth validity must be [frames,{expected_bytes}] uint8, "
            f"got {tuple(packed.shape)} {packed.dtype}."
        )
    shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = torch.bitwise_and(
        torch.bitwise_right_shift(packed.unsqueeze(-1), shifts), 1
    )
    return bits.reshape(packed.shape[0], -1)[:, :frame_elements].reshape(
        packed.shape[0], *shape
    ).to(torch.bfloat16)


def write_true_depth_cache_shard(
    target: torch.Tensor,
    validity: torch.Tensor,
    stats: torch.Tensor,
    *,
    shards_dir: str | Path,
    done_path: str | Path,
    shard_index: int,
) -> dict[str, Any]:
    if (
        target.device.type != "cpu"
        or validity.device.type != "cpu"
        or stats.device.type != "cpu"
    ):
        raise ValueError("True depth cache shards must be CPU tensors.")
    if target.dtype != torch.bfloat16 or validity.dtype != torch.bool:
        raise ValueError("True depth target must be bfloat16 and validity must be bool.")
    if target.ndim != 4 or target.shape != validity.shape or target.shape[0] <= 0:
        raise ValueError("True depth target/validity must match as [frames,views,H,W].")
    if stats.dtype != torch.float32 or tuple(stats.shape) != (target.shape[0], 2):
        raise ValueError(f"True depth stats must be [{target.shape[0]},2] float32.")
    directory = Path(shards_dir)
    fields = {
        "target": _atomic_write_tensor(
            target, directory / depth_shard_filename(shard_index, "target")
        ),
        "validity": _atomic_write_tensor(
            pack_depth_validity(validity),
            directory / depth_shard_filename(shard_index, "validity"),
        ),
        "stats": _atomic_write_tensor(
            stats, directory / depth_shard_filename(shard_index, "stats")
        ),
    }
    record = {
        "index": int(shard_index),
        "frames": int(target.shape[0]),
        "weight_encoding": VALIDITY_DTYPE,
        "fields": fields,
    }
    atomic_write_json(record, done_path)
    return record


def read_valid_depth_shard_record(
    *,
    shards_dir: str | Path,
    done_path: str | Path,
    shard_index: int,
    frame_count: int,
    depth_shape: Sequence[int],
    cache_version: int = CACHE_VERSION,
    verify_sha256: bool = False,
) -> dict[str, Any] | None:
    if int(cache_version) not in SUPPORTED_CACHE_VERSIONS:
        return None
    done = Path(done_path)
    if not done.is_file():
        return None
    try:
        with done.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if int(record.get("index", -1)) != int(shard_index) or int(
        record.get("frames", -1)
    ) != int(frame_count):
        return None
    expected_fields = (
        {"target", "weight", "stats"}
        if int(cache_version) == CACHE_VERSION
        else {"target", "validity", "stats"}
    )
    fields = record.get("fields")
    if not isinstance(fields, dict) or set(fields) != expected_fields:
        return None
    if int(cache_version) == TRUE_DEPTH_CACHE_VERSION and record.get(
        "weight_encoding"
    ) != VALIDITY_DTYPE:
        return None
    directory = Path(shards_dir)
    for field in fields:
        expected_name = depth_shard_filename(shard_index, field)
        path = directory / expected_name
        info = fields[field]
        expected_bytes = _expected_field_bytes(frame_count, depth_shape, field)
        if (
            not path.is_file()
            or not isinstance(info, dict)
            or info.get("file") != expected_name
            or int(info.get("bytes", -1)) != expected_bytes
            or path.stat().st_size != expected_bytes
            or not isinstance(info.get("sha256"), str)
        ):
            return None
        if verify_sha256 and _sha256_path(path) != info["sha256"]:
            return None
    return record


def build_depth_manifest(
    *,
    descriptor: Mapping[str, Any],
    depth_shape: Sequence[int],
    shard_size: int,
    shard_records: Sequence[Mapping[str, Any]],
    cache_version: int = CACHE_VERSION,
) -> dict[str, Any]:
    if int(cache_version) not in SUPPORTED_CACHE_VERSIONS:
        raise ValueError(f"Unsupported depth cache manifest version: {cache_version}.")
    shape = [int(value) for value in depth_shape]
    if len(shape) != 3 or shape[0] != len(descriptor["camera_keys"]):
        raise ValueError(f"Invalid VGGT depth cache shape: {shape}.")
    return {
        "format": CACHE_FORMAT,
        "version": int(cache_version),
        "fingerprint": descriptor_fingerprint(descriptor),
        "descriptor": dict(descriptor),
        "depth_shape": shape,
        "shard_size": int(shard_size),
        "shards": [dict(record) for record in shard_records],
    }


class VGGTDepthTargetCache:
    """Strict, per-worker lazy mmap reader for per-timestep depth targets."""

    def __init__(
        self,
        *,
        cache_root: str | Path,
        descriptor: Mapping[str, Any],
        episode_ends: Sequence[int],
    ) -> None:
        self.descriptor = dict(descriptor)
        self.fingerprint = descriptor_fingerprint(self.descriptor)
        self.cache_dir = resolve_depth_cache_dir(cache_root, self.descriptor)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            script = (
                "scripts/precompute_robotwin_true_depth_targets.py"
                if self.descriptor.get("target_source") == "robotwin_true_depth"
                else "scripts/precompute_vggt_depth_targets.py"
            )
            raise FileNotFoundError(
                f"Missing VGGT depth cache manifest: {manifest_path}. "
                f"Run {script} first."
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.cache_version = int(manifest.get("version", -1))
        if (
            manifest.get("format") != CACHE_FORMAT
            or self.cache_version not in SUPPORTED_CACHE_VERSIONS
        ):
            raise ValueError(f"Unsupported VGGT depth cache format in {manifest_path}.")
        descriptor_version = int(self.descriptor.get("version", self.cache_version))
        if descriptor_version != self.cache_version:
            raise ValueError(
                "Depth cache descriptor/manifest version mismatch: "
                f"{descriptor_version} vs {self.cache_version} in {manifest_path}."
            )
        if manifest.get("fingerprint") != self.fingerprint or manifest.get(
            "descriptor"
        ) != self.descriptor:
            raise ValueError(
                "VGGT depth cache fingerprint mismatch: expected "
                f"{self.fingerprint}, got {manifest.get('fingerprint')} in {manifest_path}."
            )

        self.total_frames = int(self.descriptor["total_frames"])
        self.depth_shape = tuple(int(value) for value in manifest["depth_shape"])
        self.shard_size = int(manifest["shard_size"])
        if len(self.depth_shape) != 3 or self.shard_size <= 0:
            raise ValueError(f"Invalid VGGT depth cache layout in {manifest_path}.")
        records = manifest.get("shards")
        expected_count = num_shards(self.total_frames, self.shard_size)
        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError(
                f"VGGT depth cache requires {expected_count} shards, found "
                f"{0 if not isinstance(records, list) else len(records)}."
            )
        self._records = []
        shards_dir = self.cache_dir / "shards"
        for shard_index, manifest_record in enumerate(records):
            frame_count = shard_frame_count(self.total_frames, self.shard_size, shard_index)
            record = read_valid_depth_shard_record(
                shards_dir=shards_dir,
                done_path=shards_dir / depth_shard_done_filename(shard_index),
                shard_index=shard_index,
                frame_count=frame_count,
                depth_shape=self.depth_shape,
                cache_version=self.cache_version,
            )
            if record is None or record != manifest_record:
                raise ValueError(
                    f"VGGT depth cache shard {shard_index} is missing or incomplete."
                )
            self._records.append(record)

        self.episode_ends = tuple(int(value) for value in episode_ends)
        if not self.episode_ends or self.episode_ends[-1] != self.total_frames:
            raise ValueError(
                "VGGT depth cache episode boundaries must end at total_frames."
            )
        if any(
            left >= right for left, right in zip(self.episode_ends, self.episode_ends[1:])
        ):
            raise ValueError("VGGT depth cache episode ends must be strictly increasing.")
        self._mapped_shards: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._mapped_pid: int | None = None

    @property
    def num_views(self) -> int:
        return self.depth_shape[0]

    @property
    def height(self) -> int:
        return self.depth_shape[1]

    @property
    def width(self) -> int:
        return self.depth_shape[2]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_mapped_shards"] = {}
        state["_mapped_pid"] = None
        return state

    def _mapped_shard(
        self, shard_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pid = os.getpid()
        if self._mapped_pid != pid:
            self._mapped_shards = {}
            self._mapped_pid = pid
        mapped = self._mapped_shards.get(shard_index)
        if mapped is None:
            frames = int(self._records[shard_index]["frames"])
            depth_elements = frames
            for value in self.depth_shape:
                depth_elements *= value
            shards_dir = self.cache_dir / "shards"
            target = torch.from_file(
                str(shards_dir / depth_shard_filename(shard_index, "target")),
                shared=False,
                size=depth_elements,
                dtype=torch.bfloat16,
            ).view(frames, *self.depth_shape)
            if self.cache_version == CACHE_VERSION:
                weight = torch.from_file(
                    str(shards_dir / depth_shard_filename(shard_index, "weight")),
                    shared=False,
                    size=depth_elements,
                    dtype=torch.bfloat16,
                ).view(frames, *self.depth_shape)
            else:
                frame_bytes = (
                    int(np.prod(self.depth_shape, dtype=np.int64)) + 7
                ) // 8
                weight = torch.from_file(
                    str(shards_dir / depth_shard_filename(shard_index, "validity")),
                    shared=False,
                    size=frames * frame_bytes,
                    dtype=torch.uint8,
                ).view(frames, frame_bytes)
            stats = torch.from_file(
                str(shards_dir / depth_shard_filename(shard_index, "stats")),
                shared=False,
                size=frames * 2,
                dtype=torch.float32,
            ).view(frames, 2)
            mapped = (target, weight, stats)
            self._mapped_shards[shard_index] = mapped
        return mapped

    def get(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(index)
        if not 0 <= index < self.total_frames:
            raise IndexError(
                f"VGGT depth cache index {index} is outside [0,{self.total_frames})."
            )
        shard_index, row_index = divmod(index, self.shard_size)
        target, weight, stats = self._mapped_shard(shard_index)
        if self.cache_version == TRUE_DEPTH_CACHE_VERSION:
            weight_row = unpack_depth_validity(
                weight[row_index : row_index + 1], self.depth_shape
            )[0]
        else:
            weight_row = weight[row_index]
        return target[row_index], weight_row, stats[row_index]

    def get_future(
        self,
        index: int,
        *,
        future_offsets: Sequence[int],
        global_sample_stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        offsets = tuple(int(value) for value in future_offsets)
        if (
            not offsets
            or any(value <= 0 for value in offsets)
            or any(left >= right for left, right in zip(offsets, offsets[1:]))
        ):
            raise ValueError("Future offsets must be strictly increasing positive integers.")
        if global_sample_stride <= 0:
            raise ValueError("`global_sample_stride` must be positive.")
        index = int(index)
        episode_index = bisect.bisect_right(self.episode_ends, index)
        if episode_index >= len(self.episode_ends):
            raise IndexError(f"VGGT depth cache index {index} has no episode boundary.")
        episode_end = self.episode_ends[episode_index]
        targets, weights, stats, padding = [], [], [], []
        for offset in offsets:
            requested = index + offset * int(global_sample_stride)
            is_pad = requested >= episode_end
            future_index = min(requested, episode_end - 1)
            target, weight, frame_stats = self.get(future_index)
            targets.append(target)
            weights.append(weight)
            stats.append(frame_stats)
            padding.append(is_pad)
        return (
            torch.stack(targets),
            torch.stack(weights),
            torch.stack(stats),
            torch.tensor(padding, dtype=torch.bool),
        )


__all__ = [
    "CACHE_FORMAT",
    "CACHE_VERSION",
    "TRUE_DEPTH_CACHE_VERSION",
    "DEFAULT_SHARD_SIZE",
    "NORMALIZATION_VERSION",
    "VGGTDepthTargetCache",
    "assigned_shard_indices",
    "build_depth_cache_descriptor",
    "build_true_depth_cache_descriptor",
    "build_depth_manifest",
    "depth_shard_done_filename",
    "depth_shard_filename",
    "normalize_vggt_depth",
    "pack_depth_validity",
    "num_shards",
    "read_valid_depth_shard_record",
    "resolve_depth_cache_dir",
    "shard_frame_count",
    "write_depth_cache_shard",
    "write_true_depth_cache_shard",
    "unpack_depth_validity",
]
