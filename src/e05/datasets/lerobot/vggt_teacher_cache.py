from __future__ import annotations

import bisect
import hashlib
import json
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


CACHE_FORMAT = "e05_vggt_teacher_features"
CACHE_VERSION = 1
CACHE_DTYPE = "bfloat16"
DEFAULT_SHARD_SIZE = 256


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@lru_cache(maxsize=16)
def sha256_file(path_value: str) -> str:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"VGGT teacher checkpoint not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor_fingerprint(descriptor: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(descriptor).encode("utf-8")).hexdigest()


def build_cache_descriptor(
    lerobot_dataset,
    *,
    dataset_name: str | None = None,
    camera_keys: Sequence[str],
    image_size: int,
    teacher_checkpoint_path: str | Path,
    feature_layer: int,
    feature_dim: int,
) -> dict[str, Any]:
    """Describe the selected frames and every input that changes teacher features."""
    logical_to_lerobot = {
        str(meta["key"]): str(meta["lerobot_key"])
        for meta in lerobot_dataset.image_meta
    }
    requested = tuple(str(key) for key in camera_keys)
    missing = sorted(set(requested) - set(logical_to_lerobot))
    if missing:
        raise ValueError(f"VGGT cache camera keys are absent from shape_meta: {missing}")

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
    descriptor = {
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
            "feature_layer": int(feature_layer),
            "feature_dim": int(feature_dim),
        },
        "dtype": CACHE_DTYPE,
    }
    if dataset_name:
        descriptor["dataset_name"] = str(dataset_name)
    return descriptor


def resolve_cache_dir(cache_root: str | Path, descriptor: Mapping[str, Any]) -> Path:
    return Path(cache_root).expanduser().resolve() / descriptor_fingerprint(descriptor)


def atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.partial.{uuid.uuid4().hex}"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)


def shard_filename(shard_index: int) -> str:
    return f"{int(shard_index):06d}.bf16"


def shard_done_filename(shard_index: int) -> str:
    return f"{int(shard_index):06d}.done.json"


def num_shards(total_frames: int, shard_size: int) -> int:
    if total_frames <= 0 or shard_size <= 0:
        raise ValueError("`total_frames` and `shard_size` must be positive.")
    return (int(total_frames) + int(shard_size) - 1) // int(shard_size)


def assigned_shard_indices(
    total_frames: int,
    shard_size: int,
    *,
    rank: int,
    world_size: int,
) -> list[int]:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid distributed shard assignment: rank={rank}, world_size={world_size}.")
    return list(range(rank, num_shards(total_frames, shard_size), world_size))


def shard_frame_count(total_frames: int, shard_size: int, shard_index: int) -> int:
    start = int(shard_index) * int(shard_size)
    return max(0, min(int(shard_size), int(total_frames) - start))


def expected_shard_bytes(frame_count: int, feature_shape: Sequence[int]) -> int:
    elements = int(frame_count)
    for value in feature_shape:
        elements *= int(value)
    return elements * torch.tensor([], dtype=torch.bfloat16).element_size()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bfloat16_shard(
    tensor: torch.Tensor,
    *,
    shard_path: str | Path,
    done_path: str | Path,
    shard_index: int,
) -> dict[str, Any]:
    if tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16:
        raise ValueError("VGGT cache shards must be CPU bfloat16 tensors.")
    if tensor.ndim != 4 or tensor.shape[0] <= 0:
        raise ValueError(
            "VGGT cache shard must be [frames,views,tokens,dim], got "
            f"{tuple(tensor.shape)}."
        )
    tensor = tensor.contiguous()
    output = Path(shard_path)
    done = Path(done_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.partial.{uuid.uuid4().hex}"
    tensor.view(torch.uint16).numpy().tofile(temporary)
    digest = _sha256_path(temporary)
    os.replace(temporary, output)
    record = {
        "index": int(shard_index),
        "file": output.name,
        "frames": int(tensor.shape[0]),
        "bytes": int(output.stat().st_size),
        "sha256": digest,
    }
    atomic_write_json(record, done)
    return record


def read_valid_shard_record(
    *,
    shard_path: str | Path,
    done_path: str | Path,
    shard_index: int,
    frame_count: int,
    feature_shape: Sequence[int],
) -> dict[str, Any] | None:
    shard = Path(shard_path)
    done = Path(done_path)
    if not shard.is_file() or not done.is_file():
        return None
    try:
        with done.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    expected_bytes = expected_shard_bytes(frame_count, feature_shape)
    if (
        int(record.get("index", -1)) != int(shard_index)
        or record.get("file") != shard.name
        or int(record.get("frames", -1)) != int(frame_count)
        or int(record.get("bytes", -1)) != expected_bytes
        or shard.stat().st_size != expected_bytes
        or not isinstance(record.get("sha256"), str)
    ):
        return None
    return record


def build_manifest(
    *,
    descriptor: Mapping[str, Any],
    feature_shape: Sequence[int],
    shard_size: int,
    shard_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    shape = [int(value) for value in feature_shape]
    if len(shape) != 3 or shape[-1] != int(descriptor["teacher"]["feature_dim"]):
        raise ValueError(f"Invalid VGGT cache feature shape: {shape}")
    return {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "fingerprint": descriptor_fingerprint(descriptor),
        "descriptor": dict(descriptor),
        "feature_shape": shape,
        "shard_size": int(shard_size),
        "shards": [dict(record) for record in shard_records],
    }


class VGGTTeacherFeatureCache:
    """Strict, per-worker lazy mmap reader for precomputed teacher features."""

    def __init__(
        self,
        *,
        cache_root: str | Path,
        descriptor: Mapping[str, Any],
        episode_ends: Sequence[int],
    ) -> None:
        self.descriptor = dict(descriptor)
        self.fingerprint = descriptor_fingerprint(self.descriptor)
        self.cache_dir = resolve_cache_dir(cache_root, self.descriptor)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing VGGT teacher cache manifest: {manifest_path}. "
                "Run scripts/precompute_vggt_teacher_features.py first."
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format") != CACHE_FORMAT or int(manifest.get("version", -1)) != CACHE_VERSION:
            raise ValueError(f"Unsupported VGGT teacher cache format in {manifest_path}.")
        if manifest.get("fingerprint") != self.fingerprint or manifest.get("descriptor") != self.descriptor:
            raise ValueError(
                "VGGT teacher cache fingerprint mismatch: expected "
                f"{self.fingerprint}, got {manifest.get('fingerprint')} in {manifest_path}."
            )

        self.total_frames = int(self.descriptor["total_frames"])
        self.feature_shape = tuple(int(value) for value in manifest["feature_shape"])
        self.shard_size = int(manifest["shard_size"])
        if len(self.feature_shape) != 3 or self.shard_size <= 0:
            raise ValueError(f"Invalid VGGT teacher cache layout in {manifest_path}.")
        records = manifest.get("shards")
        expected_count = num_shards(self.total_frames, self.shard_size)
        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError(
                f"VGGT teacher cache requires {expected_count} shards, found "
                f"{0 if not isinstance(records, list) else len(records)}."
            )
        self._records = []
        for shard_index, manifest_record in enumerate(records):
            frame_count = shard_frame_count(self.total_frames, self.shard_size, shard_index)
            record = read_valid_shard_record(
                shard_path=self.cache_dir / "shards" / shard_filename(shard_index),
                done_path=self.cache_dir / "shards" / shard_done_filename(shard_index),
                shard_index=shard_index,
                frame_count=frame_count,
                feature_shape=self.feature_shape,
            )
            if record is None or record != manifest_record:
                raise ValueError(f"VGGT teacher cache shard {shard_index} is missing or incomplete.")
            self._records.append(record)

        self.episode_ends = tuple(int(value) for value in episode_ends)
        if not self.episode_ends or self.episode_ends[-1] != self.total_frames:
            raise ValueError(
                "VGGT cache episode boundaries must end at total_frames: "
                f"ends[-1]={self.episode_ends[-1] if self.episode_ends else None}, "
                f"total_frames={self.total_frames}."
            )
        if any(left >= right for left, right in zip(self.episode_ends, self.episode_ends[1:])):
            raise ValueError("VGGT cache episode ends must be strictly increasing.")
        self._mapped_shards: dict[int, torch.Tensor] = {}
        self._mapped_pid: int | None = None

    @property
    def num_views(self) -> int:
        return self.feature_shape[0]

    @property
    def tokens_per_view(self) -> int:
        return self.feature_shape[1]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_mapped_shards"] = {}
        state["_mapped_pid"] = None
        return state

    def _mapped_shard(self, shard_index: int) -> torch.Tensor:
        pid = os.getpid()
        if self._mapped_pid != pid:
            self._mapped_shards = {}
            self._mapped_pid = pid
        mapped = self._mapped_shards.get(shard_index)
        if mapped is None:
            frames = int(self._records[shard_index]["frames"])
            elements = frames
            for value in self.feature_shape:
                elements *= value
            mapped = torch.from_file(
                str(self.cache_dir / "shards" / shard_filename(shard_index)),
                shared=False,
                size=elements,
                dtype=torch.bfloat16,
            ).view(frames, *self.feature_shape)
            self._mapped_shards[shard_index] = mapped
        return mapped

    def get(self, index: int) -> torch.Tensor:
        index = int(index)
        if not 0 <= index < self.total_frames:
            raise IndexError(f"VGGT cache index {index} is outside [0,{self.total_frames}).")
        shard_index, row_index = divmod(index, self.shard_size)
        return self._mapped_shard(shard_index)[row_index]

    def get_pair(
        self,
        index: int,
        *,
        future_frame_index: int,
        global_sample_stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        index = int(index)
        if future_frame_index < 0 or global_sample_stride <= 0:
            raise ValueError("Future frame index must be non-negative and stride must be positive.")
        episode_index = bisect.bisect_right(self.episode_ends, index)
        if episode_index >= len(self.episode_ends):
            raise IndexError(f"VGGT cache index {index} has no episode boundary.")
        episode_end = self.episode_ends[episode_index]
        requested = index + int(future_frame_index) * int(global_sample_stride)
        future_is_pad = requested >= episode_end
        future_index = min(requested, episode_end - 1)
        return self.get(index), self.get(future_index), future_is_pad


__all__ = [
    "CACHE_FORMAT",
    "CACHE_VERSION",
    "DEFAULT_SHARD_SIZE",
    "VGGTTeacherFeatureCache",
    "assigned_shard_indices",
    "atomic_write_json",
    "build_cache_descriptor",
    "build_manifest",
    "descriptor_fingerprint",
    "expected_shard_bytes",
    "num_shards",
    "read_valid_shard_record",
    "resolve_cache_dir",
    "shard_done_filename",
    "shard_filename",
    "shard_frame_count",
    "write_bfloat16_shard",
]
