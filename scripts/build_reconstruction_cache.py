"""Build the per-frame reconstruction cache used for real-robot depth-only training.

Inputs:
  * a LeRobot v2.1 dataset (meta/info.json, meta/episodes.jsonl, data/*.parquet);
  * one depth file per episode, ``<depth-dir>/episode_<episode_index:06d>.npz``,
    holding one ``[T,H,W]`` array per camera key (T = episode length). Depth must
    be registered to the camera's RGB image (same field of view and pixel grid,
    possibly at a different resolution with the same aspect ratio);
  * per-camera intrinsics (JSON) and, optionally, static world-to-camera
    extrinsics (JSON). Without extrinsics every view stores the identity
    ``[I | 0]`` placeholder.

Output: ``<output>/<index:08d>.pt`` for every value of the dataset's ``index``
column, plus ``manifest.json``. The files follow the contract of
``e05.datasets.lerobot.vggt_reconstruction_cache``; list the output directory in
``data.train.reconstruction_cache_roots`` (one entry per dataset).

By default depth is stored at the VGGT input size of each view (the size the
processor gives the RGB frame for ``--vggt-image-size``, e.g. 640x480 ->
288x224), using nearest-neighbour resampling, crop and padding identical to
``prepare_vggt_reconstruction_targets``. Intrinsics are transformed accordingly.
Run with ``scripts/env.sh`` sourced (it puts ``src`` and VGGT-Omega on PYTHONPATH).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from e05.datasets import cache_io
from e05.datasets.cache_io import cache_exists, shard_path
from e05.datasets.lerobot.processors.e05_processor import prepare_vggt_reconstruction_targets
from e05.datasets.lerobot.vggt_reconstruction_cache import (
    VGGTReconstructionTargetCache,
    reconstruction_dataset_fingerprint,
)

BUILD_PARAMS = "build_params.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True, type=Path, help="LeRobot v2.1 dataset root.")
    p.add_argument("--camera-keys", required=True, nargs="+",
                   help="Camera keys in VGGT order; must equal processor.vggt_camera_keys.")
    p.add_argument("--depth-dir", required=True, type=Path,
                   help="Directory of per-episode depth .npz files.")
    p.add_argument("--depth-pattern", default="episode_{episode_index:06d}.npz",
                   help="File name pattern inside --depth-dir (default: %(default)s).")
    p.add_argument("--depth-scale", type=float, default=None,
                   help="Multiplier converting stored depth values to meters, e.g. 0.001 for "
                        "millimeters. Required for integer arrays; defaults to 1.0 for float arrays.")
    p.add_argument("--intrinsics", required=True, type=Path,
                   help="JSON: {camera: {fx, fy, cx, cy, width, height}} or {camera: {K, width, height}}.")
    p.add_argument("--extrinsics", type=Path, default=None,
                   help="Optional JSON: {camera: 3x4 or 4x4 OpenCV world-to-camera matrix}.")
    p.add_argument("--output", required=True, type=Path, help="Cache directory to create or resume.")
    p.add_argument("--vggt-image-size", type=int, default=256,
                   help="processor.vggt_image_size of the training configuration.")
    p.add_argument("--keep-native-resolution", action="store_true",
                   help="Store depth at the RGB resolution instead of the VGGT input size.")
    p.add_argument("--omit-valid", action="store_true",
                   help="Do not store the validity mask. The loader then uses depth > 0, which is "
                        "equivalent because invalid pixels are stored as 0. Saves a third of the size.")
    p.add_argument("--pack-shards", action="store_true",
                   help="Finally pack frames into shards/<index // 1000>.zip (read by e05.datasets.cache_io).")
    p.add_argument("--workers", type=int, default=1, help="Parallel episode workers.")
    p.add_argument("--check-samples", type=int, default=64, help="Frames re-read for the final self-check.")
    return p.parse_args(argv)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rgb_shape(info, key):
    feature = info["features"].get(f"observation.images.{key}")
    if feature is None:
        known = sorted(k[len("observation.images."):] for k in info["features"] if k.startswith("observation.images."))
        raise KeyError(f"Camera {key!r} is not a dataset feature (observation.images.{key}); known: {known}")
    shape, names = list(feature["shape"]), feature.get("names") or ["height", "width", "channels"]
    try:
        return int(shape[names.index("height")]), int(shape[names.index("width")])
    except ValueError:
        return int(shape[0]), int(shape[1])


def load_intrinsics(path, keys):
    raw = json.loads(Path(path).read_text())
    result = {}
    for key in keys:
        if key not in raw:
            raise KeyError(f"Intrinsics file {path} has no entry for camera {key!r}")
        entry = raw[key]
        if "K" in entry:
            k = np.asarray(entry["K"], dtype=np.float64)
        else:
            k = np.array([[entry["fx"], 0.0, entry["cx"]], [0.0, entry["fy"], entry["cy"]], [0.0, 0.0, 1.0]])
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError(f"Invalid intrinsics for {key!r}: {k.tolist()}")
        if "width" not in entry or "height" not in entry:
            raise KeyError(f"Intrinsics for {key!r} must give the width and height of their pixel grid")
        result[key] = (k, int(round(float(entry["width"]))), int(round(float(entry["height"]))))
    return result


def load_extrinsics(path, keys):
    if path is None:
        identity = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
        return {key: identity for key in keys}
    raw = json.loads(Path(path).read_text())
    result = {}
    for key in keys:
        if key not in raw:
            raise KeyError(f"Extrinsics file {path} has no entry for camera {key!r}")
        e = np.asarray(raw[key], dtype=np.float64)
        if e.shape == (4, 4):
            e = e[:3]
        if e.shape != (3, 4) or not np.isfinite(e).all():
            raise ValueError(f"Extrinsics for {key!r} must be a finite 3x4 or 4x4 matrix")
        result[key] = e
    return result


def episode_indices(dataset_root, info, episode):
    """Return the ``index`` column of one episode after checking its layout."""
    chunk = episode["episode_index"] // int(info.get("chunks_size", 1000))
    path = dataset_root / info["data_path"].format(episode_chunk=chunk, episode_index=episode["episode_index"])
    table = pq.read_table(path, columns=["index", "frame_index", "episode_index"])
    index = table.column("index").to_numpy().astype(np.int64)
    frame = table.column("frame_index").to_numpy()
    owner = table.column("episode_index").to_numpy()
    if len(index) != int(episode["length"]):
        raise ValueError(f"{path}: {len(index)} rows, episodes.jsonl says {episode['length']}")
    if not (np.array_equal(frame, np.arange(len(frame))) and (owner == episode["episode_index"]).all()):
        raise ValueError(f"{path}: frame_index/episode_index columns are not 0..T-1 of this episode")
    if len(index) and not np.array_equal(index, index[0] + np.arange(len(index))):
        raise ValueError(f"{path}: the index column is not contiguous")
    return index


def metric_depth(array, scale, label):
    if np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_:
        if scale is None:
            raise ValueError(f"{label}: integer depth needs an explicit --depth-scale (e.g. 0.001 for mm)")
        return array.astype(np.float32) * np.float32(scale)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{label}: unsupported depth dtype {array.dtype}")
    return array.astype(np.float32) * np.float32(1.0 if scale is None else scale)


def view_target_shape(height, width, image_size):
    """Mirror the processor: crop to aspect [0.5, 2], then VGGT's balanced size."""
    from vggt_omega.utils.load_fn import _balanced_target_shape

    crop_h, crop_w = height, width
    aspect = height / max(width, 1)
    if aspect < 0.5:
        crop_w = min(width, max(1, int(round(height / 0.5))))
    elif aspect > 2.0:
        crop_h = min(height, max(1, int(round(width * 2.0))))
    return (crop_h, crop_w), _balanced_target_shape(crop_h / max(crop_w, 1), image_size, 16)


class FrameBuilder:
    """Turns one frame of per-camera metric depth into a cache entry."""

    def __init__(self, args, rgb_shapes, intrinsics, extrinsics):
        self.keys = list(args.camera_keys)
        self.image_size = int(args.vggt_image_size)
        self.native = bool(args.keep_native_resolution)
        self.store_valid = not args.omit_valid
        self.rgb_shapes = rgb_shapes
        # Intrinsics expressed on each camera's RGB pixel grid.
        ks = []
        for key in self.keys:
            k, width, height = intrinsics[key]
            rgb_h, rgb_w = rgb_shapes[key]
            if abs(width / height - rgb_w / rgb_h) > 0.01 * rgb_w / rgb_h:
                raise ValueError(f"{key}: intrinsics grid {width}x{height} and RGB {rgb_w}x{rgb_h} differ in aspect ratio")
            k = k.copy()
            k[0] *= rgb_w / width
            k[1] *= rgb_h / height
            ks.append(k)
        self.intrinsics = torch.as_tensor(np.stack(ks), dtype=torch.float32)
        self.extrinsics = torch.as_tensor(np.stack([extrinsics[key] for key in self.keys]), dtype=torch.float32)
        # Only the RGB shape is read from these placeholders.
        self.images = {key: torch.empty((1, 1, *rgb_shapes[key]), dtype=torch.uint8) for key in self.keys}

    def register(self, key, depth):
        """Bring one ``[T,H,W]`` depth stack onto the RGB pixel grid (nearest)."""
        rgb_h, rgb_w = self.rgb_shapes[key]
        h, w = depth.shape[-2:]
        if (h, w) == (rgb_h, rgb_w):
            return depth
        if abs(w / h - rgb_w / rgb_h) > 0.01 * rgb_w / rgb_h:
            raise ValueError(f"{key}: depth {w}x{h} and RGB {rgb_w}x{rgb_h} differ in aspect ratio; register depth to RGB first")
        return torch.nn.functional.interpolate(depth[:, None], size=(rgb_h, rgb_w), mode="nearest-exact")[:, 0]

    def frame(self, depth):
        """``depth``: float32 ``[V,H_rgb,W_rgb]`` in meters (non-finite or <= 0 = invalid)."""
        valid = torch.isfinite(depth) & (depth > 0)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        if self.native:
            target = {"depth": depth, "valid": valid, "intrinsics": self.intrinsics.clone(),
                      "extrinsics": self.extrinsics}
        else:
            target = prepare_vggt_reconstruction_targets(
                self.images, self.keys, depth, valid, self.intrinsics, self.extrinsics, self.image_size)
        stored = target["depth"].to(torch.float16)
        valid = target["valid"] & torch.isfinite(stored) & (stored > 0)
        stored = torch.where(valid, stored, torch.zeros_like(stored)).contiguous()
        entry = {"depth": stored, "intrinsics": target["intrinsics"].float().contiguous(),
                 "extrinsics": target["extrinsics"].float().contiguous()}
        if self.store_valid:
            entry["valid"] = valid.contiguous()
        return entry


def process_episode(job):
    args, rgb_shapes, intrinsics, extrinsics, episode, indices = job
    torch.set_num_threads(1)
    out = Path(args.output)
    todo = [i for i, index in enumerate(indices) if not cache_exists(out / f"{int(index):08d}.pt")]
    if not todo:
        return episode["episode_index"], 0, len(indices)
    path = Path(args.depth_dir) / args.depth_pattern.format(episode_index=episode["episode_index"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing depth file for episode {episode['episode_index']}: {path}")
    builder = FrameBuilder(args, rgb_shapes, intrinsics, extrinsics)
    stacks = []
    with np.load(path, allow_pickle=False) as payload:
        for key in builder.keys:
            if key not in payload.files:
                raise KeyError(f"{path} has no array {key!r}; found {payload.files}")
            array = payload[key]
            if array.ndim == 4 and array.shape[-1] == 1:
                array = array[..., 0]
            if array.ndim != 3 or array.shape[0] != len(indices):
                raise ValueError(f"{path}[{key}] must be [T,H,W] with T={len(indices)}, got {array.shape}")
            depth = torch.from_numpy(metric_depth(array, args.depth_scale, f"{path}[{key}]"))
            stacks.append(builder.register(key, depth))
    for i in todo:
        entry = builder.frame(torch.stack([stack[i] for stack in stacks]))
        target = out / f"{int(indices[i]):08d}.pt"
        tmp = target.with_name(f".{target.name}.tmp{os.getpid()}")
        torch.save(entry, tmp)
        os.replace(tmp, target)
    return episode["episode_index"], len(todo), len(indices) - len(todo)


def pack_shards(out):
    groups = {}
    for path in out.glob("*.pt"):
        if path.stem.isdecimal():
            groups.setdefault(shard_path(path), []).append(path)
    for shard, files in sorted(groups.items()):
        shard.parent.mkdir(exist_ok=True)
        tmp = shard.with_name(shard.name + ".tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as dst:
            names = {f.name for f in files}
            if shard.is_file():
                with zipfile.ZipFile(shard) as src:
                    for name in src.namelist():
                        if name not in names:
                            dst.writestr(name, src.read(name))
            for f in sorted(files):
                dst.write(f, f.name)
        os.replace(tmp, shard)
        for f in files:
            f.unlink()
    cache_io._members.cache_clear()


def file_digest(path):
    return None if path is None else hashlib.sha256(Path(path).read_bytes()).hexdigest()


def directory_size(out):
    return sum(p.stat().st_size for p in out.rglob("*") if p.is_file())


def self_check(args, out, all_indices, expected_hw):
    cache = VGGTReconstructionTargetCache(out, args.camera_keys, allow_missing=False, validate=True)
    rng = random.Random(0)
    picks = sorted({int(all_indices[0]), int(all_indices[-1]),
                    *(int(rng.choice(all_indices)) for _ in range(max(0, args.check_samples - 2)))})
    ratios, depths = [], []
    for index in picks:
        entry = cache.get(index)
        if tuple(entry["depth"].shape) != (len(args.camera_keys), *expected_hw):
            raise ValueError(f"Frame {index}: depth shape {tuple(entry['depth'].shape)}, expected {expected_hw}")
        ratios.append(entry["valid"].float().mean(dim=(1, 2)))
        depths.append(entry["depth"][entry["valid"]])
    ratios = torch.stack(ratios).mean(0)
    values = torch.cat(depths)
    if values.numel() > 2_000_000:
        values = values[torch.randperm(values.numel(), generator=torch.Generator().manual_seed(0))[:2_000_000]]
    quantiles = torch.quantile(values.float(), torch.tensor([0.01, 0.5, 0.99])).tolist() if values.numel() else []
    return {"checked_frames": len(picks),
            "valid_ratio": {k: round(float(v), 4) for k, v in zip(args.camera_keys, ratios)},
            "depth_m_p01_p50_p99": [round(v, 4) for v in quantiles]}


def main(argv=None):
    args = parse_args(argv)
    root, out = args.dataset_root.resolve(), args.output.resolve()
    info = json.loads((root / "meta" / "info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v2."):
        raise ValueError(f"Expected a LeRobot v2.x dataset, got codebase_version={info.get('codebase_version')}")
    if len(set(args.camera_keys)) != len(args.camera_keys):
        raise ValueError("--camera-keys must not repeat a camera")
    episodes = sorted(read_jsonl(root / "meta" / "episodes.jsonl"), key=lambda e: e["episode_index"])
    rgb_shapes = {key: rgb_shape(info, key) for key in args.camera_keys}
    intrinsics = load_intrinsics(args.intrinsics, args.camera_keys)
    extrinsics = load_extrinsics(args.extrinsics, args.camera_keys)
    fingerprint = reconstruction_dataset_fingerprint(root)

    if args.keep_native_resolution:
        shapes = {rgb_shapes[k] for k in args.camera_keys}
        if len(shapes) != 1 or any(view_target_shape(*rgb_shapes[k], args.vggt_image_size)[0] != rgb_shapes[k]
                                   for k in args.camera_keys):
            raise ValueError("--keep-native-resolution requires cameras with one RGB resolution that the processor "
                             "does not crop (aspect ratio between 1:2 and 2:1). Use the default storage instead.")
        expected_hw = rgb_shapes[args.camera_keys[0]]
    else:
        sizes = [view_target_shape(*rgb_shapes[k], args.vggt_image_size)[1] for k in args.camera_keys]
        expected_hw = (max(s[0] for s in sizes), max(s[1] for s in sizes))

    params = {
        "camera_keys": list(args.camera_keys),
        "storage": "native" if args.keep_native_resolution else "vggt_input",
        "vggt_image_size": int(args.vggt_image_size),
        "stored_hw": list(expected_hw),
        "valid_stored": not args.omit_valid,
        "depth_scale_to_meters": args.depth_scale,
        "intrinsics_sha256": file_digest(args.intrinsics),
        "extrinsics_sha256": file_digest(args.extrinsics),
        "dataset_fingerprint": fingerprint,
    }
    out.mkdir(parents=True, exist_ok=True)
    params_path = out / BUILD_PARAMS
    if params_path.is_file():
        previous = json.loads(params_path.read_text())
        if previous != params:
            changed = sorted(k for k in set(previous) | set(params) if previous.get(k) != params.get(k))
            raise ValueError(f"{out} was built with different settings ({changed}); use a new output directory.")
    elif any(out.iterdir()):
        raise ValueError(f"{out} is not empty and has no {BUILD_PARAMS}; use a new output directory.")
    else:
        params_path.write_text(json.dumps(params, indent=2) + "\n")
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        manifest_path.unlink()  # rewritten as complete only after every frame exists

    print(f"dataset {root.name}: {len(episodes)} episodes, {info['total_frames']} frames; cameras "
          + ", ".join(f"{k} {rgb_shapes[k][1]}x{rgb_shapes[k][0]}" for k in args.camera_keys)
          + f"; stored depth {expected_hw[1]}x{expected_hw[0]} float16"
          + ("" if args.extrinsics else "; extrinsics: identity placeholder"), flush=True)
    jobs, all_indices = [], []
    for episode in episodes:
        indices = episode_indices(root, info, episode)
        all_indices.extend(indices.tolist())
        jobs.append((args, rgb_shapes, intrinsics, extrinsics, episode, indices))
    if len(set(all_indices)) != len(all_indices) or len(all_indices) != int(info["total_frames"]):
        raise ValueError("The dataset's index column is not unique or does not cover total_frames.")
    if all_indices != list(range(len(all_indices))):
        print("note: the index column is not 0..N-1 in episode order; use reconstruction_cache_roots, "
              "not reconstruction_cache_root.", flush=True)

    start, written, skipped = time.time(), 0, 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(process_episode, jobs)
            for n, (episode_index, new, old) in enumerate(results, 1):
                written, skipped = written + new, skipped + old
                if n % 10 == 0 or n == len(jobs):
                    print(f"[{n}/{len(jobs)}] written {written}, existing {skipped}, {time.time() - start:.0f}s", flush=True)
    else:
        for n, job in enumerate(jobs, 1):
            episode_index, new, old = process_episode(job)
            written, skipped = written + new, skipped + old
            if n % 10 == 0 or n == len(jobs):
                print(f"[{n}/{len(jobs)}] written {written}, existing {skipped}, {time.time() - start:.0f}s", flush=True)

    if args.pack_shards:
        pack_shards(out)
    missing = [i for i in all_indices if not cache_exists(out / f"{i:08d}.pt")]
    if missing:
        raise RuntimeError(f"{len(missing)} frames are missing, e.g. {missing[:5]}; rerun to resume.")
    manifest = {
        "complete": True,
        "camera_keys": list(args.camera_keys),
        "depth_units": "meters",
        "extrinsics": "opencv_world_to_camera",
        "image_convention": "opencv",
        "dataset_fingerprint": fingerprint,
        "num_frames": len(all_indices),
        "index_range": [min(all_indices), max(all_indices)],
        "extrinsics_source": "static_json" if args.extrinsics else "identity_placeholder",
        "depth_dtype": "float16",
        "generator": "scripts/build_reconstruction_cache.py",
        **{k: params[k] for k in ("storage", "vggt_image_size", "stored_hw", "valid_stored", "depth_scale_to_meters")},
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, manifest_path)

    report = self_check(args, out, all_indices, expected_hw)
    size = directory_size(out)
    report.update({"frames": len(all_indices), "written": written, "existing": skipped,
                   "total_bytes": size, "bytes_per_frame": round(size / max(1, len(all_indices))),
                   "elapsed_s": round(time.time() - start, 1)})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
