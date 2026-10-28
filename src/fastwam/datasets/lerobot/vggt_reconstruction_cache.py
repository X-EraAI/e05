"""Indexed raw metric RGB-D/calibration targets for VGGT reconstruction training.

Each entry is ``<root>/<global_frame:08d>.pt`` containing depth ``[V,H,W]`` in
meters, validity, intrinsics ``[V,3,3]`` and OpenCV camera-from-world
extrinsics ``[V,3,4]``.  Keeping this separate from the legacy patchified depth
cache prevents accidental log-depth normalization before geometry loss.
"""
from pathlib import Path
import bisect
import hashlib
import json
import torch
from fastwam.datasets.cache_io import cache_exists, load_cache


def reconstruction_dataset_fingerprint(root):
    """Bind a cache to the source dataset's episode/task/frame layout."""
    digest = hashlib.sha256()
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl"):
        digest.update(name.encode())
        digest.update((Path(root) / "meta" / name).read_bytes())
    return digest.hexdigest()


class MultiDatasetReconstructionCache:
    """Translate selected/concatenated row indices to original dataset indices.

    The legacy cache remains indexed by concatenated row. This adapter is used
    only with explicit per-dataset roots and never decodes RGB to resolve a row.
    """

    def __init__(self, roots, datasets, camera_keys):
        if len(roots) != len(datasets):
            raise ValueError("reconstruction_cache_roots must match dataset_dirs one-to-one.")
        self.caches, self.indices, self.ends = [], [], []
        total = 0
        for root, dataset in zip(roots, datasets):
            manifest_path = Path(root) / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if not manifest.get("complete"):
                raise ValueError(f"Reconstruction cache is incomplete: {manifest_path}")
            if manifest.get("camera_keys") != list(camera_keys):
                raise ValueError(f"Reconstruction camera order mismatch: {manifest_path}")
            if (manifest.get("depth_units") != "meters"
                    or manifest.get("extrinsics") != "opencv_world_to_camera"
                    or manifest.get("image_convention") != "opencv"):
                raise ValueError(f"Reconstruction geometry convention mismatch: {manifest_path}")
            if manifest.get("dataset_fingerprint") != reconstruction_dataset_fingerprint(dataset.root):
                raise ValueError(f"Reconstruction dataset fingerprint mismatch: {manifest_path}")
            indices = torch.as_tensor(dataset.hf_dataset.with_format(None)["index"], dtype=torch.int64)
            if len(indices) != dataset.num_frames:
                raise ValueError("Reconstruction source index count does not match dataset length.")
            self.indices.append(indices)
            total += len(indices)
            self.ends.append(total)
            self.caches.append(VGGTReconstructionTargetCache(root, camera_keys, allow_missing=False, validate=True))

    def resolve(self, index):
        index = int(index)
        if index < 0 or not self.ends or index >= self.ends[-1]:
            raise IndexError(f"Reconstruction row out of bounds: {index}")
        dataset_index = bisect.bisect_right(self.ends, index)
        start = self.ends[dataset_index - 1] if dataset_index else 0
        return dataset_index, int(self.indices[dataset_index][index - start])

    def get(self, index):
        dataset_index, source_index = self.resolve(index)
        return self.caches[dataset_index].get(source_index)

    def get_pair(self, index, *, future_index, future_is_pad=False):
        if self.resolve(index)[0] != self.resolve(future_index)[0]:
            raise ValueError("Reconstruction targets cannot cross dataset boundaries.")
        return {"current": self.get(index), "future": self.get(future_index),
                "future_is_pad": bool(future_is_pad)}


class VGGTReconstructionTargetCache:
    def __init__(self, root, camera_keys=None, allow_missing=False, validate=False):
        self.root = Path(root).expanduser() if root else None
        self.camera_keys = tuple(camera_keys or ())
        self.allow_missing = bool(allow_missing)
        self.validate = bool(validate)

    def get(self, index: int):
        path = None if self.root is None else self.root / f"{int(index):08d}.pt"
        if path is not None and cache_exists(path):
            payload = load_cache(path)
            required = {"depth", "intrinsics", "extrinsics"}
            if not required.issubset(payload):
                raise ValueError(f"Reconstruction target {path} lacks {required - set(payload)}")
            depth = payload["depth"].float()
            valid = payload.get("valid", depth > 0)
            if self.validate:
                views = len(self.camera_keys)
                if (depth.ndim != 3 or depth.shape[0] != views or valid.shape != depth.shape
                        or payload['intrinsics'].shape != (views, 3, 3)
                        or payload['extrinsics'].shape != (views, 3, 4)):
                    raise ValueError(f"Invalid reconstruction tensor shapes: {path}")
                if (not torch.isfinite(payload['intrinsics']).all()
                        or not torch.isfinite(payload['extrinsics']).all()
                        or not torch.isfinite(depth[valid.bool()]).all()
                        or not (depth[valid.bool()] > 0).all()):
                    raise ValueError(f"Invalid reconstruction geometry: {path}")
            return {"depth": depth, "valid": valid.bool(), "intrinsics": payload["intrinsics"].float(), "extrinsics": payload["extrinsics"].float()}
        if not self.allow_missing:
            raise FileNotFoundError(f"Missing reconstruction target: {path}")
        # Debug-only fallback keeps model construction/smoke tests runnable.
        v = max(len(self.camera_keys), 1)
        depth = torch.ones(v, 240, 320)
        k = torch.eye(3).repeat(v, 1, 1)
        k[:, 0, 0] = k[:, 1, 1] = 256.0
        k[:, 0, 2], k[:, 1, 2] = 160.0, 120.0
        e = torch.zeros(v, 3, 4); e[:, :, :3] = torch.eye(3)
        return {"depth": depth, "valid": torch.ones_like(depth, dtype=torch.bool), "intrinsics": k, "extrinsics": e}

    def get_pair(self, index: int, *, future_index: int, future_is_pad: bool = False):
        """Read current and endpoint targets without mixing episode boundaries.

        ``index`` and ``future_index`` are global dataset-frame indices.  The
        dataset computes ``future_is_pad`` from its episode index, so this
        cache remains a deliberately simple indexed file store.
        """
        current = self.get(int(index))
        future = self.get(int(future_index))
        return {
            "current": current,
            "future": future,
            "future_is_pad": bool(future_is_pad),
        }
