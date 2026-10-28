"""Extract a complete language-conditioned VGGT branch from an e05 checkpoint."""
import argparse
from collections import Counter, OrderedDict
from collections.abc import Mapping
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


COMPONENTS = {"aggregator", "camera_head", "dense_head", "language_projector"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(source, output, report_path, source_run):
    source, output, report_path = map(Path, (source, output, report_path))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {output}")
    if len({source.resolve(), output.resolve(), report_path.resolve()}) != 3:
        raise ValueError("Source, output and report must be distinct files.")
    source_hash = sha256(source)
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Source checkpoint must be a mapping.")
    state = payload.get("vggt_backbone")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Source must contain a nonempty vggt_backbone state dict.")
    if {key.split(".")[0] for key in state} != COMPONENTS:
        raise ValueError("Expected aggregator, camera_head, dense_head and language_projector.")
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid or non-finite tensor: {key}")
    # Own each tensor's storage: no action/bridge storage can enter the export.
    selected = OrderedDict((key, tensor.detach().cpu().clone()) for key, tensor in state.items())
    if hasattr(state, "_metadata"):
        selected._metadata = copy.deepcopy(state._metadata)
    provenance = {
        "checkpoint": source.name,
        "run": source_run,
        "sha256": source_hash,
        "format": payload.get("format"),
        "step": payload.get("step"),
    }
    exported = {
        "format": "e05_embodied_vggt",
        "version": 1,
        "text_dim": payload["text_dim"],
        "step": payload.get("step"),
        "source": provenance,
        "vggt_backbone": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".extract-vggt-", suffix=".pt", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        torch.save(exported, temporary)
        restored = torch.load(temporary, map_location="cpu", weights_only=True, mmap=True)
        if set(restored) != set(exported) or set(restored["vggt_backbone"]) != set(state):
            raise RuntimeError("Export changed checkpoint or tensor keys.")
        for key, original in state.items():
            actual = restored["vggt_backbone"][key]
            if (original.dtype != actual.dtype or original.shape != actual.shape
                    or not torch.equal(original.reshape(-1).contiguous().view(torch.uint8),
                                       actual.reshape(-1).contiguous().view(torch.uint8))):
                raise RuntimeError(f"Export changed tensor bytes: {key}")
        if sha256(source) != source_hash:
            raise RuntimeError("Source checkpoint changed during extraction.")
        # Exclusive creation also prevents overwriting a concurrently created file.
        temporary.chmod(0o644)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "passed": True,
        "source": provenance,
        "output_file": output.name,
        "output_sha256": sha256(output),
        "output_size_bytes": output.stat().st_size,
        "tensor_count": len(state),
        "tensor_elements": sum(value.numel() for value in state.values()),
        "tensor_bytes": sum(value.numel() * value.element_size() for value in state.values()),
        "components": dict(Counter(key.split(".")[0] for key in state)),
        "dtypes": dict(Counter(str(value.dtype) for value in state.values())),
        "all_tensors_finite": True,
        "tensor_keys_shapes_dtypes_and_bytes_preserved": True,
        "source_sha256_unchanged": True,
        "excluded": ["action_expert", "vggt_bridge", "proprio_encoder", "optimizer"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = extract(args.source, args.output, args.report or args.output.with_suffix(".json"), args.source_run)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
