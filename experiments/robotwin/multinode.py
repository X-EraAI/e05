"""Shared-filesystem coordination for independent RoboTwin evaluation nodes."""

import argparse
import csv
import fcntl
import io
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path


def shard_tasks(tasks: list[str], num_nodes: int, node_rank: int) -> list[str]:
    if num_nodes < 1:
        raise ValueError("MULTIRUN.num_nodes must be >= 1")
    if not 0 <= node_rank < num_nodes:
        raise ValueError("MULTIRUN.node_rank must be in [0, num_nodes)")
    return tasks[node_rank::num_nodes]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _write_json(path: Path, payload: dict) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))


@contextmanager
def _lock(output_dir: Path):
    directory = output_dir / ".multinode"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "merge.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _run_dir(output_dir: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("Multi-node evaluation requires a fresh MULTIRUN.run_id / "
                         "EVAL_RUN_ID containing only letters, digits, _, - or .")
    return output_dir / ".multinode" / run_id


def start_node(
    output_dir: Path, *, run_id: str, num_nodes: int, node_rank: int,
    tasks: list[str], phases: tuple[str, ...], signature: dict,
) -> Path:
    """Claim one rank once. Reusing an ID is rejected to exclude stale results."""
    shard_tasks(tasks, num_nodes, node_rank)
    directory = _run_dir(output_dir, run_id)
    manifest = {
        "run_id": run_id, "num_nodes": num_nodes, "tasks": tasks,
        "phases": list(phases), "signature": signature,
    }
    with _lock(output_dir):
        manifest_path = directory / "manifest.json"
        active_path = output_dir / ".multinode" / "active.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Multi-node configuration mismatch: all nodes must use "
                                 "the same checkpoint, task list and evaluation settings")
            if json.loads(active_path.read_text())["run_id"] != run_id:
                raise ValueError("This run was superseded; use a fresh EVAL_RUN_ID")
        else:
            _write_json(manifest_path, manifest)
            _write_json(active_path, {"run_id": run_id})
        node_dir = directory / f"node_{node_rank}"
        if node_dir.exists():
            raise ValueError(f"Node rank {node_rank} already claimed for {run_id}; "
                             "use distinct ranks, or a fresh EVAL_RUN_ID for a new run")
        node_dir.mkdir()
    return node_dir


def _write_summary(directory: Path, payload: dict) -> None:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
    for row in payload["per_task"]:
        writer.writerow([row["task_name"], row["clean_success_rate"], row["random_success_rate"]])
    writer.writerow(["__overall__", payload["overall"]["clean_mean_success_rate"],
                     payload["overall"]["random_mean_success_rate"]])
    _atomic_write(directory / "summary.csv", stream.getvalue())
    _atomic_write(directory / "failed_tasks.txt", "")
    _write_json(directory / "summary.json", payload)


def _merge_locked(output_dir: Path, run_id: str) -> bool:
    directory = _run_dir(output_dir, run_id)
    manifest = json.loads((directory / "manifest.json").read_text())
    rows_by_task = {}
    for rank in range(manifest["num_nodes"]):
        node_dir = directory / f"node_{rank}"
        done_path = node_dir / "complete.json"
        if not done_path.exists():
            return False
        if not json.loads(done_path.read_text())["success"]:
            raise ValueError(f"Node {rank} failed; refusing to publish an incomplete summary")
        payload = json.loads((node_dir / "summary.json").read_text())
        expected = shard_tasks(manifest["tasks"], manifest["num_nodes"], rank)
        rows = payload["per_task"]
        if [row["task_name"] for row in rows] != expected:
            raise ValueError(f"Node {rank} task coverage mismatch")
        for row in rows:
            for phase in ("clean", "random"):
                value = row[f"{phase}_success_rate"]
                if value is None and phase not in manifest["phases"]:
                    continue
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f"Invalid or missing {phase} result for {row['task_name']}")
            rows_by_task[row["task_name"]] = row
    rows = [rows_by_task[task] for task in manifest["tasks"]]
    overall = {}
    for phase in ("clean", "random"):
        values = [row[f"{phase}_success_rate"] for row in rows
                  if row[f"{phase}_success_rate"] is not None]
        overall[f"{phase}_mean_success_rate"] = sum(values) / len(values) if values else None
    payload = {"per_task": rows, "overall": overall}
    # A non-numeric parent keeps archived runs out of checkpoint report discovery.
    _write_summary(directory / "aggregate", payload)
    active = json.loads((output_dir / ".multinode" / "active.json").read_text())
    if active["run_id"] != run_id:
        raise ValueError("A newer run owns this output directory; archived this run's "
                         "summary without overwriting the newer run")
    _write_summary(output_dir, payload)
    return True


def finish_node(output_dir: Path, run_id: str, node_rank: int, *, success: bool) -> bool:
    with _lock(output_dir):
        directory = _run_dir(output_dir, run_id) / f"node_{node_rank}"
        _write_json(directory / "complete.json", {"success": success})
        return _merge_locked(output_dir, run_id) if success else False


def merge_run(output_dir: Path, run_id: str) -> bool:
    with _lock(output_dir):
        return _merge_locked(output_dir, run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and merge completed RoboTwin nodes")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not merge_run(args.output_dir.resolve(), args.run_id):
        parser.exit(1, "Some nodes have not finished; no aggregate summary was published.\n")
    print(f"Merged summary: {args.output_dir / 'summary.json'}")
