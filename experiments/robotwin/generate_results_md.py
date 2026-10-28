#!/usr/bin/env python3
"""Convert RoboTwin per-checkpoint summaries into one wide table per run."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


STEP_RE = re.compile(r"^(?:step_)?(\d+)$")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate results.md for RoboTwin evaluation runs."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=project_root / "evaluate_results" / "robotwin",
        help="RoboTwin evaluation root (default: evaluate_results/robotwin).",
    )
    parser.add_argument("--run", help="Only process this run directory.")
    return parser.parse_args()


def finite_rate(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    rate = float(value)
    return rate if math.isfinite(rate) else None


def read_summary(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"WARNING: skipping invalid summary {path}: {error}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"WARNING: skipping non-object summary {path}", file=sys.stderr)
        return None
    return data


def format_rate(value: object) -> str:
    rate = finite_rate(value)
    return "N/A" if rate is None else f"{rate * 100.0:.2f}%"


def render_markdown(run_name: str, summaries: list[tuple[int, dict]]) -> str:
    task_names: list[str] = []
    seen_tasks: set[str] = set()
    rates_by_step: dict[int, dict[str, tuple[object, object]]] = {}

    for step, summary in summaries:
        task_rates: dict[str, tuple[object, object]] = {}
        per_task = summary.get("per_task")
        if isinstance(per_task, list):
            for task in per_task:
                if not isinstance(task, dict):
                    continue
                task_name = str(task.get("task_name", ""))
                task_rates[task_name] = (
                    task.get("clean_success_rate"),
                    task.get("random_success_rate"),
                )
                if task_name not in seen_tasks:
                    seen_tasks.add(task_name)
                    task_names.append(task_name)
        rates_by_step[step] = task_rates

    headers = ["task"]
    for step, _ in summaries:
        headers.extend([f"{step}steps_clean", f"{step}steps_random"])
    lines = [
        f"# {run_name}",
        "",
        "| " + " | ".join(headers) + " |",
        "| :--- | " + " | ".join("---:" for _ in headers[1:]) + " |",
    ]

    for task_name in task_names:
        values = [task_name]
        for step, _ in summaries:
            clean_rate, random_rate = rates_by_step[step].get(
                task_name, (None, None)
            )
            values.extend([format_rate(clean_rate), format_rate(random_rate)])
        lines.append("| " + " | ".join(values) + " |")

    overall_values = ["Overall"]
    for _, summary in summaries:
        overall = summary.get("overall")
        if not isinstance(overall, dict):
            overall = {}
        overall_values.extend(
            [
                format_rate(overall.get("clean_mean_success_rate")),
                format_rate(overall.get("random_mean_success_rate")),
            ]
        )
    lines.extend(["| " + " | ".join(overall_values) + " |", ""])
    return "\n".join(lines)


def summaries_for_run(run_dir: Path) -> list[tuple[int, dict]]:
    summaries = []
    for summary_path in run_dir.glob("*/summary.json"):
        match = STEP_RE.fullmatch(summary_path.parent.name)
        if not match:
            continue
        summary = read_summary(summary_path)
        if summary is not None:
            summaries.append((int(match.group(1)), summary))
    return sorted(summaries, key=lambda item: item[0])


def find_run_directories(results_root: Path) -> list[Path]:
    """Find runs in both ``<run>/<step>`` and grouped layouts.

    Some evaluations are stored as ``<group>/<run>/<step>/summary.json`` while
    older ones live directly under ``<run>/<step>/summary.json``.  A run is
    therefore identified from the parent of any valid checkpoint directory,
    instead of assuming a fixed depth below ``results_root``.
    """
    run_dirs = {
        summary_path.parent.parent
        for summary_path in results_root.rglob("summary.json")
        if STEP_RE.fullmatch(summary_path.parent.name)
    }
    return sorted(run_dirs, key=lambda path: path.relative_to(results_root).parts)


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    if not results_root.is_dir():
        print(f"ERROR: results root does not exist: {results_root}", file=sys.stderr)
        return 2

    wrote = 0
    for run_dir in find_run_directories(results_root):
        if args.run and run_dir.name != args.run:
            continue
        summaries = summaries_for_run(run_dir)
        if not summaries:
            continue
        output_path = run_dir / "results.md"
        output_path.write_text(
            render_markdown(run_dir.name, summaries),
            encoding="utf-8",
        )
        print(f"Wrote {output_path} ({len(summaries)} checkpoints)")
        wrote += 1

    if wrote == 0:
        print("No matching RoboTwin summary.json files found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
