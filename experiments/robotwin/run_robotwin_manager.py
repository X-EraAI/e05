import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

if __package__:
    from .multinode import finish_node, shard_tasks, start_node
else:
    from multinode import finish_node, shard_tasks, start_node

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
EVAL_STEP_LIMIT_FILE = PROJECT_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_names",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
        "EVALUATION.phase",
        # The official RoboTwin protocol is phase-dependent: clean uses
        # seen instructions and randomized uses unseen instructions.  The
        # manager sets this explicitly for every worker below.
        "EVALUATION.instruction_type",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _load_all_tasks() -> list[str]:
    if not EVAL_STEP_LIMIT_FILE.exists():
        raise FileNotFoundError(f"Task list file not found: {EVAL_STEP_LIMIT_FILE}")
    with EVAL_STEP_LIMIT_FILE.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {EVAL_STEP_LIMIT_FILE}")
    tasks = list(task_map.keys())
    # Keep original order and remove duplicates.
    seen = set()
    dedup_tasks: list[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup_tasks.append(task)
    return dedup_tasks


def _resolve_task_selection(
    *,
    task_name: Any,
    task_names: Any,
) -> list[str]:
    scalar_task = None if task_name is None else str(task_name).strip()
    if scalar_task == "":
        scalar_task = None

    selected_tasks: list[str] | None = None
    if task_names is not None:
        if isinstance(task_names, (list, tuple, ListConfig)):
            selected_tasks = [str(task).strip() for task in task_names]
        else:
            raise TypeError(
                "EVALUATION.task_names must be a YAML list of task names, "
                f"got {type(task_names).__name__}"
            )
        selected_tasks = [task for task in selected_tasks if task]
        if not selected_tasks:
            selected_tasks = None

    if scalar_task is not None and selected_tasks is not None:
        raise ValueError(
            "Set only one of EVALUATION.task_name or EVALUATION.task_names, not both"
        )
    if scalar_task is not None:
        return [scalar_task]
    if selected_tasks is None:
        return _load_all_tasks()

    dedup_tasks: list[str] = []
    seen: set[str] = set()
    for task in selected_tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup_tasks.append(task)
    return dedup_tasks


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return last_value


def _phase_result_filename(phase: str) -> str:
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _resolve_eval_phases(value: Any) -> tuple[str, ...]:
    """Resolve the manager phase selector into an ordered phase sequence."""
    phase = str(value).strip().lower()
    phases = {
        "clean": ("clean",),
        "random": ("random",),
        "both": ("clean", "random"),
    }.get(phase)
    if phases is None:
        raise ValueError(
            "EVALUATION.phase must be one of ['clean', 'random', 'both'], "
            f"got {value!r}"
        )
    return phases


def _load_existing_task_rates(
    summary_json: Path,
    tasks: list[str],
) -> dict[str, dict[str, float | None]]:
    """Load prior rates so clean/random-only runs can be merged in-place."""
    rates: dict[str, dict[str, float | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    if not summary_json.is_file():
        return rates
    try:
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return rates
    if not isinstance(payload, dict) or not isinstance(payload.get("per_task"), list):
        return rates
    task_set = set(tasks)
    for item in payload["per_task"]:
        if not isinstance(item, dict):
            continue
        task_name = str(item.get("task_name", ""))
        if task_name not in task_set:
            continue
        for phase in ("clean", "random"):
            value = item.get(f"{phase}_success_rate")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rates[task_name][phase] = float(value)
    return rates


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _prepare_worker_env(
    *, output_dir: Path, task_name: str, phase: str,
) -> dict[str, str]:
    env = os.environ.copy()
    cache_parent = output_dir / ".warp_cache"
    cache_parent.mkdir(parents=True, exist_ok=True)
    # Warp reads WARP_CACHE_PATH, not WARP_CACHE_ROOT. Fresh directories also
    # isolate concurrent managers/nodes using the same evaluation output path.
    env["WARP_CACHE_PATH"] = tempfile.mkdtemp(
        prefix=f"{task_name}_{phase}_", dir=str(cache_parent),
    )
    return env


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    phase: str  # "clean" | "random"
    process: subprocess.Popen[str]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    configured_gpu_ids = cfg.MULTIRUN.get("gpu_ids")
    if configured_gpu_ids is None:
        gpu_ids = list(range(num_gpus))
    else:
        gpu_ids = [int(gpu_id) for gpu_id in configured_gpu_ids]
        if not gpu_ids:
            raise ValueError("`MULTIRUN.gpu_ids` must not be empty.")
        if any(gpu_id < 0 for gpu_id in gpu_ids):
            raise ValueError("`MULTIRUN.gpu_ids` must contain non-negative integers.")
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError("`MULTIRUN.gpu_ids` must not contain duplicates.")

    run_output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    tasks = _resolve_task_selection(
        task_name=cfg.EVALUATION.task_name,
        task_names=cfg.EVALUATION.get("task_names"),
    )
    eval_phases = _resolve_eval_phases(cfg.EVALUATION.get("phase", "both"))

    extra_overrides = _collect_worker_overrides()

    num_nodes = int(cfg.MULTIRUN.get("num_nodes", 1))
    node_rank = int(cfg.MULTIRUN.get("node_rank", 0))
    all_tasks = tasks
    tasks = shard_tasks(all_tasks, num_nodes, node_rank)
    aggregate_output_dir = run_output_dir
    run_id = cfg.MULTIRUN.get("run_id")
    if num_nodes > 1:
        raw_output_dir = OmegaConf.to_container(cfg.EVALUATION, resolve=False)["output_dir"]
        if "${now:" in str(raw_output_dir):
            raise ValueError("Multi-node evaluation requires an explicit, shared "
                             "EVALUATION.output_dir (not the default timestamp directory)")
        # Ignore machine-local scheduling and unused training output timestamps.
        signature = OmegaConf.to_container(cfg, resolve=True)
        for key in ("MULTIRUN", "gpu_id", "output_dir"):
            signature.pop(key, None)
        signature["ckpt"] = str(ckpt_path)
        signature["EVALUATION"].pop("output_dir", None)
        run_output_dir = start_node(
            aggregate_output_dir, run_id=run_id, num_nodes=num_nodes,
            node_rank=node_rank, tasks=all_tasks, phases=eval_phases,
            signature=signature,
        )

    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"

    # A clean-only run followed by a random-only run can share output_dir and
    # retain the previously measured phase in summary.json. The default
    # both-phase run intentionally starts a fresh summary for both phases.
    if len(eval_phases) == 1:
        task_rates = _load_existing_task_rates(aggregate_output_dir / "summary.json", tasks)
    else:
        task_rates = {
            task: {"clean": None, "random": None} for task in tasks
        }
    failed_records: list[dict[str, Any]] = []
    pending_tasks = deque(tasks)
    running_states: list[RunningState] = []

    phase_to_task_config = {
        "clean": "demo_clean",
        "random": "demo_randomized",
    }
    phase_to_instruction_type = {
        "clean": "seen",
        "random": "unseen",
    }

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def build_cmd(*, task_name: str, gpu_id: int, phase: str) -> list[str]:
        task_config = phase_to_task_config[phase]
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={str(ckpt_path)}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={task_config}",
            f"EVALUATION.output_dir={str(run_output_dir)}",
            f"EVALUATION.instruction_type={phase_to_instruction_type[phase]}",
        ]
        cmd.extend(extra_overrides)
        return cmd

    def launch_phase(task_name: str, gpu_id: int, phase: str) -> RunningState:
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id, phase=phase)
        # Warp's NVRTC compilation is not safe when several evaluation
        # processes share the same kernel cache: concurrent workers can remove
        # one another's .pch files. Give every worker a fresh cache directory.
        env = _prepare_worker_env(
            output_dir=run_output_dir, task_name=task_name, phase=phase,
        )
        log(
            f"launch task={task_name} phase={phase} gpu={gpu_id} "
            f"instruction_type={phase_to_instruction_type[phase]} "
            f"warp_cache={env['WARP_CACHE_PATH']} cmd={' '.join(cmd)}"
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
        )
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            phase=phase,
            process=process,
        )

    def terminate_all_running() -> None:
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(f"terminating task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
            state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"killing task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()

    def gpu_running_count(gpu_id: int) -> int:
        count = 0
        for state in running_states:
            if state.gpu_id != gpu_id:
                continue
            if state.process.poll() is None:
                count += 1
        return count

    def try_launch_pending(gpu_id: int) -> None:
        while len(pending_tasks) > 0 and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            task_name = pending_tasks.popleft()
            running_states.append(
                launch_phase(
                    task_name=task_name,
                    gpu_id=gpu_id,
                    phase=eval_phases[0],
                )
            )

    def write_outputs() -> None:
        clean_mean = _mean_or_none([task_rates[t]["clean"] for t in tasks])
        random_mean = _mean_or_none([task_rates[t]["random"] for t in tasks])

        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
            for task in tasks:
                writer.writerow(
                    [
                        task,
                        task_rates[task]["clean"],
                        task_rates[task]["random"],
                    ]
                )
            writer.writerow(["__overall__", clean_mean, random_mean])

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "clean_success_rate": _to_jsonable(task_rates[task]["clean"]),
                    "random_success_rate": _to_jsonable(task_rates[task]["random"]),
                }
                for task in tasks
            ],
            "overall": {
                "clean_mean_success_rate": _to_jsonable(clean_mean),
                "random_mean_success_rate": _to_jsonable(random_mean),
            },
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(
                    f"{rec['task_name']},{rec['phase']},gpu={rec['gpu_id']},"
                    f"return_code={rec['return_code']},reason={rec['reason']}\n"
                )

    log(
        f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} phases={list(eval_phases)} "
        f"node_rank={node_rank} num_nodes={num_nodes} "
        f"output_dir={run_output_dir}"
    )

    # Launch initial tasks for each GPU up to capacity.
    for gpu_id in gpu_ids:
        try_launch_pending(gpu_id)

    has_failure = False
    failure_message = ""

    while len(running_states) > 0:
        progressed = False
        for state in list(running_states):
            gpu_id = state.gpu_id
            return_code = state.process.poll()
            if return_code is None:
                continue
            progressed = True
            running_states.remove(state)

            if return_code != 0:
                has_failure = True
                failure_message = (
                    f"worker failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, return_code={return_code}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "process_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            result_file = run_output_dir / state.task_name / _phase_result_filename(state.phase)
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                has_failure = True
                failure_message = (
                    f"result parse failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, error={repr(exc)}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "result_parse_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            task_rates[state.task_name][state.phase] = success_rate
            log(
                f"done task={state.task_name} phase={state.phase} gpu={gpu_id} "
                f"success_rate={success_rate:.4f}"
            )

            phase_index = eval_phases.index(state.phase)
            if phase_index + 1 < len(eval_phases):
                running_states.append(
                    launch_phase(
                        task_name=state.task_name,
                        gpu_id=gpu_id,
                        phase=eval_phases[phase_index + 1],
                    )
                )
            else:
                try_launch_pending(gpu_id)

        if has_failure:
            break
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    # Mark not started tasks when failure happened.
    if has_failure:
        for task_name in pending_tasks:
            failed_records.append(
                {
                    "task_name": task_name,
                    "phase": "not_started",
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")

    if num_nodes > 1:
        merged = finish_node(aggregate_output_dir, run_id, node_rank, success=not has_failure)
        if merged:
            log(f"all nodes finished; merged summary: {aggregate_output_dir / 'summary.json'}")
        elif not has_failure:
            log("local shard finished; the last successful node will merge all results")

    if has_failure:
        raise RuntimeError(failure_message)

    log("manager finished successfully")


if __name__ == "__main__":
    main()
