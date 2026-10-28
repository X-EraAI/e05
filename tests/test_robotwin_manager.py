import json
from pathlib import Path

import pytest

from experiments.robotwin.run_robotwin_manager import (
    _load_existing_task_rates,
    _prepare_worker_env,
    _resolve_eval_phases,
)


def test_worker_env_isolates_repeated_launches_and_preserves_curobo(tmp_path, monkeypatch):
    monkeypatch.setenv("WARP_CACHE_PATH", "/shared/old_warp_cache")
    monkeypatch.setenv("PYTHONPATH", "/curobo_sm80")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", "/curobo_sm80/torch_extensions")
    workers = [
        _prepare_worker_env(output_dir=tmp_path, task_name="click_alarmclock", phase=phase)
        for phase in ("clean", "clean", "random")
    ]
    caches = [Path(env["WARP_CACHE_PATH"]) for env in workers]
    assert len(set(caches)) == len(workers)
    for cache, env in zip(caches, workers):
        assert cache.is_dir()
        assert cache.parent == tmp_path / ".warp_cache"
        assert env["PYTHONPATH"] == "/curobo_sm80"
        assert env["TORCH_EXTENSIONS_DIR"] == "/curobo_sm80/torch_extensions"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("clean", ("clean",)),
        ("random", ("random",)),
        ("both", ("clean", "random")),
        (" CLEAN ", ("clean",)),
    ],
)
def test_resolve_eval_phases(value, expected):
    assert _resolve_eval_phases(value) == expected


def test_resolve_eval_phases_rejects_unknown_value():
    with pytest.raises(ValueError, match="EVALUATION.phase"):
        _resolve_eval_phases("invalid")


def test_load_existing_task_rates_preserves_prior_phase(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "per_task": [
                    {
                        "task_name": "task_a",
                        "clean_success_rate": 0.5,
                        "random_success_rate": None,
                    },
                    {
                        "task_name": "ignored",
                        "clean_success_rate": 1.0,
                        "random_success_rate": 1.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _load_existing_task_rates(summary_path, ["task_a", "task_b"]) == {
        "task_a": {"clean": 0.5, "random": None},
        "task_b": {"clean": None, "random": None},
    }


def test_load_existing_task_rates_ignores_invalid_summary(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("not json", encoding="utf-8")

    assert _load_existing_task_rates(summary_path, ["task_a"]) == {
        "task_a": {"clean": None, "random": None},
    }
