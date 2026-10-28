import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def expand_episode_ranges(
    episode_ranges: Sequence[Sequence[int]] | None,
    *,
    total_episodes: int | None = None,
) -> list[int] | None:
    """Expand half-open episode ranges while validating bounds and overlap."""
    if episode_ranges is None:
        return None

    episode_indices: list[int] = []
    seen: set[int] = set()
    for range_index, episode_range in enumerate(episode_ranges):
        if len(episode_range) != 2:
            raise ValueError(
                f"episode_ranges[{range_index}] must be [start, end), got {episode_range}"
            )
        start, end = int(episode_range[0]), int(episode_range[1])
        if start < 0 or end <= start:
            raise ValueError(
                f"episode_ranges[{range_index}] must satisfy 0 <= start < end, "
                f"got [{start}, {end})"
            )
        if total_episodes is not None and end > total_episodes:
            raise ValueError(
                f"episode_ranges[{range_index}] ends at {end}, but dataset only has "
                f"{total_episodes} episodes"
            )
        for episode_index in range(start, end):
            if episode_index in seen:
                raise ValueError(f"episode_ranges overlap at episode {episode_index}")
            seen.add(episode_index)
            episode_indices.append(episode_index)

    if not episode_indices:
        raise ValueError("episode_ranges must select at least one episode")
    return episode_indices


def split_episode_indices(
    episode_indices: Sequence[int],
    *,
    val_set_proportion: float,
    is_training_set: bool,
    seed: int,
) -> list[int]:
    """Apply Fast-WAM's deterministic episode-level train/validation split."""
    if not 0.0 <= val_set_proportion < 1.0:
        raise ValueError(
            f"val_set_proportion must be in [0, 1), got {val_set_proportion}"
        )

    selected = list(episode_indices)
    if val_set_proportion < 1e-6:
        return selected

    rng = np.random.default_rng(seed)
    rng.shuffle(selected)
    split_idx = int(len(selected) * (1 - val_set_proportion))
    if is_training_set:
        return selected[:split_idx]
    return selected[split_idx:]


def read_episode_tasks(
    dataset_dir: str | Path,
    episode_indices: Iterable[int],
) -> list[str]:
    """Read unique task instructions referenced by selected LeRobot episodes."""
    episodes_path = Path(dataset_dir) / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes file: {episodes_path}")

    requested = set(int(index) for index in episode_indices)
    if not requested:
        return []

    tasks: list[str] = []
    seen_tasks: set[str] = set()
    found: set[int] = set()
    with episodes_path.open("r", encoding="utf-8") as episodes_file:
        for line_index, line in enumerate(episodes_file):
            if line_index not in requested:
                continue
            record = json.loads(line)
            episode_index = int(record["episode_index"])
            if episode_index != line_index:
                raise ValueError(
                    f"{episodes_path} is not ordered by episode_index: line {line_index + 1} "
                    f"contains episode {episode_index}"
                )
            found.add(episode_index)
            for task in record.get("tasks", []):
                task = str(task)
                if task not in seen_tasks:
                    seen_tasks.add(task)
                    tasks.append(task)

    missing = sorted(requested - found)
    if missing:
        preview = missing[:10]
        suffix = "..." if len(missing) > len(preview) else ""
        raise ValueError(
            f"Selected episodes are missing from {episodes_path}: {preview}{suffix}"
        )
    return tasks
