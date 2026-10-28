# Set each worker's compiler/autotune cache before importing DeepSpeed/Triton.
# Shared NFS file locks can allow rank exit callbacks to race on one .tmp file.
import os
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_run_name = Path(os.environ.get("E05_OUTPUT_DIR", "train")).name
_cache = _root / ".cache" / "triton" / _run_name / ("rank_" + os.environ.get("LOCAL_RANK", "0"))
_cache.mkdir(parents=True, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(_cache)

import hydra
from omegaconf import DictConfig

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
