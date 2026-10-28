#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$("$E05_ENV/bin/python" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0))')}"
export E05_EVAL_OUTPUT_DIR="${E05_EVAL_OUTPUT_DIR:-$E05_ROOT/runs/eval_$(date +%Y%m%d_%H%M%S)}"
exec "$E05_ENV/bin/python" experiments/robotwin/run_robotwin_manager.py \
 task=e05 MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=1 "$@"
