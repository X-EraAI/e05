#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
export E05_OUTPUT_DIR="${E05_OUTPUT_DIR:-$E05_ROOT/runs/train_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$E05_OUTPUT_DIR"
exec "$E05_ENV/bin/python" -m accelerate.commands.launch \
 --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
 --num_machines 1 --num_processes "${NUM_GPUS:-8}" --machine_rank 0 \
 --main_process_port "${MASTER_PORT:-29651}" scripts/train.py --config-name release "$@"
