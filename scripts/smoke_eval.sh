#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
export E05_EVAL_OUTPUT_DIR="${E05_EVAL_OUTPUT_DIR:-$E05_ROOT/runs/smoke_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$E05_EVAL_OUTPUT_DIR"
bash scripts/evaluate.sh EVALUATION.eval_num_episodes=1 EVALUATION.phase=both \
 'EVALUATION.task_names=[adjust_bottle,beat_block_hammer,handover_block,open_microwave,pick_dual_bottles,place_bread_basket,stack_blocks_two,turn_switch]' \
 seed=42 EVALUATION.action_horizon=32 EVALUATION.replan_steps=24 EVALUATION.num_inference_steps=10 \
 "$@" 2>&1 | tee "$E05_EVAL_OUTPUT_DIR/eval.log"
"$E05_ENV/bin/python" scripts/check_eval_smoke.py "$E05_EVAL_OUTPUT_DIR"
