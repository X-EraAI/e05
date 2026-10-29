#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
export NUM_GPUS=8
export E05_SMOKE_BATCH_SIZE="${E05_SMOKE_BATCH_SIZE:-8}"
export E05_SMOKE_ACCUM="${E05_SMOKE_ACCUM:-1}"
for override in "$@"; do
 if [[ "$override" == batch_size=* ]]; then export E05_SMOKE_BATCH_SIZE="${override#batch_size=}"; fi
 if [[ "$override" == gradient_accumulation_steps=* ]]; then export E05_SMOKE_ACCUM="${override#gradient_accumulation_steps=}"; fi
done
case "$E05_SMOKE_BATCH_SIZE" in 16|8|4|2) ;; *) echo "Smoke batch must be 16, 8, 4, or 2" >&2; exit 2;; esac
export E05_OUTPUT_DIR="${E05_OUTPUT_DIR:-$E05_ROOT/runs/smoke_train_$(date +%Y%m%d_%H%M%S)}"
export E05_SMOKE_AUDIT_DIR="$E05_OUTPUT_DIR/audit"
mkdir -p "$E05_OUTPUT_DIR"
"$E05_ENV/bin/python" scripts/check_training_resources.py
bash scripts/train.sh batch_size="$E05_SMOKE_BATCH_SIZE" gradient_accumulation_steps="$E05_SMOKE_ACCUM" stop_after_steps=4 save_weights_every=4 save_state_every=0 \
 save_state_at_stop=false verify_reconstruction_gradients=true log_every=1 "$@" 2>&1 | tee "$E05_OUTPUT_DIR/train.log"
"$E05_ENV/bin/python" scripts/check_training_smoke.py "$E05_OUTPUT_DIR"
"$E05_ENV/bin/python" scripts/make_example.py
"$E05_ENV/bin/python" scripts/infer.py --input examples/observation.npz \
 --checkpoint "$E05_OUTPUT_DIR/checkpoints/weights/step_000004.pt" \
 --output "$E05_OUTPUT_DIR/reloaded_prediction.npz" --reconstruction --cached-text-only
