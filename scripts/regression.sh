#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p outputs/validation
"$E05_ENV/bin/python" -m pytest -q tests --junitxml=outputs/validation/regression.xml "$@" 2>&1 | tee outputs/validation/regression.log
