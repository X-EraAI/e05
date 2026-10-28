#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p artifacts
"$E05_ENV/bin/python" -m pytest -q tests --junitxml=artifacts/regression.xml "$@" 2>&1 | tee artifacts/regression.log
