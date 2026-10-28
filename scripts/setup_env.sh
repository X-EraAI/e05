#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export E05_ENV="${E05_ENV:-$ROOT/.envs/e05}"
export CONDA_PKGS_DIRS="$ROOT/.cache/conda" PIP_CACHE_DIR="$ROOT/.cache/pip"
if [[ -e "$E05_ENV/conda-meta/history" ]]; then
 echo "Environment already exists: $E05_ENV. Run scripts/check_env.py to inspect it."; exit 0
fi
"${CONDA_EXE:-conda}" create -y --copy --prefix "$E05_ENV" --file "$ROOT/environment.conda.lock.txt"
source "$ROOT/scripts/env.sh"
"$E05_ENV/bin/python" -m pip install --no-deps -r requirements.lock.txt
"$E05_ENV/bin/python" -m pip install --no-deps --no-build-isolation -e .
"$E05_ENV/bin/python" scripts/check_env.py
