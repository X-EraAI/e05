#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export E05_ENV="${E05_ENV:-$ROOT/.envs/e05}"
export CONDA_PKGS_DIRS="$ROOT/.cache/conda" PIP_CACHE_DIR="$ROOT/.cache/pip"
export XDG_CACHE_HOME="$ROOT/.cache/xdg" TMPDIR="$ROOT/.cache/tmp"
export CONDATOS="$ROOT/.cache/conda/tos"
mkdir -p "$TMPDIR" "$CONDATOS"
if [[ -e "$E05_ENV/conda-meta/history" ]]; then
 echo "Conda prefix exists; continuing locked Python dependency installation: $E05_ENV"
else
"${CONDA_EXE:-conda}" create -y --copy --override-channels \
 -c https://repo.anaconda.com/pkgs/main -c https://conda.anaconda.org/conda-forge \
 --prefix "$E05_ENV" --file "$ROOT/environment.conda.lock.txt"
fi
source "$ROOT/scripts/env.sh"
cd "$ROOT"
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
"$E05_ENV/bin/python" -m pip install --index-url https://pypi.org/simple --no-deps -r requirements.lock.txt
"$E05_ENV/bin/python" -m pip install --index-url https://pypi.org/simple --no-deps --no-build-isolation -e .
"$E05_ENV/bin/python" scripts/check_env.py
