#!/usr/bin/env bash
# Source this file; do not activate or modify an existing user environment.
E05_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export E05_ROOT
export E05_ENV="${E05_ENV:-$E05_ROOT/.envs/e05}"
export PATH="$E05_ENV/bin:/usr/local/cuda/bin:$PATH"
export CONDA_PREFIX="$E05_ENV"
export PYTHONPATH="$E05_ROOT/src:$E05_ROOT/third_party/vggt-omega:$E05_ROOT/third_party/curobo/src:$E05_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HOME="$E05_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$E05_ROOT/.cache/datasets"
export HF_LEROBOT_HOME="$E05_ROOT/.cache/lerobot"
export MODELSCOPE_CACHE="$E05_ROOT/.cache/modelscope"
export XDG_CACHE_HOME="$E05_ROOT/.cache/xdg"
export TORCH_HOME="$E05_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$E05_ROOT/.cache/torch_extensions"
export TRITON_CACHE_DIR="$E05_ROOT/.cache/triton"
export WARP_CACHE_PATH="$E05_ROOT/.cache/warp"
export NUMBA_CACHE_DIR="$E05_ROOT/.cache/numba"
export MPLCONFIGDIR="$E05_ROOT/.cache/matplotlib"
export CUDA_CACHE_PATH="$E05_ROOT/.cache/cuda"
export TMPDIR="$E05_ROOT/.cache/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PIP_CACHE_DIR="$E05_ROOT/.cache/pip" CONDA_PKGS_DIRS="$E05_ROOT/.cache/conda"
export DIFFSYNTH_MODEL_BASE_PATH="$E05_ROOT/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export WANDB_MODE=disabled
export GIT_PYTHON_REFRESH=quiet
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="$E05_ENV/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
export HYDRA_FULL_ERROR=1
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$HF_DATASETS_CACHE"
cd "$E05_ROOT"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# SAPIEN's bundled discovery only checks /usr/share; honor system ICDs in /etc too.
if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
 for e05_icd in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
  if [[ -f "$e05_icd" ]]; then export VK_ICD_FILENAMES="$e05_icd"; break; fi
 done
 unset e05_icd
fi
