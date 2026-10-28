#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [[ ! -d "$E05_ROOT/third_party/RoboTwin/assets/embodiments" ]]; then
 echo "RoboTwin assets are missing; install the assets described in docs/INSTALL.md first." >&2
 exit 1
fi
(
 cd "$E05_ROOT/third_party/RoboTwin"
 "$E05_ENV/bin/python" script/update_embodiment_config_path.py
)
"$E05_ENV/bin/python" "$E05_ROOT/scripts/setup_oidn.py"
export TORCH_CUDA_ARCH_LIST="$("$E05_ENV/bin/python" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0))')"
"$E05_ENV/bin/python" - <<'PY'
import importlib
for module,name in [('geom','geom_cu'),('kinematics','kinematics_fused_cu'),('ls','line_search_cu'),('opt','lbfgs_step_cu'),('tensor_step','tensor_step_cu')]:
 m=importlib.import_module('curobo.curobolib.'+module);print(name,getattr(m,name).__file__,flush=True)
PY
