# Installation

Linux with eight compatible CUDA GPUs is the supported smoke target. The local
validation uses the package versions in `requirements.lock.txt` and the Conda
packages in `environment.conda.lock.txt`. The validated eight-GPU smoke used
batch 8 per GPU with gradient accumulation 2
and peaked at approximately 59.47 GiB of Torch allocation per GPU. Driver, context
and reserved memory need additional headroom. This is not a measured minimum
VRAM requirement; batch 16 exceeded device memory on the validation node.

`bash scripts/setup_env.sh` creates an isolated prefix. Always source
`scripts/env.sh` before running Python directly. It resets PYTHONPATH and places
all compiler, dataset, download and temporary caches under this project.
Do not copy an existing environment's editable-install `.pth` files.
The installer uses explicit official Conda channels and PyPI, and resumes Python
dependency installation if the Conda prefix already exists. If Conda requests
channel terms acceptance, review and accept the terms for your own installation.
A directory existing at `.envs/e05` does not by itself mean installation succeeded.

The Git repository contains code, configurations and documentation. Trained e05
weights, processed data and simulator assets are supplied separately; no release
archive is required. The 109k checkpoint, embodied-pretrained VGGT checkpoint and processed dataset
are published under [X-EraAI on Hugging Face](https://huggingface.co/X-EraAI).
See the [pinned download and installation commands](../README.md#download-published-resources). See [checkpoint resources](CHECKPOINTS.md) for their intended uses.

Download the publicly available initialization and language components:

```bash
source scripts/env.sh
python scripts/download_components.py
```

Files are pinned by SHA256 in `configs/resources/component_manifest.json`. Repository
revisions may be mutable; the downloader refuses bytes that differ from the
recorded release. The separate VGGT aggregator is a tensor-identical extraction
of the public text-alignment checkpoint, not an internally trained model.
Serialization may change its file checksum across Torch versions; tensor equality
with the full public checkpoint is the compatibility criterion.

RoboTwin source and local integration changes are vendored at the recorded
upstream commit in `third_party/RoboTwin/README.vendor.md`. Use the compatible
asset snapshot if available separately, or obtain the matching assets
from the RoboTwin project and compare against `configs/resources/simulator_assets.jsonl`.
Required simulator task configurations are included in
`third_party/RoboTwin/task_config`: clean/random evaluation, camera definitions,
embodiment paths and episode step limits. Keep these files in Git.
Place assets under `third_party/RoboTwin/assets`; never link them into another
working checkout. Asset licenses are independent of the simulator code license.

For the Aloha robot used by the supplied RoboTwin configuration, download the
upstream archives at this fixed revision into this checkout:

```bash
source scripts/env.sh
hf download TianxingChen/RoboTwin2.0 \
  --repo-type dataset --revision 3dc3b798668feb99ac61cc9086d84cbcc3d79186 \
  --include background_texture.zip embodiments.zip objects.zip \
  --local-dir downloads/robotwin-assets
python - <<'PYASSETS'
from pathlib import Path
import zipfile
root = Path("third_party/RoboTwin/assets").resolve()
root.mkdir(parents=True, exist_ok=True)
for name in ("background_texture.zip", "embodiments.zip", "objects.zip"):
    with zipfile.ZipFile(Path("downloads/robotwin-assets") / name) as archive:
        for member in archive.infolist():
            if "__MACOSX" in Path(member.filename).parts:
                continue
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe archive path: {member.filename}")
            archive.extract(member, root)
PYASSETS
bash scripts/prepare_sim.sh
```

`prepare_sim.sh` installs OIDN 2.3.3 from a SHA256-pinned upstream archive into
the isolated SAPIEN package. SAPIEN's bundled OIDN 2.0.1 lacks support for the
Blackwell validation GPUs; the older environment had an unrecorded local upgrade.
The installer retains upstream license notices and refuses to modify an environment
outside this checkout. OIDN denoising remains enabled.

It also generates the embodiment planner configurations with paths for
this checkout before compiling cuRobo for the detected GPU architecture. The
upstream assets match the historical Aloha setup after path normalization;
four other robots' planner templates differ in acceleration/jerk settings from
the historical snapshot. This asset source does not assert an identical setup
for those other robots.

The launch environment supplies `third_party/curobo/src` directly and uses local
JIT caches. It does not depend on an installation in the source project's environment.
The system CUDA compiler and NVIDIA Vulkan/driver libraries remain system prerequisites.

Run `python scripts/check_env.py` to verify module origins, cache paths and eight
visible GPUs. Do not start GPU smoke while an unrelated job lacks sufficient free
memory; do not terminate other users' workloads to make room.

A fresh installation from these lockfiles was replayed in an independent checkout
with the installation fixes included here. `pip check` and the module-origin check
passed. This environment check alone does not establish training or evaluation results.
For the resource layout and commands, see [README](../README.md) or
[中文说明](../README_zh-CN.md).
