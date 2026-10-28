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

Files are pinned by SHA256 in `artifacts/component_manifest.json`. Repository
revisions may be mutable; the downloader refuses bytes that differ from the
recorded release. The separate VGGT aggregator is a tensor-identical extraction
of the public text-alignment checkpoint, not an internally trained model.
Serialization may change its file checksum across Torch versions; tensor equality
with the full public checkpoint is the compatibility criterion.

RoboTwin source and local integration changes are vendored at the recorded
upstream commit in `third_party/RoboTwin/README.vendor.md`. Use the compatible
asset snapshot if available separately, or obtain the matching assets
from the RoboTwin project and compare against `artifacts/simulator_assets.jsonl`.
Place assets under `third_party/RoboTwin/assets`; never link them into another
working checkout. Asset licenses are independent of the simulator code license.

Compile cuRobo for the detected GPU architecture with `bash scripts/prepare_sim.sh`.
The launch environment supplies `third_party/curobo/src` directly and uses local
JIT caches. It does not depend on an installation in the source project's environment.
The system CUDA compiler and NVIDIA Vulkan/driver libraries remain system prerequisites.

Run `python scripts/check_env.py` to verify module origins, cache paths and eight
visible GPUs. Do not start GPU smoke while an unrelated job lacks sufficient free
memory; do not terminate other users' workloads to make room.

The local validation used an independent environment with the recorded dependencies.
A complete fresh installation from the lockfiles has not been separately replayed.
For the resource layout and commands, see [README](../README.md) or
[中文说明](../README_zh-CN.md).
