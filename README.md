# E0.5

**From Reconstructive to Future-Predictive 3D Foundations for Robotic World-Action Modeling**

**X-Era AI Lab**

[Paper](https://arxiv.org/pdf/2610.36011) · [Project Page](https://www.x-era.ai/e05/index.html)

English | [简体中文](README_zh-CN.md)

Official implementation of **E0.5**, a world-action model that places 3D foundation models at the core of robotic perception and control. Conditioned on visual observations, language instructions, and robot states, E0.5 jointly predicts robot actions and future 3D geometry, extending geometric reconstruction to the prediction of how the world evolves through interaction.

This repository provides the E0.5 implementation, training and inference workflows, and RoboTwin evaluation tools. Released resources include the RoboTwin policy checkpoint, processed training data, and an embodied-pretrained VGGT backbone for real-robot adaptation. The Python package and configuration entrypoints use the name `e05`; resource requirements and usage instructions are detailed below.

## Model overview

| Component | Specification |
|---|---|
| Visual observations | Three cameras: `cam_high`, `cam_left_wrist`, `cam_right_wrist`, in that order |
| Language input | Natural-language task instruction |
| Robot state | 14 joint/gripper values |
| Action output | 32 × 14 joint-action chunk, restored to dataset units |
| Auxiliary target | Future geometry at an offset of 32 frames |
| Training initialization | RoboTwin: public VGGT-Omega; real robot: e05 embodied-pretrained VGGT. ActionDiT and bridge are randomly initialized. |

Geometry ground truth is required for the reconstruction loss during training, but is not an inference input. The public Python interface is:

```python
from e05 import E05, create_e05
```

Hydra configurations use `e05.create_e05`; RoboTwin uses `task=e05` and `e05_policy`. Implementation and shared modules live under `src/e05/`; the loader remains compatible with the original released checkpoints. See [model naming](docs/NAMING.md) and the [model card](docs/MODEL_CARD.md).

## Installation

Run the following commands from the repository root:

```bash
bash scripts/setup_env.sh
source scripts/env.sh
python scripts/check_env.py
python scripts/download_components.py
```

The installer creates an isolated Conda environment at `.envs/e05`. The provided environment check targets a Linux machine with eight visible CUDA GPUs; standalone model inference uses one GPU. A compatible NVIDIA driver and CUDA toolkit are required. RoboTwin evaluation additionally requires Vulkan rendering support.

Dependencies are recorded in `requirements.lock.txt` and `environment.conda.lock.txt`. Source `scripts/env.sh` in each new shell to configure imports and project-local caches. The component downloader verifies public VGGT initialization weights, the text encoder, and tokenizer against [recorded checksums](configs/resources/component_manifest.json). It does **not** download the trained e05 109k checkpoint, the embodied-pretrained VGGT checkpoint, or processed training data.

See [installation details](docs/INSTALL.md) for environment validation and simulator setup.
Fresh-checkout findings and their scope are recorded in the [independent reproduction notes](docs/INDEPENDENT_REPRODUCTION.md).

## Data and checkpoints

Prepare the resources required by your workflow:

| Workflow | Required resources |
|---|---|
| RoboTwin training | Processed LeRobot data, reconstruction cache, text embeddings and masks, normalization statistics, and public VGGT initialization |
| Real-robot training | Task-specific processed data and statistics, text components, and the e05 embodied-pretrained VGGT checkpoint |
| Standalone inference | A trained e05 checkpoint, its normalization statistics, text components, and an input observation |
| RoboTwin evaluation | A trained e05 checkpoint, its normalization statistics, text components, and compatible simulator assets |

The default local layout is:

```text
data/
├── dataset_stats.json
├── robotwin_clean_xera_v21/
│   ├── data/
│   ├── meta/
│   └── videos/
├── reconstruction_cache/
│   ├── manifest.json
│   └── shards/
└── text_embeds_cache/robotwin/
    ├── manifest.json
    └── shards/
checkpoints/
├── release/
│   ├── step_109000.pt
│   └── dataset_stats.json
├── embodied-vggt/
│   └── e05_embodied_vggt_step_260000.pt
└── ...                         # Public components placed by the downloader
third_party/RoboTwin/assets/    # Required for simulation evaluation
```

Use the normalization statistics associated with the checkpoint. The reference RoboTwin training dataset contains 50 robot tasks, 2,500 episodes, and 549,787 frames, with metadata specifying 15 FPS. Preserve its frame indices and cache correspondence; raw-data collection and label regeneration are outside this repository's scope.

After preparing the reference dataset, validate it with:

```bash
python scripts/validate_data.py --data-root ./data --full
```

For data stored elsewhere, set `E05_DATA_ROOT` to the directory containing `dataset_stats.json` and the three data/cache directories above, and pass that directory to `--data-root`. The validator checks the reference dataset's exact coverage. See the [data card](docs/DATA_CARD.md) for the format and provenance.

The 109k checkpoint is an intermediate checkpoint recorded at epoch 50 of the original 70-epoch budget. Its identity is documented in the [model card](docs/MODEL_CARD.md). Checkpoint resources are listed below; obtaining the required weights and data is a prerequisite to running the corresponding workflows.

| Checkpoint resource | Purpose |
|---|---|
| `release/step_109000.pt` | Complete RoboTwin e05 model for inference and evaluation |
| `embodied-vggt/e05_embodied_vggt_step_260000.pt` | Embodied-pretrained VGGT for real-robot training; extracted from the step-260000 e05 model |

Paths are relative to `checkpoints/`. The embodied checkpoint contains the aggregator, language projector, camera head, and dense head, with no action expert, bridge, or proprio encoder. See the [checkpoint resource manifest](configs/resources/checkpoint_manifest.json) and [checkpoint documentation](docs/CHECKPOINTS.md) for checksums, provenance, and loading instructions. These are separate checkpoint resources, not Git-tracked binaries; public downloads are linked below.

### Download published resources

- [RoboTwin 109k policy](https://huggingface.co/X-EraAI/e05-robotwin)
- [Embodied-pretrained VGGT](https://huggingface.co/X-EraAI/e05-embodied-vggt)
- [Processed RoboTwin dataset](https://huggingface.co/datasets/X-EraAI/e05-robotwin-data)

Run only the downloads needed for your workflow. RoboTwin training uses public VGGT initialization; the 109k policy is for inference/evaluation, and the embodied VGGT is for real-robot training. The processed dataset is approximately 378 GiB; downloading and installing it retains two copies plus extraction overhead. The commands below pin the published Hub revisions.

```bash
source scripts/env.sh
hf download X-EraAI/e05-robotwin --revision 886707967d97c0122e31426b6792799da1322fbd --local-dir checkpoints/release
hf download X-EraAI/e05-embodied-vggt --revision 2fdf396328d0c6808e2d8492fbd80b7f77bbe929 --local-dir checkpoints/embodied-vggt
(cd checkpoints/release && sha256sum -c SHA256SUMS)
(cd checkpoints/embodied-vggt && sha256sum -c SHA256SUMS)
hf download X-EraAI/e05-robotwin-data --repo-type dataset --revision 8c1b06a69506d2cb76dffb74529dfbc5ab894cd1 --local-dir downloads/robotwin-data
python scripts/install_data.py downloads/robotwin-data --output data
python scripts/validate_data.py --data-root data --full
```

## Training

### RoboTwin

RoboTwin training starts from public VGGT-Omega initialization:

```bash
bash scripts/train.sh
```

Defaults are eight GPUs, batch size 8 per GPU, gradient accumulation 1, BF16 mixed precision, and seed 42. VGGT is jointly trained with ActionDiT and the bridge. Image augmentation and the official reconstruction loss are enabled; the reconstruction loss weight is `0.01`, and both learning rates are `1e-4`. W&B is disabled by default.

Run the tested training smoke configuration:

```bash
bash scripts/smoke_train.sh batch_size=8 gradient_accumulation_steps=2
```

This completes four optimizer updates at effective batch size 128, saves a separate smoke checkpoint, and reloads it for action and geometry inference. The full 70-epoch scheduler budget is retained; `stop_after_steps=4` controls only the early stop.

Original RoboTwin training used 32 GPUs with effective batch size 256. On eight GPUs, `bash scripts/train.sh gradient_accumulation_steps=4` matches that batch count, but does not establish exact reproduction of the distributed run. See [training and evaluation](docs/TRAIN_EVAL.md) for configuration and output details.

### Real-robot adaptation

Real-robot training initializes the VGGT branch from our **embodied-pretrained VGGT** checkpoint, rather than the public VGGT-Omega initialization used for RoboTwin. The checkpoint is extracted from the complete e05 model at step 260,000 and preserves the learned language projection and geometry heads. ActionDiT and bridge weights are initialized independently.

```bash
export E05_EMBODIED_VGGT_CHECKPOINT=/path/to/e05_embodied_vggt_step_260000.pt
```

Add `+model_init=embodied_vggt` to the Hydra overrides of your real-robot training command. The [initialization preset](configs/model_init/embodied_vggt.yaml) loads `model.vggt_backbone_pretrained_checkpoint_path` and disables public VGGT and full-model initialization. Do not pass this VGGT-only resource as `resume` or `model.pretrained_checkpoint_path`.

This preset changes model initialization only. Supply a real-robot task configuration with the appropriate data, normalization, camera order, action/state dimensions, and reconstruction targets. The default training task and the smoke results above remain RoboTwin; applying the initialization preset does not itself define or validate a real-robot training pipeline.

## Standalone inference

Prepare an NPZ observation with these fields:

| Field | Type and shape |
|---|---|
| `cam_high`, `cam_left_wrist`, `cam_right_wrist` | `uint8[H, W, 3]`, RGB |
| `state` | `float32[14]`, current joint/gripper state |
| `instruction` | Scalar Unicode string |

```bash
python scripts/infer.py \
  --input /path/to/observation.npz \
  --checkpoint /path/to/step_109000.pt \
  --output runs/prediction.npz \
  --reconstruction
```

Place `dataset_stats.json` beside the checkpoint. Output `action` has shape `[32, 14]` in the original dataset units; `normalized_action` is provided for inspection. Omit `--reconstruction` when geometry outputs are not needed. Predicted depth uses the learned scene-normalized scale, not metres.

If the reference dataset is installed at the default path, `python scripts/make_example.py` extracts an observation into `examples/observation.npz`. Inference on your own observation does not require the training dataset or geometry ground truth.

## RoboTwin evaluation

Prepare compatible assets under `third_party/RoboTwin/assets/`, then compile simulator dependencies and run the evaluation smoke test:

```bash
bash scripts/prepare_sim.sh
export E05_CHECKPOINT=/path/to/step_109000.pt
export E05_STATS_PATH=/path/to/dataset_stats.json
bash scripts/smoke_eval.sh
```

The smoke test schedules eight tasks across eight GPUs, with at most one worker per GPU. Each task runs one clean/seen and one random/unseen episode, for 16 rollouts total. The protocol uses seed 42, action horizon 32, replan interval 24, and 10 denoising steps.

To evaluate all 50 tasks with 100 episodes per task in each setting:

```bash
bash scripts/evaluate.sh EVALUATION.eval_num_episodes=100 EVALUATION.phase=both
```

Logs, resolved configurations, and results are saved under `runs/`. The full evaluation command was not rerun as part of the code validation.

## License and acknowledgments

Repository code is covered by [LICENSE](LICENSE), with third-party materials retaining their respective terms. This project builds on VGGT-Omega, FastWAM, RoboTwin, cuRobo, and Wan/T5 components. VGGT-Omega includes noncommercial research terms; the repository's code license does not relicense model weights, processed data, or simulator assets. Consult [third-party notices](THIRD_PARTY_NOTICES.md) and the licenses retained alongside those materials.
