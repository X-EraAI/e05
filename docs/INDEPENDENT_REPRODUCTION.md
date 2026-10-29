# Independent reproduction

This audit starts from `Likeooo/e05` commit
`1b64aa60e71a3faa165c58f1691cb8eabb3a1e4b` in a fresh checkout and Conda prefix.
It tests the README workflow and records required corrections to that commit.
During independent reproduction, the source FastWAM/E05 repositories and their
environments were not used as editable installs or modified. The verified source
fixes were subsequently integrated into E05 with user authorization; existing
environments were not modified. No remote changes were pushed.

## Resource provenance

- RoboTwin policy: `X-EraAI/e05-robotwin`, revision
  `886707967d97c0122e31426b6792799da1322fbd`.
- Embodied VGGT: `X-EraAI/e05-embodied-vggt`, revision
  `2fdf396328d0c6808e2d8492fbd80b7f77bbe929`.
- Processed data: `X-EraAI/e05-robotwin-data`, revision
  `8c1b06a69506d2cb76dffb74529dfbc5ab894cd1`.
- Simulator archives: `TianxingChen/RoboTwin2.0`, revision
  `3dc3b798668feb99ac61cc9086d84cbcc3d79186`.
- Text encoder/tokenizer: independently downloaded from ModelScope and verified
  against `configs/resources/component_manifest.json`.
- Public VGGT: the user authorized copying the already downloaded public weight
  because access to its Hub download was unavailable. This independent file copy
  matches the release manifest SHA256. The aggregator was extracted locally and
  verified tensor-for-tensor against the public weight.

All published model/data files were downloaded independently and checked against
checksums. The dataset contains 50 robot tasks, 2,500 episodes, 549,787 frames,
158,984 language entries, and metadata FPS 15. Full cache archive hashes and
geometry/text coverage checks passed; data and policy normalization statistics
are byte-identical. No videos, labels or reconstruction caches were regenerated.
Local dataset read indices were created in this checkout's cache.

## Corrections discovered by the fresh checkout

1. Explicit Conda channels and canonical conda-forge URLs avoid inherited mirror
   configuration. Anaconda channel terms were accepted with user authorization,
   with acceptance records confined to the new project.
2. The installer explicitly uses PyPI and continues Python dependency installation
   when a Conda prefix exists. A prefix left by an interrupted installation is not
   considered complete. `pip check` and module-origin checks passed after installation.
3. Required RoboTwin task configuration files were missing from Git because the
   vendored ignore file excluded them. The five required configuration-code files
   were recovered from the read-only source project, with hashes recorded in the
   audit. They are now included and protected by a regression check.
4. Embodiment planner configurations must be generated with the new checkout's
   absolute paths before cuRobo compilation. Aloha assets/configurations match the
   historical reference after path normalization. Four non-Aloha planner templates
   differ in acceleration/jerk settings; no other-robot reproduction is claimed.
5. NVIDIA Vulkan ICD discovery now also checks `/etc/vulkan/icd.d`.
6. SAPIEN's bundled OIDN 2.0.1 did not support the validation node's Blackwell GPUs.
   The prior environment contained an unrecorded OIDN 2.3.3 modification. The new
   `setup_oidn.py`, called by `prepare_sim.sh`, downloads a SHA256-pinned official
   2.3.3 archive and installs it only in the isolated prefix, retaining licenses.
   Ray tracing and OIDN remain enabled. See the
   [upstream release](https://github.com/RenderKit/oidn/releases/tag/v2.3.3).
7. Training smoke audit now distinguishes public and embodied initialization.
   Evaluation audit now rejects OIDN errors even if workers finish their rollouts.

The first evaluation attempt failed on missing task configuration. The second
completed 16 rollouts but emitted OIDN errors and was explicitly invalidated.
Neither attempt is accepted as successful reproduction evidence.

## Training checks

The user authorized batch 2 per GPU after batch 4 resource preflight failed on
an occupied node. Both checks used eight GPUs, accumulation 1 (effective batch 16),
full model, BF16, image augmentation, the original official reconstruction loss,
and random action expert/bridge initialization. The full 70-epoch scheduler budget
was retained (`max_steps=2405340`); `stop_after_steps=4` stopped each smoke after
four optimizer updates.

```bash
source scripts/env.sh
E05_OUTPUT_DIR="$E05_ROOT/runs/reproduction_train_public_bs2" \
  bash scripts/smoke_train.sh batch_size=2 gradient_accumulation_steps=1
E05_OUTPUT_DIR="$E05_ROOT/runs/reproduction_train_embodied_bs2" \
  bash scripts/smoke_train.sh batch_size=2 gradient_accumulation_steps=1 \
  +model_init=embodied_vggt
```

Both runs passed all eight-rank audits: finite losses/gradients, no skipped updates,
parameter changes and gradient connectivity through VGGT, language projection,
geometry heads, bridge and action expert. Both saved and reloaded their independent
step-4 checkpoints for finite action `[32,14]` and geometry predictions. Peak Torch
allocated memory was approximately 22.01 GiB per rank; this is not total device
memory or a guaranteed minimum requirement.

These checks use RoboTwin data for both initialization modes. They do not validate
a real-robot task configuration, convergence, or exact reproduction of the original
32-GPU training. Neither training check initializes from the 109k policy.

## Current verification status

- Fresh dependency installation, `pip check`, module/cache isolation: passed.
- Full processed-data installation and validation: passed.
- Existing and added regression tests: 96 passed.
- Original 109k standalone online-text action/geometry inference: passed.
- Public and embodied VGGT eight-GPU batch-2 training plus reload: passed.
- Ray tracing with 32 samples/pixel, depth 8 and OIDN 2.3.3: passed.
- Final eight-GPU evaluation: passed, 16/16 rollouts completed without renderer
  errors or worker exceptions; clean 7/8 and random 3/8 successes. These small
  smoke samples are not benchmark estimates or a rerun of historical evaluation.

Detailed local evidence remains under `audit/reproduction` and
`runs/reproduction_*` in the separate `e05-repro-clean` checkout. These large
logs/checkpoints are intentionally ignored by Git and were not copied into E05. The final evidence audit passed and the portable summary is
[`docs/validation/independent_reproduction.json`](validation/independent_reproduction.json).
The original commit required the corrections listed above; an unmodified checkout
of that commit does not reproduce this result. The verified source fixes and this portable report have been integrated into
E05. No remote changes were pushed during reproduction or integration.
