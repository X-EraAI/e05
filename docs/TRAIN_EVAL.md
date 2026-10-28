# Training and evaluation

## RoboTwin

Source `scripts/env.sh` first. Public initialization is downloaded and verified by
`scripts/download_components.py`. Default RoboTwin training uses all processed training data,
8 GPUs, batch 8/GPU, accumulation 1, original augmentation, geometry loss and model.

```bash
bash scripts/train.sh
bash scripts/train.sh gradient_accumulation_steps=4
bash scripts/smoke_train.sh
```

The first configuration has global batch 64; the second has global batch 256.
Neither resumes the source run's 32-rank optimizer state. Schedules are calculated
from the selected single-node recipe. Smoke stops after four updates using
`stop_after_steps`, without changing the configured training budget.

Outputs are separate timestamped directories under `runs/`. A directory's
`config.yaml`, `dataset_stats.json`, logs and checkpoint belong together. Passing
`resume=/path/to/weights.pt` loads weights only; full-state directory restore
requires matching model, optimizer grouping, schedule and world size. Loading the
109k model for more training is not part of this release's smoke validation.

```bash
bash scripts/prepare_sim.sh
bash scripts/smoke_eval.sh
bash scripts/evaluate.sh EVALUATION.eval_num_episodes=100 EVALUATION.phase=both
```

The manager maps clean to seen instructions and random to unseen instructions.
The two phases change both visual randomization and language distribution.
Eight independent workers are assigned across eight GPUs, at most one per GPU.
Default policy parameters: 32-step horizon, replan every 24 actions, 10 denoising
steps, seed 42, skip intermediate rendering within an action chunk.

Smoke uses the original 109k weights and eight specified tasks for sixteen total
rollouts. Success/failure of manipulation is recorded, while worker failures or
missing task results fail validation. Never silently remove crashed tasks from
reported means. Full 50-task evaluation is an available command, not executed as
part of the code validation.

## Reduced-batch training smoke

When the eight GPUs are shared, run the explicitly smaller smoke recipe:

```bash
bash scripts/smoke_train.sh batch_size=4
# If batch 4 runs out of memory:
bash scripts/smoke_train.sh batch_size=2
```

Both commands still use 8 GPUs, accumulation 1, full model size, augmentation,
public VGGT initialization, random ActionDiT/bridge, and four optimizer steps.
Their effective batches are 32 and 16 respectively. The 70-epoch scheduler budget
is retained for the selected recipe. Reports record the actual batch and explicitly
distinguish these checks from validation of batch 8/GPU. Resource preflight thresholds
for reduced batches are admission heuristics; an actual run may still encounter OOM.
The published full training and default smoke settings remain batch 8/GPU.

## Public e05 entrypoints

All release configurations use `e05.create_e05`. RoboTwin defaults to `task=e05`.
To attempt a larger smoke batch without changing model, loss or
augmentation settings, run `bash scripts/smoke_train.sh batch_size=16` (8 GPUs,
accumulation 1, effective batch 128). Passing the resource check is not a guarantee
that batch 16 fits; consult the actual smoke report for results.

The validated configuration uses gradient accumulation after batch 16
exhausted device memory:

```bash
bash scripts/smoke_train.sh batch_size=8 gradient_accumulation_steps=2
```

This keeps the effective batch at 128 (8 GPUs × 8 examples × 2 microbatches).
The audit checks 8 microbatches and 4 optimizer updates per rank. It retains the
70-epoch scheduler budget and all model, augmentation and loss settings.

## Real-robot initialization

Real-robot training uses the e05 embodied-pretrained VGGT resource extracted
from `beta_joint_7n_2ep_cosine_reader_w12_v1`, step 260,000. RoboTwin retains
public VGGT-Omega initialization.

Set `E05_EMBODIED_VGGT_CHECKPOINT` to the extracted VGGT file and append
`+model_init=embodied_vggt` to the Hydra overrides of the real-robot training
command. This loads the aggregator, language projector and both geometry heads,
while leaving ActionDiT and bridge independently initialized. The preset clears
public VGGT and full-policy initialization paths. It does not restore optimizer
state and is not a substitute for a real-robot task/data configuration.

Do not use the VGGT-only file as `resume`, `model.pretrained_checkpoint_path`,
or an inference policy checkpoint. See [checkpoint resources](CHECKPOINTS.md)
for the manifest, extraction and loading details. The eight-GPU smoke results
in this document apply to RoboTwin, not real-robot training.
