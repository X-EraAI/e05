# Validation protocol

The release runs the existing language-conditioned backbone, language padding mask,
checkpoint, loss, current/future target, and augmentation regression tests relevant
to E05. Tests targeting unshipped EE/MoE/finetuning configurations or old `work/`
diagnostics were removed from the copied suite; E05 configuration, standalone
inference and lossless cache-shard tests replace those interfaces.

The original training smoke target uses 8 ranks, batch 8/rank, accumulation 1, original model size,
augmentation enabled and official loss 0.01. `stop_after_steps=4` stops execution;
`max_steps` remains inferred from 70 epochs and the full dataset, retaining the
single-node recipe's learning-rate and intrinsics-warmup schedules. This is not
an exact continuation of the historical 32-rank training run.

The released 109k checkpoint is never initialized from the smoke output. Evaluation
smoke executes 8 tasks × clean/random × 1 episode = 16 rollouts, at seed 42,
action horizon 32, replan 24 and 10 denoising steps. A completed unsuccessful
manipulation is a valid rollout; worker crashes and missing results fail validation.

Environment and execution reports are written to `artifacts/` and `runs/`.
Historical benchmark results live separately in `artifacts/historical/`.

For short smoke runs, `save_state_at_stop=false` saves the step-4 weight checkpoint
without serializing an unused optimizer state. The default is true for normal
training. This output-only option does not change loss, gradients or optimizer steps.

Actual execution status, including incomplete acceptance items, is recorded in
[RELEASE_STATUS.md](RELEASE_STATUS.md). The protocol above is not a claim that
blocked tests have passed.

The pre-migration validation used 8 ranks × batch 4 × accumulation 1. The current
post-rename validation first attempted batch 16; both allocator attempts exhausted
GPU memory (0 completed optimizer updates with the default allocator, 1 with
expandable segments). The user authorized the fallback of
8 ranks × batch 8 × accumulation 2 (effective batch 128). The audit checks exactly
8 microbatches and 4 optimizer updates per rank, six gradient groups, finite losses
and gradients, parameter changes, and absence of training/cleanup tracebacks.
Distributed workers set distinct Triton
cache directories before importing DeepSpeed to avoid shared-NFS exit races.

The current training run and regression suite also passed worker shutdown checks. On Linux, worker
tensor IPC uses abstract Unix sockets, so temporary socket files are not unlinked
on NFS. Persistent workers close before interpreter shutdown. This leaves sampling,
model parameters, augmentation and optimizer semantics unchanged. The earlier
reduced-batch audit additionally compared augmentation hashes before/after the
cleanup fix. See `tests/test_worker_shutdown.py`.
