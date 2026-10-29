# e05 — RoboTwin 109k

Source run: `langrecon_aug_4n_bs8_acc1_ep70_rdma_v1`.
Checkpoint: `step_109000.pt`, step 109,000, recorded epoch 50 of a 70-epoch budget.
Original topology: 4 nodes × 8 GPUs, batch 8/GPU, accumulation 1, global batch 256.
This is an intermediate checkpoint, not the final 70-epoch model.

Three current RGB views and language condition the geometry backbone; the action
branch also receives proprioception. VGGT predicts future geometry at offset 32;
ActionDiT predicts 32 steps of 14-D joint actions. Geometry ground truth is training
supervision, not an input required by the policy at inference.

RoboTwin training: public VGGT-Omega initialization, random ActionDiT/bridge, independent
bridge projections, trainable VGGT, official geometry loss ×0.01, appearance
augmentation enabled, both learning rates 1e-4, BF16 mixed precision, seed 42.
The original weight file contains FP32 VGGT backbone parameters and BF16 action/
bridge parameters; E05 explicitly uses legacy optimizer grouping to retain this
behavior. Current generic training defaults must not replace it implicitly.

The reference weight file was validated without serialization changes. Its
identity is recorded in `docs/validation/checkpoint_integrity.json`. It is not included in the Git repository; download it from
[X-EraAI/e05-robotwin](https://huggingface.co/X-EraAI/e05-robotwin). When using an externally supplied checkpoint, retain its matching
normalization statistics. The portable training configuration is
`configs/release.yaml`.

Historical results use the repaired language-padding-mask inference implementation.
Clean macro success: 89.48% (5,000 evaluated). Random macro success: 49.431867%
(4,961 evaluated of 5,000 planned). Random task `put_bottles_dustbin` used 87/100
and `put_object_cabinet` used 74/100 episodes; do not report this as a complete
10,000-episode benchmark. Historical artifacts preserve these denominators.
Only parameters recoverable from historical evidence should be attributed to that
run; the new smoke protocol is recorded separately and is not a benchmark rerun.

Limitations: simulation joint control on this embodiment; no real-robot safety,
commercial permission, or exact multi-node training reproduction is established
by this release. Respect upstream VGGT model/code terms and dataset provenance.

Geometry output includes `future_scale_convention=scene_normalized`. Predicted
depth is in the learned scene-normalized scale, not directly metres; training
cache depth is metric before the official loss normalization.

Public model ID: `e05`; Python API: `e05.E05` / `e05.create_e05`. The original
serialized checkpoint format and source run IDs remain unchanged for compatibility
and provenance. See the code repository documentation `docs/NAMING.md`.

Real-robot training uses a separate embodied-pretrained VGGT resource extracted
from the complete step-260000 e05 model. See [checkpoint resources](CHECKPOINTS.md).
That resource is a VGGT branch initializer, not the RoboTwin policy documented here.
