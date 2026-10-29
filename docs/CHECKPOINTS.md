# e05 checkpoint resources

Checkpoint files are distributed separately from the Git repository. The catalog
in [checkpoint_manifest.json](../configs/resources/checkpoint_manifest.json) records their
intended use, local paths, sizes and SHA256. Public resources are hosted at [X-EraAI/e05-robotwin](https://huggingface.co/X-EraAI/e05-robotwin) and
[X-EraAI/e05-embodied-vggt](https://huggingface.co/X-EraAI/e05-embodied-vggt). Public upstream VGGT and text components are listed separately in
[component_manifest.json](../configs/resources/component_manifest.json).

| Resource | Use | Default path |
|---|---|---|
| RoboTwin e05, step 109,000 | Complete policy for RoboTwin inference/evaluation | `checkpoints/release/step_109000.pt` |
| Embodied-pretrained VGGT, step 260,000 | Initialize VGGT for real-robot training | `checkpoints/embodied-vggt/e05_embodied_vggt_step_260000.pt` |

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

## Initialization by task

**RoboTwin:** retain the public VGGT-Omega initialization in `configs/release.yaml`
and `configs/task/e05.yaml`. ActionDiT and bridge are randomly initialized. The
109k policy is used for inference/evaluation, not the initialization of this
training recipe.

**Real robot:** initialize the complete language-conditioned VGGT branch from the
embodied-pretrained resource below. ActionDiT and bridge are initialized
independently, allowing the task's action/state dimensions to be configured for
the target embodiment.

```bash
export E05_EMBODIED_VGGT_CHECKPOINT=/path/to/e05_embodied_vggt_step_260000.pt
```

Append `+model_init=embodied_vggt` to the Hydra overrides of the real-robot training
command. The preset at `configs/model_init/embodied_vggt.yaml` sets
`model.vggt_backbone_pretrained_checkpoint_path`, clears full-model and public
VGGT initialization paths, and keeps ActionDiT initialization random. Configure
the real-robot data, statistics, camera order, action/state dimensions and
reconstruction targets separately. This repository's default task remains RoboTwin.

The extracted resource contains `vggt_backbone`, not a complete e05 policy.
Do not use it with `resume`, `model.pretrained_checkpoint_path`, or the standalone
inference `--checkpoint` argument. It is also not an aggregator-only checkpoint;
loading only the aggregator would omit the learned language projection and heads.

## Embodied-pretrained VGGT

Source run: `beta_joint_7n_2ep_cosine_reader_w12_v1`.
Source checkpoint: `step_260000.pt`, full e05 model, recorded step 260,000.
The source's serialized identity is `action_vggt_lang_recon_fastwam`.

The extraction keeps all VGGT parameters and buffers:

| Component | Tensor count |
|---|---:|
| Aggregator | 1,283 |
| Camera head | 68 |
| Dense head | 60 |
| Language projector | 4 |
| **Total** | **1,415** |

Action expert, attention bridge, proprio encoder and optimizer state are excluded.
The export preserves 1,341 FP32 tensors and 74 BF16 tensors without conversion.
It contains 1,149,304,905 tensor elements, including buffers. The exported file is 4,597,332,135 bytes (approximately 4.28 GiB).
Its SHA256 is `ce5698a438b3f246d1d1fe0bcc4d1b205876c0523066cf11f08f8434d98770c5`.
The payload format is:

```text
format: e05_embodied_vggt
version: 1
step: 260000
text_dim: 4096
source: source checkpoint identity and SHA256
vggt_backbone: complete VGGT state_dict
```

The extraction report records byte-for-byte tensor equality, finite values, and
an unchanged source-file digest. See
[extraction verification](validation/embodied_vggt_extraction.json).

The complete `e05.create_e05` construction path was also exercised on CPU with
the initialization preset: all 1,415 VGGT tensor values loaded successfully,
public VGGT paths were disabled, and ActionDiT/bridge probes differed from the
source model. Two existing backbone-loading regression tests passed. See
[loading verification](validation/embodied_vggt_loading.json). The CPU
construction check used FP32 modules; the exported file retains the source
mixed dtypes. No real-robot training or rollout was executed for this extraction.

## Reproduce the extraction

The extraction utility is optional for users who already have the VGGT resource.
It requires the complete source e05 checkpoint and never modifies it:

```bash
source scripts/env.sh
python scripts/extract_embodied_vggt.py \
  --source /path/to/full_e05/step_260000.pt \
  --output checkpoints/embodied-vggt/e05_embodied_vggt_step_260000.pt \
  --source-run beta_joint_7n_2ep_cosine_reader_w12_v1 \
  --report outputs/validation/embodied_vggt_extraction.json
```

An existing output checkpoint is not overwritten. Tensor keys, shapes, dtypes and
bytes are preserved. PyTorch serialization can produce different file-level
checksums for independently regenerated exports; the manifest identifies the
specific file prepared here.

## Terms

The embodied-pretrained checkpoint is derived from VGGT-Omega. Retain the
applicable upstream model and code terms, including noncommercial terms, and the
[third-party notices](../THIRD_PARTY_NOTICES.md). The repository's MIT code license
does not relicense the checkpoint. This extraction and loading validation does
not constitute real-robot training or deployment evaluation.
