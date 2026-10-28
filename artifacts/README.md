# Release evidence and resource manifests

This directory retains resource identities, fixed Hub revisions, current validation
summaries, and historical benchmark results. It is not a checkpoint or training
output directory.

- `checkpoint_manifest.json` and `component_manifest.json`: released checkpoints and upstream components.
- `huggingface_release.json`: pinned uploads, model loading/inference, and full dataset installation validation.
- `training_smoke.json`, `training_smoke_reload.json`, `evaluation_smoke.json`: completed e05 smoke checks.
- `train_smoke/` and `eval_smoke/`: supporting configurations and structured records; console logs were removed.
- `historical/`: historical benchmark denominators and results, separate from smoke results.
- `cleanup_manifest.json`: local generated files removed before code publication.

Recorded run/checkpoint/verification paths are historical provenance. Local smoke
weights, temporary audit scripts, and downloaded verification copies have been
removed. Re-run the documented smoke commands to generate new outputs.
