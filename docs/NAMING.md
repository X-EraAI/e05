# e05 public model name and checkpoint compatibility

The released model is named **e05**. Public code uses:

```python
from e05 import E05, create_e05
```

Hydra model configurations use `_target_: e05.create_e05`; RoboTwin launchers use
`task=e05` and the public policy entrypoint `e05_policy`. The canonical implementation class is `E05` in
`src/fastwam/models/wan22/e05.py`. The internal `fastwam` dependency package remains
available so shared modules and existing integration code retain their imports.

The original 109k checkpoint's serialized format identifier remains
`action_vggt_lang_recon_fastwam`. Changing that identifier inside the `.pt` would
change the published file, so it is intentionally preserved along with parameter
keys and checkpoint metadata. Model architecture, initialization, optimizer
precision and normalization are unchanged by the public-name migration.

The old Python class/factory and `task=robotwin_langrecon` remain compatibility
aliases. New examples and configurations use e05. Source run identifiers and
historical validation logs are provenance, so they retain their recorded names.
Obsolete pre-migration smoke outputs have been removed from this checkout.
The retained validation summaries describe the e05 smoke run; recorded run and
checkpoint paths are provenance, not a requirement to keep local smoke weights.
