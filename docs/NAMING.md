# E05 implementation and checkpoint compatibility

The public Python package and model are **e05**:

```python
from e05 import E05, create_e05
```

Implementation and shared components live under `src/e05/`. Hydra uses
`_target_: e05.create_e05`; RoboTwin uses `task=e05` and `e05_policy`.
Only the language-conditioned action and geometry reconstruction model is
included. Gen3R adapters, MoE, multistage, loop, latent, depth-only and video-model
variants and their factories are not distributed.

The original 109k and embodied-backbone checkpoint files remain unchanged.
`src/e05/checkpoint_compat.py` translates original serialized identifiers and
saved configuration paths when reading these resources. It does not expose old
Python package/class aliases or load other model formats. New weight checkpoints
use the `e05` format and E05 metadata names.

Action/geometry computation, the released tensor values and normalization remain
unchanged. Historical reports retain their recorded results; obsolete Python class
identifiers are omitted from public summaries, and original reports are archived
locally. Upstream license and copyright attribution remain in the relevant notices.
