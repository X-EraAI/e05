# Vendored VGGT-Omega

This directory contains a locally adapted source snapshot of the upstream
VGGT-Omega repository. The upstream Git metadata is intentionally not included;
the commit below is the base version for the vendored code.

- Upstream project: https://github.com/facebookresearch/vggt-omega
- Upstream branch at import: `main`
- Upstream commit: `39a0cb8af88554f15ddcb5354cd52bde588fa014`
- Upstream license: see [`LICENSE`](./LICENSE)

## FastWAM local modifications

The vendored source contains these changes relative to the pinned commit:

- `vggt_omega/models/aggregator.py`: adds selectable attention K/V callbacks,
  callback-layer validation, optional patch embedding injection, and a
  `capture_only` execution mode used by FastWAM.
- `vggt_omega/models/layers/attention.py`: exposes attention K/V tensors to a
  callback, with configurable autograd detachment.
- `vggt_omega/models/layers/block.py`: propagates K/V callbacks through the
  self-attention block implementation.
- `vggt_omega/utils/load_fn.py`: accepts channel-first PyTorch tensors in
  addition to file paths.
- `requirements.txt`: removes restrictive Torch, TorchVision, and NumPy version
  constraints so the parent FastWAM environment owns dependency resolution.
- `README.md`: documents FastWAM's distributed checkpoint and capture workflow.
- `tests/vggt_fastwam/`: adds regression coverage for tensor image loading and
  the K/V callback integration.

These adaptations are applied directly to the vendored files. To reconstruct
an exact patch, check out the upstream commit above separately and diff it
against this directory, excluding `README.vendor.md`.

When updating VGGT-Omega, reapply and validate every item in this list, then
replace the pinned commit hash.
