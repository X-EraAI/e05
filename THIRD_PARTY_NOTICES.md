# Third-party materials

Vendored code, upstream model weights, data, and simulator assets retain their
applicable third-party terms.

- VGGT-Omega: facebookresearch/vggt-omega, pinned source commit documented in
  `third_party/vggt-omega/README.vendor.md`. Preserve its FAIR Noncommercial Research
  License and acceptable-use terms in `third_party/vggt-omega/LICENSE`.
- The public initialization mirror model card declares CC-BY-NC-4.0; retain the
  model card and applicable upstream terms. Do not label the complete E05 model
  unrestricted MIT or commercially licensed.
- Vendored official geometry loss and augmentation retain their LICENSE and
  PROVENANCE files under `src/e05`.
- RoboTwin source: MIT, commit `bf44be51cf5717a5595ce59447f2cf5263d2aa95`;
  see `third_party/RoboTwin/LICENSE` and README.vendor.md.
- cuRobo: preserve the NVIDIA notices and LICENSE in `third_party/curobo`.
- Wan/T5 components retain their upstream model terms. The component manifest
  identifies their repositories and SHA256; no Wan video model is required.
- Processed Xera/RoboTwin data, language and simulation assets are separate
  third-party-derived materials. Retain their source attribution and terms;
  RoboTwin's code license alone does not establish their redistribution license.

Local adaptations: portable configuration and entrypoints, isolated execution,
lossless cache shard loading, non-writing inference text cache, smoke evidence,
release packaging, and worker/cache lifecycle fixes for NFS. The released model architecture is retained. Package and public model names are unified under E05; a compatibility reader preserves access to the original released checkpoint files.

Embodied-pretrained VGGT resource: extracted from the VGGT branch of the e05
step-260000 checkpoint. The aggregator, language projector and geometry heads
retain applicable VGGT-Omega upstream model/code terms. This extraction does
not change those terms or apply the repository MIT code license to the weights.
Resource identity is recorded in `configs/resources/checkpoint_manifest.json`.

Open Image Denoise 2.3.3 is downloaded separately from the official RenderKit/oidn
release by scripts/setup_oidn.py to support Blackwell rendering. Its upstream
LICENSE.txt and third-party-programs notices are retained alongside the installed
libraries under the isolated SAPIEN package's oidn_library/licenses-2.3.3 directory.
The repository does not redistribute these binaries in Git.
