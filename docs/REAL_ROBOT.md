# Real-robot post-training recipe

This page records the configuration used for the AgileX Piper and Zerith H1
experiments and the tools for applying it to your own robot. The real-robot
datasets and checkpoints are not released. The RoboTwin defaults in
`configs/task/e05.yaml` and `configs/release.yaml` are **not** a real-robot
recipe. In particular, their camera and point-map losses must be disabled for
real data, as explained below.

The repository provides:

- [`configs/task/real_robot_template.yaml`](../configs/task/real_robot_template.yaml),
  a task configuration with this recipe and placeholders for everything that
  depends on your robot (cameras, state/action size, data paths);
- [`scripts/build_reconstruction_cache.py`](../scripts/build_reconstruction_cache.py),
  which converts your per-episode depth maps into the reconstruction cache;
- [`scripts/infer.py`](../scripts/infer.py), which runs a trained policy on one
  observation using the cameras and dimensions of its training configuration.

The [workflow](#workflow-for-your-robot) section below walks through them.

## Data

| Item | Setting |
|---|---|
| Format | LeRobot v2.1, 30 fps |
| Cameras | Three RGB views (head camera, left wrist, right wrist, in that VGGT order), recorded at 640×480 |
| State / action | AgileX Piper: 14-D joints and grippers. Zerith H1: 23-D. Each platform uses its own normalization statistics. |
| VGGT input | `vggt_image_size: 256`, so 4:3 frames become 224×288 |
| Action chunk | 32 steps |
| VGGT augmentation | None (`processor.vggt_augmentation: null`); the release configuration enables it for RoboTwin |

## Geometry supervision: depth only

Real-robot post-training keeps the future-geometry objective, with the target
frame 32 steps ahead (the endpoint of the action chunk, the same as RoboTwin). The
supervision is limited to depth:

- **Depth source.** Depth sensors are not used. Each camera view is labeled with
  metric depth (meters) estimated by Depth Anything 3 (`DA3METRIC-LARGE`) on the
  640×480 frames.
- **Intrinsics.** These are the factory calibration read from each camera's SDK.
- **Extrinsics.** None are available. The cache format requires an extrinsics
  tensor, so every view stores the identity `[I | 0]` placeholder. With the
  camera and point-map terms disabled, the placeholders are not supervised
  directly. They still shape the depth supervision, however; see
  [depth normalization](#depth-normalization-and-scale) below.
- **Loss.** `backend: official` with `lambda_vggt_recon: 0.01`. The official
  backend reads only the `official` block. It ignores the legacy `lambda_camera`,
  `lambda_depth`, `lambda_point`, `use_conf_loss` and `gradient_weight` keys, so
  setting those has no effect. Disable the camera and point-map terms by setting
  their official weights to zero. The depth term keeps the official defaults
  (`weight: 5.0`, `normalize_gt: global`, see
  `src/e05/models/wan22/vggt_omega_loss/defaults.yaml`).
- **Synthetic-data flag.** `data.train.reconstruction_is_synthetic` controls the
  official surface-normal auxiliary depth loss (`apply_aux_loss_to_synthetic_only:
  true`, `weight_normal: 1.0`). The release configuration sets it to `true`. Our
  real-robot runs kept that value, so the normal loss was applied to the smooth
  DA3 pseudo-labels. Set it to `false` if you supervise with raw sensor depth;
  this disables the auxiliary term.

The loss settings used in our runs:

```yaml
data:
  train:
    reconstruction_is_synthetic: true   # false disables the normal auxiliary loss
model:
  lambda_vggt_recon: 0.01
  reconstruction_loss_config:
    backend: official
    official:
      camera: {weight: 0.0}
      point: {weight: 0.0}
```

The release configuration leaves `official: {}`, which keeps the camera and
point weights at 5.0. That setting is correct for RoboTwin, where calibrated
extrinsics exist, but wrong for real data with placeholder extrinsics.

### Depth normalization and scale

With `normalize_gt: global`, the loss normalizes the ground truth in four steps:

1. It unprojects every view's depth with its intrinsics and extrinsics.
2. It re-centres the points on the first camera.
3. It computes the scene scale as the mean point distance.
4. It divides the supervised depth by that scale.

The three views of a sample share one scale factor. With identity placeholders,
every view is treated as if it had the first camera's pose. The scale is
therefore the mean distance of each view's points from its own camera. With
calibrated extrinsics the scale, and so the supervised depth values, would
differ.

Consequences:

- **Predicted depth is not metric.** It is in these normalized scene units. Its
  overall scale varies from frame to frame, while the relative scale between the
  three views of a frame is preserved. To use predicted depth metrically,
  estimate a per-frame scale first, for example from a reference of known size
  or distance. Apply that single factor to all three views.
- **Back-projection needs rescaled intrinsics.** Scale the factory intrinsics to
  the prediction resolution, per axis: 640×480 inputs are predicted at 224×288.
- **Camera and point-map outputs are uncalibrated.** The camera head and the
  point-map head receive no supervision during real-robot post-training. Do not
  interpret the camera poses, fields of view or point maps predicted by a
  real-robot policy as calibrated outputs.
- **DA3 is unreliable at close range.** Its pseudo-labels overestimate depth
  roughly below 15 cm, which is common in wrist views.

## Training hyperparameters

Both platforms used the same hyperparameters:

| Setting | Value |
|---|---|
| GPUs | 8 |
| Batch | `batch_size: 8`, `gradient_accumulation_steps: 2` (effective 128) |
| Learning rate | `learning_rate: 1e-4`, `vggt_learning_rate: 1e-4`, cosine schedule |
| Epochs | `num_epochs: 5` |
| Precision | `mixed_precision: bf16` |
| Regularization | `weight_decay: 0.01`, `max_grad_norm: 1.0` |
| Seed | 42 |
| Optimizer grouping | `optimizer_grouping: legacy` (the release default) |

Initialization:

- **Pretrained.** Load the embodied-pretrained VGGT with `+model_init=embodied_vggt`
  (see [real-robot initialization](TRAIN_EVAL.md#real-robot-initialization)).
  ActionDiT and the bridge are randomly initialized.
- **Without robot-data pretraining.** Use the public VGGT-Omega initialization,
  i.e. omit `+model_init=embodied_vggt`.

## Workflow for your robot

Inputs you need:

- a LeRobot v2.1 dataset with one video feature `observation.images.<camera>`
  per camera and the `observation.state` and `action` vectors;
- metric depth for every frame of every camera, registered to that camera's
  RGB image (same field of view, any resolution with the same aspect ratio);
- each camera's intrinsics, and optionally static extrinsics;
- normalization statistics and text embeddings (both described below).

The template accepts any camera keys, any number of cameras and any state and
action size. The model processes any number of VGGT views and pads views with
different aspect ratios to a common size; our recipe used three 4:3 views.

### 1. Task configuration

Copy the template and fill the `robot` block:

```bash
cp configs/task/real_robot_template.yaml configs/task/my_robot.yaml
```

```yaml
robot:
  dataset_dirs: [/data/my_robot_lerobot]
  camera_keys: [head, wrist_left, wrist_right]   # VGGT order; first = reference view
  num_cameras: 3
  action_dim: 14
  state_dim: 14
  norm_stats: null                                # or a dataset_stats.json path
  text_embedding_cache_dir: /data/my_robot_text_embeds
  reconstruction_cache_roots: [/data/my_robot_recon_cache]
```

The rest of the file references these values, so the camera order, action size
and state size cannot drift apart between the data pipeline and the model. With
a camera count other than three, also edit the list under
`data.train.shape_meta.images` (one entry per camera). The `robot` block exists
because Hydra merges the task file onto `release`: a `???` written directly on
a release key would keep the RoboTwin value instead of raising an error. Every
`???` in `robot` that you leave unfilled fails with
`Missing mandatory value: robot.<name>`.

The template also sets the recipe: depth-only loss, `lambda_vggt_recon: 0.01`,
`reconstruction_is_synthetic: true`, the t+32 target, no VGGT augmentation,
batch 8 × accumulation 2 and 5 epochs. Keys it does not mention keep their
`release` values.

### 2. Normalization statistics

Set `robot.norm_stats: null` for the first run. `RobotVideoDataset` then
computes the statistics from the dataset before training starts and writes
`<output_dir>/dataset_stats.json`. Point later runs at that file. The format is
the one read by `e05.datasets.lerobot.utils.normalizer.load_dataset_stats_from_json`.

### 3. Text embeddings

Training reads one cached T5 embedding per instruction from
`text_embedding_cache_dir`. For each task string in `meta/tasks.jsonl`:

- prompt: `A video recorded from a robot's point of view executing the following instruction: <task>`
- file name: `sha256(prompt).t5_len128.wan22ti2v5b.pt` (hex digest)
- content: `{"context": [128, 4096] bfloat16, "mask": [128] bool}`, the output
  of the Wan2.2-TI2V-5B T5 encoder (tokenizer from Wan2.1-T2V-1.3B, 128 tokens)

The repository has no standalone script for this step. The encoder code is in
`WorldActionRobotWinPolicy._get_text_context`
(`experiments/robotwin/e05_policy/deploy_policy.py`): it loads the encoder with
`_resolve_configs` and `_load_registered_model`, tokenizes with
`HuggingfaceTokenizer`, and with `write_text_embedding_cache=True` saves files
in exactly this format. That class also loads a policy checkpoint, so before
your first training run, reuse those lines in a short script of your own.

### 4. Reconstruction cache

Store each episode's depth as `<depth-dir>/episode_<episode_index:06d>.npz`,
with one array per camera key, shaped `[T, H, W]`, where `T` is the episode
length. Values are meters (float) or integers with a scale, e.g. millimeters
stored as `uint16` with `--depth-scale 0.001`. Zero, negative or non-finite
values mark invalid pixels.

Write the intrinsics as JSON, one entry per camera, in the pixel grid given by
`width` and `height`. Either form is accepted:

```json
{
  "head":       {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
  "wrist_left": {"K": [[390.0, 0, 320.0], [0, 390.0, 240.0], [0, 0, 1]], "width": 640, "height": 480}
}
```

The script rescales intrinsics and depth to the RGB resolution and then to the
VGGT input. Extrinsics are optional. Without `--extrinsics`, every view stores
the identity placeholder, which the manifest records as
`"extrinsics_source": "identity_placeholder"`. With
`--extrinsics cams.json` (`{"head": 3x4 or 4x4 OpenCV world-to-camera}`), the
calibrated poses also change the depth normalization described
[above](#depth-normalization-and-scale).

```bash
source scripts/env.sh
python scripts/build_reconstruction_cache.py \
  --dataset-root /data/my_robot_lerobot \
  --camera-keys head wrist_left wrist_right \
  --depth-dir /data/my_robot_depth --depth-scale 0.001 \
  --intrinsics /data/my_robot_intrinsics.json \
  --output /data/my_robot_recon_cache --workers 8
```

Pass the same camera order as `robot.camera_keys`, and use
`--vggt-image-size` if you changed `vggt_image_size`. The script:

- writes `<output>/<index:08d>.pt` for every value of the dataset's `index`
  column, with `depth` (float16, meters), `valid` (bool), `intrinsics`
  `[V,3,3]` and `extrinsics` `[V,3,4]`;
- by default stores depth at each view's VGGT input size, applying the
  processor's crop, nearest-neighbour resize and padding (640×480 becomes
  224×288). `--keep-native-resolution` keeps the RGB resolution instead and
  leaves resizing to the loader; this requires views the processor neither
  crops nor pads;
- checks that every camera key exists, that each depth array has the episode's
  length and that integer depth comes with `--depth-scale`;
- resumes an interrupted build by skipping existing frames, and refuses to
  resume with different settings;
- writes `manifest.json` (`complete: true`, camera order, units, conventions and
  the dataset fingerprint) only after all frames exist, then re-reads a sample
  of frames with validation and prints the valid-pixel ratio, depth quantiles
  and size.

`--pack-shards` packs the frames into `shards/<index // 1000>.zip`, which the
loader reads directly and which keeps the file count low. `--omit-valid` drops
the mask; the loader then uses `depth > 0`, which is equivalent because invalid
pixels are stored as zero.

**Size.** Each view takes H×W×2 bytes of depth plus H×W bytes of mask per
frame. At 224×288, three views take 387 KB of depth and 194 KB of mask, about
583 KB per frame in total (390 KB with `--omit-valid`); 100,000 frames take
about 58 GB (39 GB). At the native 640×480 a frame takes 2.8 MB.

**Depth stored as video.** LeRobot v2.1 has no standard depth feature, so the
script reads only `.npz`. To convert a 16-bit grayscale depth video (for
example FFV1 `gray16le`), decode it with PyAV:

```python
import av, numpy as np
def read_depth(path):
    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray() for frame in container.decode(video=0)])  # uint16 [T,H,W]
np.savez("depth/episode_000000.npz", head=read_depth("head.mkv"), wrist_left=..., wrist_right=...)
```

**Using the cache.** List one cache per dataset in
`robot.reconstruction_cache_roots`, in the order of `dataset_dirs`. This loader
path checks each manifest (completion, camera order, units and dataset
fingerprint) and maps loader rows to the dataset's `index` column. The older
single-directory key, `data.train.reconstruction_cache_root`, does neither: it
reads files by loader row, which matches the `index` column only for a single
dataset with every episode selected. Frames whose t+32 target falls past the
end of the episode use the last frame and are flagged as padding.

### 5. Train

```bash
export E05_EMBODIED_VGGT_CHECKPOINT=/path/to/e05_embodied_vggt_step_260000.pt
bash scripts/train.sh +task=my_robot +model_init=embodied_vggt
```

`scripts/train.sh` composes the flat `release` configuration, which has no task
entry in its defaults list. Add the task with `+task=...`; a plain `task=...`
fails with "Could not override 'task'". The template already sets the batch,
accumulation and epoch settings. With fewer than 8 GPUs, raise
`gradient_accumulation_steps` to keep the effective batch of 128.

### 6. Inference

`scripts/infer.py` reads the camera keys, their order, the state size and the
action size from the training configuration. Pass either your task file or the
run's saved `config.yaml`:

```bash
python scripts/infer.py \
  --config configs/task/my_robot.yaml \
  --checkpoint runs/<run>/checkpoints/weights/step_XXXXXX.pt \
  --stats runs/<run>/dataset_stats.json \
  --input observation.npz --output prediction.npz
```

The observation NPZ contains one `uint8[H, W, 3]` RGB image per camera key,
named exactly as in `robot.camera_keys`, plus `state` (`float32[state_dim]`)
and `instruction` (a scalar string). The output `action` is `[32, action_dim]`
in dataset units. Without `--stats`, the script looks for `dataset_stats.json`
beside the checkpoint, then beside `--config`, then uses
`data.train.pretrained_norm_stats`. Add `--cached-text-only` to read
instruction embeddings from `text_embedding_cache_dir` instead of loading the
T5 encoder. One call predicts one chunk with seed 42 and 10 denoising steps;
for closed-loop control, call `load_policy` and `predict` from your own
control loop.

We checked this path against our deployment code: for an AgileX Piper (14-D)
and a Zerith H1 (23-D) policy, on a real dataset frame, it returns the same
action chunk bit for bit.
