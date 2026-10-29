# Real-robot post-training recipe

This page records the configuration used for the AgileX Piper and Zerith H1
experiments. The real-robot datasets and task configurations are not released;
use this recipe to write a task configuration for your own robot data. The
RoboTwin defaults in `configs/task/e05.yaml` are **not** a real-robot recipe.
In particular, their camera and point-map losses must be disabled for real data,
as explained below.

## Data

| Item | Setting |
|---|---|
| Format | LeRobot v2.1, 30 fps |
| Cameras | Three RGB views, in the order `cam_high`, `cam_left_wrist`, `cam_right_wrist`, recorded at 640×480 |
| State / action | AgileX Piper: 14-D joints and grippers. Zerith H1: 23-D. Each platform uses its own normalization statistics. |
| VGGT input | `vggt_image_size: 256`, so 4:3 frames become 224×288 |
| Action chunk | 32 steps |

## Geometry supervision: depth only

Real-robot post-training keeps the future-geometry objective, with the target
frame 32 steps ahead (the endpoint of the action chunk, the same as RoboTwin). The
supervision is limited to depth:

- **Depth source.** Depth sensors are not used. Each camera view is labeled with
  metric depth (meters) estimated by Depth Anything 3 (`DA3METRIC-LARGE`) on the
  640×480 frames.
- **Intrinsics.** These are the factory calibration read from each camera's SDK.
- **Extrinsics.** None are available. The cache format requires an extrinsics
  tensor, so every view stores the identity `[I | 0]` placeholder. These
  placeholders do not enter any loss, because the camera and point-map losses are
  disabled.
- **Loss.** `backend: official` with `lambda_vggt_recon: 0.01`. The official
  backend reads only the `official` block. It ignores the legacy `lambda_camera`,
  `lambda_depth`, `lambda_point`, `use_conf_loss` and `gradient_weight` keys, so
  setting those has no effect. Disable the camera and point-map terms by setting
  their official weights to zero. The depth term keeps the official defaults
  (`weight: 5.0`, see `src/e05/models/wan22/vggt_omega_loss/defaults.yaml`):

  ```yaml
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

Consequences:

- During real-robot post-training, the camera head and the point-map head
  receive no supervision. Do not interpret camera poses, fields of view or point
  maps predicted by a real-robot policy as calibrated outputs. If you need 3D
  points, back-project the predicted depth with the factory intrinsics.
- DA3 pseudo-labels are least reliable at very close range (roughly below 15 cm,
  which is common in wrist views). There they overestimate depth.

## Reconstruction cache

`VGGTReconstructionTargetCache` reads targets from a per-frame file store.
Write one entry per global dataset frame:

- `<root>/<global_index:08d>.pt`, a dict containing:
  - `depth`: float tensor `[V, H, W]` in meters
  - `valid`: bool tensor `[V, H, W]`; optional, defaults to `depth > 0`
  - `intrinsics`: `[V, 3, 3]` in the pixel frame of the stored depth map
  - `extrinsics`: `[V, 3, 4]` in OpenCV world-to-camera form; here, the identity
    placeholder

  Views follow the model's VGGT camera order. The loader resizes depth to the
  VGGT input and rescales the intrinsics accordingly.
- `<root>/manifest.json`, containing:
  - `"complete": true`
  - `"camera_keys"`: the VGGT camera order
  - `"depth_units": "meters"`
  - `"extrinsics": "opencv_world_to_camera"`
  - `"image_convention": "opencv"`
  - `"dataset_fingerprint"`: computed by
    `e05.datasets.lerobot.vggt_reconstruction_cache.reconstruction_dataset_fingerprint(<dataset root>)`

Point `data.train.reconstruction_cache_root` at this directory and keep
`reconstruction_future_frame_index: 32`. Frames whose t+32 target falls past the
end of the episode use the last frame and are flagged as padding.

## Training

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
  i.e. the defaults.

Example command. Put your real-robot task configuration in `configs/task/`
with `# @package _global_` and the depth-only loss block shown above.
`scripts/train.sh` composes the flat `release` configuration, which has no task
entry in its defaults list. Add the task with `+task=...`; a plain `task=...`
fails with "Could not override 'task'".

```bash
export E05_EMBODIED_VGGT_CHECKPOINT=/path/to/e05_embodied_vggt_step_260000.pt
bash scripts/train.sh +task=<your_real_robot_task> +model_init=embodied_vggt \
  batch_size=8 gradient_accumulation_steps=2 num_epochs=5
```

Your task configuration must supply the dataset paths, normalization statistics,
text embeddings, camera order, state/action dimensions and the reconstruction
cache described above.
