# E05 processed RoboTwin data

- Robot benchmark tasks: 50; episodes: 2,500 (50/task).
- Frames: 549,787; LeRobot format: v2.1; robot: aloha-agilex.
- Metadata FPS: 15. This is the processed dataset's timing metadata, not a claim
  about the simulator's internal physics/control rate.
- Language table: 158,984 entries. LeRobot `total_tasks` counts language entries,
  not the 50 robot benchmark tasks.
- Cameras, in order: cam_high, cam_left_wrist, cam_right_wrist; RGB 240×320.
- State/action: 14 joint/gripper components in the existing dataset order.
  Preserve left-arm joints/gripper followed by right-arm joints/gripper as consumed
  by RoboTwin's joint-action interface. Do not interpret this as end-effector delta.
- Source modality metadata records state[t]=joint[t], action[t]=joint[t+1].
- Geometry cache: depth in metres, valid pixels, intrinsics [3,3,3], and OpenCV
  world-to-camera extrinsics [3,3,4]. Future target offset: 32 frames; targets
  beyond an episode are excluded by the existing padding mask.

All 2,500 episodes are used for training; there is no internal held-out loss split.
Normalization uses the exact source run's statistics, verified equal to the named
dataset statistics file. Geometric diagnostics on these frames are training-set
diagnostics, not held-out generalization measurements.

The code expects data that have already been processed. The dataset and caches
are external resources and are not included in the Git repository.
RGB/Parquet/meta files retain their content
and indexing. Cache shards contain original `.pt` files compressed losslessly;
ZIP readers return the same bytes. There is no raw-data reconstruction pipeline.
Only language entries referenced by this dataset are included; unrelated experimental
text caches and incomplete temporary files are excluded. Novel inference instructions
use the public text encoder online, without modifying the published cache.

The reference layout contains RGB/action files and metadata in the LeRobot
directory, geometry cache shards by global-frame ranges of 1,000, and text cache
shards by SHA256 prefix. Each cache manifest records sizes and hashes; ZIP shards
retain their companion JSON entry manifests.
`scripts/validate_data.py --full` verifies complete coverage and cache archive
hashes after the data are placed in the configured directory. See the
[resource layout](../README.md) for the expected paths.

`scripts/install_data.py` remains an optional utility for an existing packaged
processed dataset. Such packages are not a prerequisite for using the code and
are not distributed by the code repository.

Provenance: processed from the locally supplied RobotTwin2.0-Depth
`aloha-agilex_clean_50` dataset. This is a project-specific derivative, not the
upstream official dataset release. Upstream data, language and asset terms must
be retained; the source code MIT license does not relicense these materials.

## Public download

[Processed RoboTwin data](https://huggingface.co/datasets/X-EraAI/e05-robotwin-data) is published as existing lossless shards.
Use the [download and installation commands](../README.md#download-published-resources).
The pinned revision and validation status are recorded in [release evidence](validation/huggingface_release.json).
