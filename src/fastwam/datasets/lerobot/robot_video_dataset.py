import hashlib
from fastwam.datasets.cache_io import cache_exists, load_cache
import os
from pathlib import Path
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .vggt_teacher_cache import (
    VGGTTeacherFeatureCache,
    build_cache_descriptor,
)
from .vggt_depth_cache import (
    VGGTDepthTargetCache,
    build_depth_cache_descriptor,
    build_true_depth_cache_descriptor,
)
from .vggt_reconstruction_cache import VGGTReconstructionTargetCache, MultiDatasetReconstructionCache
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        dataset_name=None,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        episode_ranges=None,
        use_precomputed_vggt_teacher_features: bool = False,
        vggt_teacher_feature_cache_root=None,
        vggt_teacher_checkpoint_path=None,
        vggt_teacher_feature_layer: int = 23,
        vggt_teacher_feature_dim: int = 2048,
        depth_cache_root=None,
        depth_teacher_checkpoint_path=None,
        depth_target_source: str = "vggt_teacher",
        depth_target_height: int | None = None,
        depth_target_width: int | None = None,
        depth_future_offsets=None,
        depth_patch_size: int = 16,
        use_precomputed_vggt_geometry_latents: bool = False,
        vggt_geometry_latent_cache_root=None,
        reconstruction_targets: bool = False,
        reconstruction_is_synthetic: bool = False,
        reconstruction_cache_root=None,
        reconstruction_cache_roots=None,
        reconstruction_allow_missing: bool = False,
        reconstruction_future_frame_index: int | None = None,
        reconstruction_target: str = "future",
        vggt_official_augmentation=None,
        libero_image_convention="legacy",
    ):
        if reconstruction_cache_root is not None and reconstruction_cache_roots is not None:
            raise ValueError("Specify only one of reconstruction_cache_root/reconstruction_cache_roots.")
        if reconstruction_cache_roots is not None:
            if len(reconstruction_cache_roots) != len(dataset_dirs):
                raise ValueError("reconstruction_cache_roots must match dataset_dirs one-to-one.")
            if reconstruction_allow_missing:
                raise ValueError("Per-dataset reconstruction caches require allow_missing=false.")
        if libero_image_convention not in {"legacy", "opencv"}:
            raise ValueError("libero_image_convention must be legacy or opencv.")
        if reconstruction_target not in {"current", "future"}:
            raise ValueError("reconstruction_target must be 'current' or 'future'.")
        self.reconstruction_target = reconstruction_target
        self.vggt_official_augmentation = None
        if vggt_official_augmentation is not None and vggt_official_augmentation.get("enabled", True):
            from fastwam.aug.official import OfficialVGGTAugmentation
            if not reconstruction_targets or reconstruction_allow_missing:
                raise ValueError("Official augmentation requires real reconstruction targets (allow_missing=false).")
            if use_precomputed_vggt_teacher_features or use_precomputed_vggt_geometry_latents or depth_future_offsets is not None:
                raise ValueError("Official augmentation cannot use geometry/teacher caches in transformed coordinates.")
            self.vggt_official_augmentation = OfficialVGGTAugmentation(vggt_official_augmentation)
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            dataset_name=dataset_name,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            episode_ranges=episode_ranges,
        )
        self.dataset_name = str(dataset_name) if dataset_name else None
        if self.vggt_official_augmentation is not None:
            self.lerobot_dataset.strict_sample_loading = True
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.global_sample_stride = int(global_sample_stride)
        self.use_precomputed_vggt_teacher_features = bool(
            use_precomputed_vggt_teacher_features
        )
        self.vggt_teacher_feature_cache = None
        self.use_precomputed_vggt_geometry_latents = bool(use_precomputed_vggt_geometry_latents)
        self.vggt_geometry_latent_cache_root = vggt_geometry_latent_cache_root
        self.vggt_depth_target_cache = None
        self.reconstruction_target_cache = None
        self.reconstruction_targets = bool(reconstruction_targets)
        self.reconstruction_is_synthetic = bool(reconstruction_is_synthetic)
        self.reconstruction_future_frame_index = reconstruction_future_frame_index
        if self.reconstruction_targets:
            if processor.vggt_camera_keys is None:
                raise ValueError("Reconstruction targets require `vggt_camera_keys`.")
            if reconstruction_cache_roots is not None:
                self.reconstruction_target_cache = MultiDatasetReconstructionCache(
                    reconstruction_cache_roots, self.lerobot_dataset.multi_dataset._datasets,
                    processor.vggt_camera_keys,
                )
                self.lerobot_dataset.strict_sample_loading = True
            else:
                self.reconstruction_target_cache = VGGTReconstructionTargetCache(
                    reconstruction_cache_root,
                    processor.vggt_camera_keys,
                    allow_missing=reconstruction_allow_missing,
                )
        self.depth_target_source = str(depth_target_source)
        if self.depth_target_source not in {"vggt_teacher", "robotwin_true_depth"}:
            raise ValueError(
                "`depth_target_source` must be `vggt_teacher` or "
                f"`robotwin_true_depth`, got {self.depth_target_source!r}."
            )
        self.depth_future_offsets = (
            None
            if depth_future_offsets is None
            else tuple(int(value) for value in depth_future_offsets)
        )
        self.depth_patch_size = int(depth_patch_size)
        if self.depth_future_offsets is not None:
            if (
                not self.depth_future_offsets
                or any(value <= 0 for value in self.depth_future_offsets)
                or any(
                    left >= right
                    for left, right in zip(
                        self.depth_future_offsets, self.depth_future_offsets[1:]
                    )
                )
            ):
                raise ValueError(
                    "`depth_future_offsets` must be strictly increasing positive integers."
                )
            if self.depth_patch_size <= 0:
                raise ValueError("`depth_patch_size` must be positive.")
            if self.use_precomputed_vggt_teacher_features:
                raise ValueError(
                    "The legacy VGGT feature cache and depth target cache cannot be enabled together."
                )

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if self.vggt_official_augmentation is not None:
                previous = processor.vggt_augmentation
                if previous is not None and getattr(previous, "enabled", True):
                    raise ValueError("Official augmentation and processor.vggt_augmentation cannot both be enabled.")
                processor.defer_vggt_official_preprocessing = True
            if self.use_precomputed_vggt_teacher_features:
                processor.emit_vggt_future_images = False
                if vggt_teacher_feature_cache_root is None:
                    raise ValueError(
                        "`vggt_teacher_feature_cache_root` is required when "
                        "precomputed VGGT teacher features are enabled."
                    )
                if vggt_teacher_checkpoint_path is None:
                    raise ValueError(
                        "`vggt_teacher_checkpoint_path` is required to validate "
                        "the VGGT teacher cache fingerprint."
                    )
                if processor.vggt_camera_keys is None:
                    raise ValueError("VGGT teacher caching requires `vggt_camera_keys`.")
                if processor.vggt_future_frame_index is None:
                    raise ValueError(
                        "VGGT teacher caching requires `vggt_future_frame_index`."
                    )
                descriptor = build_cache_descriptor(
                    self.lerobot_dataset,
                    dataset_name=self.dataset_name,
                    camera_keys=processor.vggt_camera_keys,
                    image_size=processor.vggt_image_size,
                    teacher_checkpoint_path=vggt_teacher_checkpoint_path,
                    feature_layer=vggt_teacher_feature_layer,
                    feature_dim=vggt_teacher_feature_dim,
                )
                episode_ends = self.lerobot_dataset.episode_data_index["to"].tolist()
                self.vggt_teacher_feature_cache = VGGTTeacherFeatureCache(
                    cache_root=vggt_teacher_feature_cache_root,
                    descriptor=descriptor,
                    episode_ends=episode_ends,
                )
            if self.depth_future_offsets is not None:
                if depth_cache_root is None:
                    raise ValueError(
                        "`depth_cache_root` is required when depth targets are enabled."
                    )
                if processor.vggt_camera_keys is None:
                    raise ValueError("Depth caching requires `vggt_camera_keys`.")
                if self.depth_target_source == "vggt_teacher":
                    if depth_teacher_checkpoint_path is None:
                        raise ValueError(
                            "`depth_teacher_checkpoint_path` is required to validate "
                            "a VGGT teacher depth cache."
                        )
                    descriptor = build_depth_cache_descriptor(
                        self.lerobot_dataset,
                        camera_keys=processor.vggt_camera_keys,
                        image_size=processor.vggt_image_size,
                        teacher_checkpoint_path=depth_teacher_checkpoint_path,
                    )
                else:
                    if depth_target_height is None or depth_target_width is None:
                        raise ValueError(
                            "True depth caching requires `depth_target_height` and "
                            "`depth_target_width`."
                        )
                    descriptor = build_true_depth_cache_descriptor(
                        self.lerobot_dataset,
                        dataset_name=self.dataset_name,
                        camera_keys=processor.vggt_camera_keys,
                        target_height=int(depth_target_height),
                        target_width=int(depth_target_width),
                    )
                episode_ends = self.lerobot_dataset.episode_data_index["to"].tolist()
                self.vggt_depth_target_cache = VGGTDepthTargetCache(
                    cache_root=depth_cache_root,
                    descriptor=descriptor,
                    episode_ends=episode_ends,
                )
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        elif self.use_precomputed_vggt_teacher_features:
            raise ValueError(
                "Precomputed VGGT teacher features require a FastWAM processor."
            )
        elif self.depth_future_offsets is not None:
            raise ValueError("VGGT depth targets require a FastWAM processor.")
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        from fastwam.utils.samplers import AugmentedIndex
        aug_index = idx if isinstance(idx, AugmentedIndex) else None
        if aug_index is not None:
            idx = aug_index.index
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        teacher_cache = getattr(self, "vggt_teacher_feature_cache", None)
        if teacher_cache is not None:
            processor = self.lerobot_dataset.processor
            g0, g32, future_is_pad = teacher_cache.get_pair(
                int(sample["idx"]),
                future_frame_index=int(processor.vggt_future_frame_index),
                global_sample_stride=self.global_sample_stride,
            )
            cached_pad = torch.tensor(future_is_pad, dtype=torch.bool)
            sample_pad = sample.get("vggt_future_is_pad")
            if sample_pad is not None and bool(sample_pad.item()) != future_is_pad:
                raise RuntimeError(
                    "Cached VGGT future index disagrees with dataset padding for "
                    f"sample {sample['idx']}."
                )
            sample["vggt_teacher_g0"] = g0.flatten(0, 1)
            sample["vggt_teacher_g32"] = g32.flatten(0, 1)
            sample["vggt_future_is_pad"] = cached_pad

        if getattr(self, "use_precomputed_vggt_geometry_latents", False):
            if getattr(self, "vggt_geometry_latent_cache_root", None) is None:
                raise ValueError("vggt_geometry_latent_cache_root is required for latent cache.")
            latent_path = Path(self.vggt_geometry_latent_cache_root) / f"{int(sample['idx']):08d}.pt"
            if not latent_path.exists():
                raise FileNotFoundError(f"VGGT latent cache entry not found: {latent_path}")
            payload = torch.load(latent_path, map_location="cpu", weights_only=True)
            sample["vggt_latent_future"] = payload["latent"] if isinstance(payload, dict) else payload
            if isinstance(payload, dict) and "future_is_pad" in payload:
                sample["vggt_future_is_pad"] = payload["future_is_pad"]

        depth_cache = getattr(self, "vggt_depth_target_cache", None)
        if depth_cache is not None:
            target, weight, _, future_is_pad = depth_cache.get_future(
                int(sample["idx"]),
                future_offsets=self.depth_future_offsets,
                global_sample_stride=self.global_sample_stride,
            )
            target = self._patchify_depth(target, self.depth_patch_size)
            weight = self._patchify_depth(weight, self.depth_patch_size)
            sample["depth_target"] = target
            sample["depth_weight"] = weight
            sample["depth_future_is_pad"] = future_is_pad

        recon_cache = getattr(self, "reconstruction_target_cache", None)
        if recon_cache is not None:
            sample["vggt_recon_is_synthetic"] = torch.tensor(
                getattr(self, "reconstruction_is_synthetic", False), dtype=torch.bool)
            current_index = int(sample["idx"])
            if getattr(self, "reconstruction_target", "future") == "current":
                targets = {"current": recon_cache.get(current_index)}
            else:
                offset = int(self.reconstruction_future_frame_index or (self.num_frames - 1))
                ends = self.lerobot_dataset.episode_data_index["to"]
                episode_id = int(torch.searchsorted(ends, torch.tensor(current_index), right=True))
                episode_end = int(ends[episode_id].item())
                requested = current_index + offset * self.global_sample_stride
                future_is_pad = requested >= episode_end
                future_index = min(requested, episode_end - 1)
                pair = recon_cache.get_pair(current_index, future_index=future_index, future_is_pad=future_is_pad)
                targets = {"current": pair["current"], "future": pair["future"]}
                sample["vggt_recon_future_is_pad"] = torch.tensor(future_is_pad, dtype=torch.bool)
            if "_vggt_raw_images" in sample:
                from fastwam.aug.official import augmentation_seed
                augmentation = self.vggt_official_augmentation
                sample_seed = aug_index.sample_seed if aug_index else augmentation_seed(augmentation.seed, idx, 'sample')
                batch_seed = aug_index.batch_seed if aug_index else augmentation_seed(augmentation.seed, idx, 'batch')
                sample["vggt_images"], targets, augmentation_trace = augmentation(
                    sample.pop("_vggt_raw_images"), targets, sample_seed=sample_seed, batch_seed=batch_seed)
                if os.environ.get("FASTWAM_SMOKE_AUDIT_DIR"):
                    sample["vggt_augmentation_trace"] = augmentation_trace
                augmented = True
            else:
                augmented = False
            if "vggt_images" not in sample:
                raise ValueError("Reconstruction targets require processor output `vggt_images`.")
            target_h, target_w = sample["vggt_images"].shape[-2:]
            def resize_target(target):
                if augmented:
                    return target["depth"], target["valid"], target["intrinsics"], target["extrinsics"]
                depth = target["depth"].float()
                valid = target["valid"].bool()
                intrinsics = target["intrinsics"].float().clone()
                source_h, source_w = depth.shape[-2:]
                depth = torch.nn.functional.interpolate(depth[:, None], size=(target_h, target_w), mode="nearest-exact")[:, 0]
                valid = torch.nn.functional.interpolate(valid[:, None].float(), size=(target_h, target_w), mode="nearest-exact")[:, 0] > 0.5
                intrinsics[:, 0, :] *= target_w / source_w
                intrinsics[:, 1, :] *= target_h / source_h
                return depth, valid, intrinsics, target["extrinsics"].float()
            for name, target in targets.items():
                depth, valid, intrinsics, extrinsics = resize_target(target)
                sample[f"vggt_recon_{name}_depth"] = depth
                sample[f"vggt_recon_{name}_depth_valid"] = valid
                sample[f"vggt_recon_{name}_intrinsics"] = intrinsics
                sample[f"vggt_recon_{name}_extrinsics"] = extrinsics
            # Keep the historical single-time names for ActionVGGTFastWAM.
            # The new model consumes the explicit current/future names above,
            # while older configs still request the current-frame target.
            sample["vggt_recon_depth"] = sample["vggt_recon_current_depth"]
            sample["vggt_recon_depth_valid"] = sample["vggt_recon_current_depth_valid"]
            sample["vggt_recon_intrinsics"] = sample["vggt_recon_current_intrinsics"]
            sample["vggt_recon_extrinsics"] = sample["vggt_recon_current_extrinsics"]
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        language_context_mask = context_mask.clone()
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "video": video,
            **({"vggt_augmentation_trace": sample["vggt_augmentation_trace"]} if "vggt_augmentation_trace" in sample else {}),
            **({"vggt_images": sample["vggt_images"]} if "vggt_images" in sample else {}),
            **(
                {"vggt_future_images": sample["vggt_future_images"]}
                if "vggt_future_images" in sample
                else {}
            ),
            **(
                {"vggt_future_is_pad": sample["vggt_future_is_pad"]}
                if "vggt_future_is_pad" in sample
                else {}
            ),
            **(
                {
                    "vggt_teacher_g0": sample["vggt_teacher_g0"],
                    "vggt_teacher_g32": sample["vggt_teacher_g32"],
                }
                if "vggt_teacher_g0" in sample
                else {}
            ),
            **(
                {"vggt_latent_future": sample["vggt_latent_future"]}
                if "vggt_latent_future" in sample else {}
            ),
            **(
                {
                    "depth_target": sample["depth_target"],
                    "depth_weight": sample["depth_weight"],
                    "depth_future_is_pad": sample["depth_future_is_pad"],
                }
                if "depth_target" in sample
                else {}
            ),
            **(
                {
                    key: sample[key]
                    for key in (
                        "vggt_recon_current_depth", "vggt_recon_current_depth_valid",
                        "vggt_recon_current_intrinsics", "vggt_recon_current_extrinsics",
                        "vggt_recon_future_depth", "vggt_recon_future_depth_valid",
                        "vggt_recon_future_intrinsics", "vggt_recon_future_extrinsics",
                        "vggt_recon_future_is_pad", "vggt_recon_is_synthetic",
                    )
                    if key in sample
                }
                if "vggt_recon_current_depth" in sample
                else {}
            ),
            **(
                {
                    key: sample[key]
                    for key in (
                        "vggt_recon_depth", "vggt_recon_depth_valid",
                        "vggt_recon_intrinsics", "vggt_recon_extrinsics",
                    )
                }
                if "vggt_recon_depth" in sample
                else {}
            ),
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "language_context_mask": language_context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        return data

    @staticmethod
    def _patchify_depth(depth: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Convert ``[K,V,H,W]`` maps to ``[K,V,N,P^2]`` raster patches."""
        if depth.ndim != 4:
            raise ValueError(
                f"Depth cache tensors must be [K,V,H,W], got {tuple(depth.shape)}."
            )
        if patch_size <= 0:
            raise ValueError("`patch_size` must be positive.")
        k, views, height, width = depth.shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"Depth shape {(height, width)} is not divisible by patch_size={patch_size}."
            )
        patches = depth.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size
        )
        return patches.contiguous().view(
            k,
            views,
            (height // patch_size) * (width // patch_size),
            patch_size * patch_size,
        )

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not cache_exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Install the matching E05 processed text-cache shards."
            )
        payload = load_cache(cache_path)
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        if (getattr(self, "vggt_official_augmentation", None) is not None or
                getattr(getattr(self, "lerobot_dataset", None), "strict_sample_loading", False)):
            return self._get(idx)
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
