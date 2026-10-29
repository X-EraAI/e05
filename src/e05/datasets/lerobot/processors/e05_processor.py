from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Literal

import torch
import numpy as np
from copy import deepcopy
from ..utils.normalizer import LinearNormalizer, NormMode
from e05.utils.pytorch_utils import dict_apply
from e05.utils.logging_config import get_logger
from .base_processor import BaseProcessor

logger = get_logger(__name__)


def prepare_vggt_images(
    images: Dict[str, torch.Tensor],
    camera_keys: List[str] | tuple[str, ...],
    image_size: int,
    frame_index: int = 0,
) -> torch.Tensor:
    """Prepare one temporal index from every RGB view as ``[V,3,H,W]``.

    ``image_size`` is the balanced preprocessing reference resolution, not a
    fixed output edge length. VGGT-Omega chooses patch-aligned ``H`` and ``W``
    from each view's aspect ratio and pads mixed shapes to a common size. The
    dataset collator later adds the batch dimension.
    """
    try:
        from vggt_omega.utils.load_fn import load_and_preprocess_images
    except ImportError as exc:
        raise ImportError(
            "VGGT preprocessing requires the optional `vggt_omega` package."
        ) from exc
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise TypeError(f"`frame_index` must be an integer, got {type(frame_index)}.")
    selected = []
    for key in camera_keys:
        if key not in images:
            raise KeyError(f"VGGT camera key is missing from image sample: {key!r}.")
        sequence = images[key]
        if sequence.ndim != 4:
            raise ValueError(
                f"VGGT camera `{key}` must be [T,C,H,W], got {tuple(sequence.shape)}."
            )
        if not 0 <= frame_index < int(sequence.shape[0]):
            raise IndexError(
                f"VGGT frame index {frame_index} is out of range for `{key}` "
                f"with T={sequence.shape[0]}."
            )
        selected.append(sequence[frame_index])
    return load_and_preprocess_images(
        selected,
        mode="balanced",
        image_resolution=image_size,
        patch_size=16,
    )


def prepare_vggt_reconstruction_targets(
    images,
    camera_keys,
    depth,
    valid,
    intrinsics,
    extrinsics,
    image_size,
):
    """Apply VGGT's balanced crop/resize/pad geometry to depth and K."""
    from vggt_omega.utils.load_fn import _balanced_target_shape

    depth = torch.as_tensor(depth, dtype=torch.float32)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32).clone()
    extrinsics = torch.as_tensor(extrinsics, dtype=torch.float32)
    if depth.ndim == 4:  # [V,T,H,W], use observation frame zero
        depth, valid = depth[:, 0], valid[:, 0]
    num_views = len(camera_keys)
    if depth.ndim != 3 or depth.shape[0] != num_views:
        raise ValueError(f"Reconstruction depth must be [V,H,W] with V={num_views}, got {tuple(depth.shape)}")
    if valid.shape != depth.shape or intrinsics.shape != (num_views, 3, 3) or extrinsics.shape != (num_views, 3, 4):
        raise ValueError("Reconstruction depth/valid/intrinsics/extrinsics shapes are inconsistent.")

    resized = []
    resized_valid = []
    target_shapes = []
    geometry = []
    for view_index, key in enumerate(camera_keys):
        rgb = images[key][0]
        source_h, source_w = int(rgb.shape[-2]), int(rgb.shape[-1])
        if tuple(depth[view_index].shape) != (source_h, source_w):
            raise ValueError(
                f"RGB/depth native shape mismatch for {key}: RGB={(source_h, source_w)} "
                f"depth={tuple(depth[view_index].shape)}"
            )
        crop_left = crop_top = 0
        crop_h, crop_w = source_h, source_w
        aspect = source_h / max(source_w, 1)
        if aspect < 0.5:
            crop_w = min(source_w, max(1, int(round(source_h / 0.5))))
            crop_left = max((source_w - crop_w) // 2, 0)
        elif aspect > 2.0:
            crop_h = min(source_h, max(1, int(round(source_w * 2.0))))
            crop_top = max((source_h - crop_h) // 2, 0)
        target_h, target_w = _balanced_target_shape(crop_h / max(crop_w, 1), image_size, 16)
        d = depth[view_index, crop_top:crop_top + crop_h, crop_left:crop_left + crop_w]
        m = valid[view_index, crop_top:crop_top + crop_h, crop_left:crop_left + crop_w]
        d = torch.nn.functional.interpolate(d[None, None], size=(target_h, target_w), mode="nearest-exact")[0, 0]
        m = torch.nn.functional.interpolate(m[None, None].float(), size=(target_h, target_w), mode="nearest-exact")[0, 0] > 0.5
        resized.append(d)
        resized_valid.append(m)
        target_shapes.append((target_h, target_w))
        geometry.append((crop_left, crop_top, target_w / crop_w, target_h / crop_h))

    common_h = max(shape[0] for shape in target_shapes)
    common_w = max(shape[1] for shape in target_shapes)
    output_depth, output_valid = [], []
    for view_index, ((target_h, target_w), (crop_left, crop_top, sx, sy)) in enumerate(zip(target_shapes, geometry)):
        pad_top = (common_h - target_h) // 2
        pad_bottom = common_h - target_h - pad_top
        pad_left = (common_w - target_w) // 2
        pad_right = common_w - target_w - pad_left
        output_depth.append(torch.nn.functional.pad(resized[view_index], (pad_left, pad_right, pad_top, pad_bottom), value=0.0))
        output_valid.append(torch.nn.functional.pad(resized_valid[view_index], (pad_left, pad_right, pad_top, pad_bottom), value=False))
        intrinsics[view_index, 0, 0] *= sx
        intrinsics[view_index, 1, 1] *= sy
        intrinsics[view_index, 0, 2] = (intrinsics[view_index, 0, 2] - crop_left) * sx + pad_left
        intrinsics[view_index, 1, 2] = (intrinsics[view_index, 1, 2] - crop_top) * sy + pad_top
    return {
        "depth": torch.stack(output_depth),
        "valid": torch.stack(output_valid),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
    }


def prepare_vggt_clip_images(images, camera_keys, image_size, frame_indices):
    """Prepare a temporal clip as ``[T,V,3,H,W]`` using the shared preprocess."""
    return torch.stack([
        prepare_vggt_images(images, camera_keys, image_size, frame_index=int(index))
        for index in frame_indices
    ], dim=0)


class E05Processor(BaseProcessor):
    def __init__(
        self,
        # keys
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int, 
        action_output_dim: int,
        proprio_output_dim: int,

        action_state_transforms: Optional[List[Any]], 

        # action & state normalization
        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        norm_exception_mode: Dict[str, Dict[str, NormMode]], 

        action_state_merger, 

        # image transform
        train_transforms: Dict[str, List[Any]] | None,
        val_transforms: Dict[str, List[Any]] | None, 

        # instruction transform
        drop_high_level_prob: float = 1.0,
        use_zh_instruction: bool = False,

        tokenizer: Optional[Any] = None,
        delta_action_dim_mask: Optional[Dict[str, List[bool]]] = None,
        action_representation: str = "native",
        vggt_camera_keys: Optional[List[str]] = None,
        vggt_image_size: int = 256,
        vggt_future_frame_index: Optional[int] = None,
        emit_vggt_future_images: bool = True,
        emit_vggt_future_clip: bool = False,
        vggt_augmentation=None,
    ):
        self.shape_meta = shape_meta
        self.num_obs_steps = num_obs_steps
        self.num_output_cameras = num_output_cameras
        self.action_output_dim = action_output_dim
        self.proprio_output_dim = proprio_output_dim
        if action_representation not in {"native", "robotwin_ee_delta_chunk_local_rot6d"}:
            raise ValueError(f"Unsupported action representation: {action_representation}")
        self.action_representation = action_representation

        self.drop_high_level_prob = drop_high_level_prob
        self.use_zh_instruction = use_zh_instruction

        # image
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        self._is_train = None

        self.action_state_transforms = action_state_transforms
        self.action_state_merger = action_state_merger
        self.action_state_merger.set_shape_meta(self.shape_meta)

        self.use_stepwise_action_norm = use_stepwise_action_norm
        self.norm_default_mode = norm_default_mode
        self.norm_exception_mode = norm_exception_mode
        self._normalizer = None

        self.tokenizer = tokenizer
        self.vggt_camera_keys = vggt_camera_keys
        self.vggt_image_size = vggt_image_size
        # Old configs omit this field. Hydra normally instantiates the nested
        # callable; also accept a config when the processor is built directly.
        if vggt_augmentation is not None and not callable(vggt_augmentation):
            from hydra.utils import instantiate

            vggt_augmentation = instantiate(vggt_augmentation)
        if vggt_augmentation is not None and not callable(vggt_augmentation):
            raise TypeError("vggt_augmentation must be a callable or Hydra config.")
        self.vggt_augmentation = vggt_augmentation
        self.vggt_future_frame_index = (
            None
            if vggt_future_frame_index is None
            else int(vggt_future_frame_index)
        )
        self.emit_vggt_future_images = bool(emit_vggt_future_images)
        self.emit_vggt_future_clip = bool(emit_vggt_future_clip)
        if self.vggt_future_frame_index is not None and not (
            0 <= self.vggt_future_frame_index < self.num_obs_steps
        ):
            raise ValueError(
                "`vggt_future_frame_index` must be in "
                f"[0,{self.num_obs_steps}), got {self.vggt_future_frame_index}."
            )
        if delta_action_dim_mask is None:
            self.delta_action_dim_mask = None
        else:
            action_meta = self.shape_meta["action"]
            expected_keys = [m["key"] for m in action_meta]
            provided_keys = list(delta_action_dim_mask.keys())
            if set(provided_keys) != set(expected_keys):
                raise ValueError(
                    f"`delta_action_dim_mask` keys mismatch. Expected {expected_keys}, got {provided_keys}."
                )

            self.delta_action_dim_mask = {}
            for meta in action_meta:
                key = meta["key"]
                expected_dim = meta["shape"]
                mask = delta_action_dim_mask[key]
                if len(mask) != expected_dim:
                    raise ValueError(
                        f"`delta_action_dim_mask[{key}]` length must be {expected_dim}, got {len(mask)}."
                    )
                self.delta_action_dim_mask[key] = torch.as_tensor(mask, dtype=torch.bool)

    @property
    def is_train(self):
        if self._is_train is None:
            raise ValueError("is_train has not been set. Please call train() and eval() first.")
        return self._is_train

    @property
    def normalizer(self) -> LinearNormalizer:
        if self._normalizer is None:
            raise ValueError("normalizer has not been set. Please call set_normalizer_from_stats() first.")
        return self._normalizer

    def train(self):
        self._is_train = True
        return self

    def eval(self):
        self._is_train = False
        return self

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any] = None):
        self._normalizer = LinearNormalizer(
            use_stepwise_action_norm=self.use_stepwise_action_norm,
            shape_meta=self.shape_meta,
            default_mode=self.norm_default_mode,
            exception_mode=self.norm_exception_mode,
            stats=dataset_stats,
        )

    def augment_instruction(self, data: Dict[str, str] | List[str]) -> List[str]:
        """
        Args:
            data: Dict[str, str] | List[str], lerobot sample in raw mcap

        Returns:
            List[str], processed instructions
        """
        # if single instruction, convert to list
        if "coarse_task" in data:
            high_level_instruction = data["coarse_task"]
        else:
            high_level_instruction = ""
        if "task" not in data:
            return f"[high] {high_level_instruction}"

        low_level_instruction = data["task"]
        # Galaxea lerobot use @ to split Chinese and English instruction
        if "@" in low_level_instruction:
            zh, eng = low_level_instruction.split("@")
            low_level_instruction = zh if self.use_zh_instruction else eng

        if np.random.rand() < self.drop_high_level_prob:
            instruction = f"{low_level_instruction}"
        else: 
            instruction = f"[High]: {high_level_instruction}, [Low]: {low_level_instruction}"
        
        return instruction

    def action_state_transform(self, batch):
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["raw_shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, \
                    f"Action key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."
                    
        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["raw_shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, \
                f"State key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."
        
        if self.action_state_transforms is not None: 
            for trans in self.action_state_transforms:
                batch = trans.forward(batch)
        
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, \
                    f"Action key {k} actual transformed shape {actual_shape} mismatch with meta shape {meta_shape}."
        
        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, \
                f"State key {k} actual transformed shape {actual_shape} mismatch with meta raw shape {meta_shape}."
        
        return batch

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the data for the policy model.
        
        Args:
            Data: Dict[str, Any], lerobot sample in raw mcap obtained from dataset __getitem__:
                - "action": Optional, Dict[str, torch.Tensor] -> [action_horizon, action_dim]
                - "state": Dict[str, torch.Tensor] -> [num_obs_steps, state_dim]
                - "images": Dict[str, torch.Tensor] -> [num_obs_steps, C, H, W]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "idx": int, sample index
                
        Returns:
            Sample: Dict[str, Any], which can collated:
                - "input_ids": torch.Tensor -> [max_image_text_tokens,]
                - "attention_mask": torch.Tensor -> [max_image_text_tokens,]
                - "pixel_values": torch.Tensor -> [num_input_cameras, C, H, W]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "proprio": torch.Tensor -> [num_obs_steps, proprio_dim]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "action": Optional, torch.Tensor -> [action_horizon, action_dim]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "gt_action: Optional, deepcopy of input action for open loop eval, which is left untouched
                - "idx": int, sample index
        """
        sample = {}
        # 1. instruction
        sample["instruction"] = self.augment_instruction(data)
        sample["image_is_pad"] = data["image_is_pad"]

        # 2. image
        if self.vggt_camera_keys is not None:
            if self.is_train and getattr(self, "defer_vggt_official_preprocessing", False):
                sample["_vggt_raw_images"] = [data["images"][key][0].clone() for key in self.vggt_camera_keys]
            else:
                sample["vggt_images"] = prepare_vggt_images(
                    data["images"], self.vggt_camera_keys, self.vggt_image_size
                )
            if self.is_train and self.vggt_augmentation is not None and "_vggt_raw_images" not in sample:
                # Explicit camera/time axes: independent cameras, one current
                # frame each. Keep geometry labels and future RGB untouched.
                sample["vggt_images"] = self.vggt_augmentation(
                    sample["vggt_images"].unsqueeze(1)
                ).squeeze(1)
            if self.vggt_future_frame_index is not None and "_vggt_raw_images" not in sample:
                if self.emit_vggt_future_clip:
                    indices = list(range(0, self.num_obs_steps, 4))[:9]
                    if len(indices) != 9:
                        raise ValueError("Online VGGT latent targets require 9 source frames.")
                    sample["vggt_future_images"] = prepare_vggt_clip_images(
                        data["images"], self.vggt_camera_keys, self.vggt_image_size, indices
                    )
                    sample["vggt_future_is_pad"] = torch.as_tensor(
                        data["image_is_pad"][indices], dtype=torch.bool
                    )
                if self.emit_vggt_future_images and not self.emit_vggt_future_clip:
                    sample["vggt_future_images"] = prepare_vggt_images(
                        data["images"],
                        self.vggt_camera_keys,
                        self.vggt_image_size,
                        frame_index=self.vggt_future_frame_index,
                    )
                sample["vggt_future_is_pad"] = torch.as_tensor(
                    data["image_is_pad"][self.vggt_future_frame_index],
                    dtype=torch.bool,
                )
            # RGB-D targets are already view-stacked. Resize depth with nearest
            # (metric Z values must never be bilinearly mixed) and update K.
            if "vggt_recon_depth" in data and "_vggt_raw_images" not in sample:
                recon = prepare_vggt_reconstruction_targets(
                    data["images"], self.vggt_camera_keys,
                    data["vggt_recon_depth"], data["vggt_recon_depth_valid"],
                    data["vggt_recon_intrinsics"], data["vggt_recon_extrinsics"],
                    self.vggt_image_size,
                )
                sample["vggt_recon_depth"] = recon["depth"]
                sample["vggt_recon_depth_valid"] = recon["valid"]
                sample["vggt_recon_intrinsics"] = recon["intrinsics"]
                sample["vggt_recon_extrinsics"] = recon["extrinsics"]

        processed_images = []
        for meta in self.shape_meta["images"]:
            key, shape = meta["key"], meta["shape"]
            image = data["images"][key]  # [num_obs_steps, C, H, W]
            assert image.ndim == 4, f"Expected 4 dimensions (num_obs_steps, C, H, W), got shape {image.shape}"
            
            # Apply transforms efficiently on the merged batch
            transforms = self.train_transforms if self.is_train else self.val_transforms
            current_transforms = transforms[key] if isinstance(transforms, dict) else transforms
            for trans in current_transforms:
                image = trans(image)
            
            meta_shape = [self.num_obs_steps] + shape
            assert image.shape == meta_shape, \
                f"Expected shape {meta_shape}, got {image.shape} after transforms for key {key}"

            processed_images.append(image)
        pixel_values = torch.stack(processed_images, dim=0) # [num_input_cameras, T, C, H, W]
        
        if self.num_output_cameras > pixel_values.shape[0]:
            out = torch.zeros((self.num_output_cameras,) + pixel_values.shape[1:], device=pixel_values.device, dtype=pixel_values.dtype)
            out[0: pixel_values.shape[0]] = pixel_values
            sample["pixel_values"] = out
        elif self.num_output_cameras < pixel_values.shape[0]:
            logger.warning(f"num_output_cameras {self.num_output_cameras} is less than the number of cameras in data {pixel_values.shape[0]}, "
                           f"truncating the input to the first {self.num_output_cameras} cameras.")
            sample["pixel_values"] = pixel_values[:self.num_output_cameras]
        else:
            sample["pixel_values"] = pixel_values

        # Copy action before transform for open-loop evaluation, 
        # disabled for training dataset as it may cause collating key problem.
        if not self.is_train and "action" in data:
            sample["gt_action"] = deepcopy(data["action"])

        # 3. action & state
        if "action" in data and self.delta_action_dim_mask is not None:
            action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
            if bool(action_is_pad.any().item()):
                for key, dim_mask in self.delta_action_dim_mask.items():
                    cur_action = data["action"][key]
                    cur_action_is_pad = action_is_pad.to(device=cur_action.device)
                    cur_dim_mask = dim_mask.to(device=cur_action.device)
                    pad_delta_mask = cur_action_is_pad.unsqueeze(1) & cur_dim_mask.unsqueeze(0)
                    cur_action[pad_delta_mask] = 0.0
        data = self.action_state_transform(data)
        data = self.normalizer.forward(data)
        data = self.action_state_merger.forward(data)

        if "action" in data:
            sample["action"] = data["action"] # [action_horizon, action_dim]
            sample["action_is_pad"] = data["action_is_pad"] # [action_horizon,]
            sample["action_dim_is_pad"] = data["action_dim_is_pad"] # [action_dim,]
            assert sample["action"].shape[-1] == self.action_output_dim
            # sample["action"][sample["action_is_pad"], :-1] = 0.0 # NOTE: we assume use delta_eef_pose + gripper， so pad action is 0

        
        # TODO: rename all "state" into "proprio"
        sample["proprio"] = data["state"] # [num_obs_steps, proprio_dim]
        sample["proprio_is_pad"] = data["state_is_pad"] # [num_obs_steps,]
        sample["proprio_dim_is_pad"] = data["state_dim_is_pad"] # [proprio_dim,]
        assert sample["proprio"].shape[-1] == self.proprio_output_dim

        sample["idx"] = data["idx"]

        # sample = self.tokenizer(sample)
        
        return sample

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess the data for the policy model.
        
        Args:
            data: Dict[str, Any], lerobot sample in raw mcap

        Returns:
            data: Dict[str, Any], processed data including unnormalized action
        """
        assert "action" in data, "Action is required in postprocess"
        data["state"] = data.pop("proprio")
        data = self.action_state_merger.backward(data)
        data = self.normalizer.backward(data)
        if self.action_state_transforms is not None:
            for trans in reversed(self.action_state_transforms):
                data = trans.backward(data)

        start_obs_step = self.num_obs_steps - 1
        data["action"] = dict_apply(data["action"], lambda x: x[:, start_obs_step:, :])
        return data
