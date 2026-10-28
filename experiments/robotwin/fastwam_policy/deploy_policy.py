import logging
import os
import sys
import time
import inspect
import hashlib
import atexit
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.cache_io import cache_exists, load_cache
from fastwam.datasets.lerobot.processors.fastwam_processor import (
    FastWAMProcessor,
    prepare_vggt_images,
    prepare_vggt_reconstruction_targets,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.transforms.robotwin_ee import (
    absolute_ee_action,
    quaternion_pose_to_ee_state,
)
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.models.wan22.helpers.loader import (
    _load_registered_model,
    _resolve_configs,
)
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from .reconstruction_visualizer import ReconstructionVisualizer

logger = logging.getLogger(__name__)
_TEXT_CONTEXT_CACHE: dict[Path, tuple[torch.Tensor, torch.Tensor]] = {}
_ROBOTWIN_CAMERA_MAP = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resolve_project_path(path_value: str) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_value))))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _is_action_only_model_config(model_cfg: DictConfig) -> bool:
    target = str(model_cfg.get("_target_", ""))
    return target.endswith(
        (
            "create_action_vggt_fastwam",
            "create_action_vggt_moe_fastwam",
            "create_action_vggt_moe_recon_fastwam",
            "create_action_vggt_multistage_recon_fastwam",
            "create_action_vggt_loop_fastwam",
            "create_action_vggt_mot_fastwam",
            "create_action_vggt_mot_recon_fastwam",
            "create_action_vggt_logit_diff_fastwam",
            "create_action_vggt_mod_fastwam",
            "create_action_vggt_fastwam_4d",
            "create_action_vggt_latent_fastwam",
            "create_action_vggt_depth_fastwam",
            "create_action_vggt_mod_depth_fastwam",
            "create_action_vggt_recon_fastwam",
            "create_e05",
            "create_action_vggt_lang_recon_fastwam",
            "create_action_vggt_loop_recon_fastwam",
            "create_action_geoaware_fastwam",
            "create_action_ca_vggt_fastwam",
        )
    )


def _restore_training_model_and_data(
    cfg: DictConfig,
    training_config_path: Optional[str],
) -> DictConfig:
    if _is_none_like(training_config_path):
        restored = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    else:
        config_path = _resolve_project_path(str(training_config_path))
        if not config_path.is_file():
            raise FileNotFoundError(f"Training config not found: {config_path}")
        training_cfg = OmegaConf.load(config_path)
        missing = [
            key for key in ("model", "data") if training_cfg.get(key) is None
        ]
        if missing:
            raise ValueError(
                f"Training config {config_path} is missing: {', '.join(missing)}"
            )
        restored = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        restored.model = OmegaConf.create(
            OmegaConf.to_container(training_cfg.model, resolve=False)
        )
        restored.data = OmegaConf.create(
            OmegaConf.to_container(training_cfg.data, resolve=False)
        )
        logger.info("Restored model/data config from %s", config_path)

    with open_dict(restored.model):
        if "vggt_backbone_pretrained_checkpoint_path" in restored.model.keys():
            restored.model.vggt_backbone_pretrained_checkpoint_path = None
        if "pretrained_checkpoint_path" in restored.model.keys():
            restored.model.pretrained_checkpoint_path = None
            restored.model.skip_vggt_load_from_pretrain = True
        if _is_action_only_model_config(restored.model):
            restored.model.skip_action_dit_load_from_pretrain = True
            restored.model.action_dit_pretrained_path = None
            if str(restored.model.get("_target_", "")).endswith(
                ("create_action_vggt_fastwam_4d", "create_action_vggt_latent_fastwam", "create_action_vggt_depth_fastwam", "create_action_vggt_mod_depth_fastwam")
            ):
                restored.model.action_only_inference = True
        else:
            restored.model.load_text_encoder = True
            restored.model.skip_dit_load_from_pretrain = True
            restored.model.action_dit_pretrained_path = None
    return restored


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8))
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        text_embedding_cache_dir: Optional[str],
        context_len: int,
        online_text_embeddings: bool = True,
        write_text_embedding_cache: bool = False,
        online_text_cache_size: int = 256,
        reconstruction_visualization_enabled: bool = False,
        reconstruction_visualization_output_dir: Optional[str] = None,
        reconstruction_visualization_max_episodes: int = 3,
        reconstruction_visualization_every_n_replans: int = 1,
        reconstruction_visualization_include_gt_depth: bool = True,
        reconstruction_visualization_save_panel: bool = True,
        reconstruction_visualization_save_npz: bool = True,
        reconstruction_visualization_save_pointcloud: bool = True,
        reconstruction_visualization_save_video: bool = True,
        reconstruction_visualization_video_fps: int = 2,
        reconstruction_visualization_point_stride: int = 4,
    ) -> None:
        model_cfg_copy = deepcopy(model_cfg)
        if "vggt_backbone_pretrained_checkpoint_path" in model_cfg_copy.keys():
            model_cfg_copy.vggt_backbone_pretrained_checkpoint_path = None
        if "pretrained_checkpoint_path" in model_cfg_copy.keys():
            model_cfg_copy.pretrained_checkpoint_path = None
            model_cfg_copy.skip_action_dit_load_from_pretrain = True
            model_cfg_copy.skip_vggt_load_from_pretrain = True
        OmegaConf.resolve(model_cfg_copy)

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()
        self.device = torch.device(device)
        self.device = torch.device(device)

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)
        self._ee_delta_actions = (
            self.processor.action_representation == "robotwin_ee_delta_chunk_local_rot6d"
        )

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self._text_embedding_cache_dir = (
            None
            if _is_none_like(text_embedding_cache_dir)
            else _resolve_project_path(str(text_embedding_cache_dir))
        )
        self._context_len = int(context_len)
        self._online_text_embeddings = bool(online_text_embeddings)
        self._write_text_embedding_cache = bool(write_text_embedding_cache)
        self._online_text_cache = {}
        self._online_text_cache_size = max(0, int(online_text_cache_size))
        self._text_encoder = None
        self._tokenizer = None
        if self._online_text_embeddings:
            _, text_config, _, tokenizer_config = _resolve_configs(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
                redirect_common_files=True,
            )
            text_config.download_if_necessary()
            tokenizer_config.download_if_necessary()
            self._text_encoder = _load_registered_model(
                text_config.path,
                "wan_video_text_encoder",
                torch_dtype=model_dtype,
                device=device,
            ).eval()
            self._tokenizer = HuggingfaceTokenizer(
                name=tokenizer_config.path,
                seq_len=self._context_len,
                clean="whitespace",
            )

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
        self._reconstruction_visualization_max_episodes = max(
            0, int(reconstruction_visualization_max_episodes)
        )
        self._reconstruction_visualization_every_n_replans = max(
            1, int(reconstruction_visualization_every_n_replans)
        )
        self._reconstruction_visualization_include_gt_depth = bool(
            reconstruction_visualization_include_gt_depth
        )
        self._reconstruction_replan_count = 0
        self._reconstruction_visualizer = None
        if reconstruction_visualization_enabled:
            if getattr(self.model, "reconstruction_loss_config", None) is None:
                raise ValueError(
                    "Reconstruction visualization requires a model with enabled "
                    "VGGT reconstruction heads."
                )
            if _is_none_like(reconstruction_visualization_output_dir):
                raise ValueError(
                    "Reconstruction visualization requires a resolved output directory."
                )
            self._reconstruction_visualizer = ReconstructionVisualizer(
                str(reconstruction_visualization_output_dir),
                save_panel=reconstruction_visualization_save_panel,
                save_npz=reconstruction_visualization_save_npz,
                save_pointcloud=reconstruction_visualization_save_pointcloud,
                save_video=reconstruction_visualization_save_video,
                video_fps=reconstruction_visualization_video_fps,
                point_stride=reconstruction_visualization_point_stride,
            )
            atexit.register(self._reconstruction_visualizer.close)

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _raw_state(self, observation: Dict[str, Any]) -> np.ndarray:
        if not self._ee_delta_actions:
            return np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        endpose = observation["endpose"]
        pose = np.concatenate(
            (
                np.asarray(endpose["left_endpose"], dtype=np.float32),
                [float(endpose["left_gripper"])],
                np.asarray(endpose["right_endpose"], dtype=np.float32),
                [float(endpose["right_gripper"])],
            )
        ).astype(np.float32)
        return quaternion_pose_to_ee_state(torch.from_numpy(pose)).numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _build_robotwin_vggt_tensor(
        self,
        observation: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        camera_keys = self.processor.vggt_camera_keys
        if camera_keys is None:
            return None
        obs_data = observation["observation"]
        missing_config = [
            key for key in camera_keys if key not in _ROBOTWIN_CAMERA_MAP
        ]
        if missing_config:
            raise KeyError(
                "Unsupported RoboTwin VGGT camera keys: "
                f"{missing_config}. Supported keys: {sorted(_ROBOTWIN_CAMERA_MAP)}"
            )

        image_sequences: dict[str, torch.Tensor] = {}
        for key in camera_keys:
            simulator_key = _ROBOTWIN_CAMERA_MAP[key]
            if simulator_key not in obs_data or "rgb" not in obs_data[simulator_key]:
                raise KeyError(
                    f"RoboTwin observation is missing `{simulator_key}.rgb` "
                    f"for configured VGGT camera `{key}`."
                )
            image = np.asarray(obs_data[simulator_key]["rgb"])
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"RoboTwin VGGT view `{key}` must be RGB [H,W,3], "
                    f"got {tuple(image.shape)}."
                )
            image_sequences[key] = (
                torch.from_numpy(np.ascontiguousarray(image))
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

        views = prepare_vggt_images(
            image_sequences,
            camera_keys=list(camera_keys),
            image_size=int(self.processor.vggt_image_size),
        )
        if (
            views.ndim != 4
            or views.shape[0] != len(camera_keys)
            or views.shape[1] != 3
        ):
            raise ValueError(
                "RoboTwin VGGT preprocessing must return [V,3,H,W] with "
                "one output per configured camera, got "
                f"{tuple(views.shape)} for camera_keys={list(camera_keys)}."
            )
        return views.unsqueeze(0).contiguous()

    def _get_text_context(
        self,
        prompt: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._text_embedding_cache_dir is None:
            raise ValueError(
                "Action-only RoboTwin eval requires "
                "`data.train.text_embedding_cache_dir` in the training config."
            )
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = self._text_embedding_cache_dir / (
            f"{digest}.t5_len{self._context_len}.wan22ti2v5b.pt"
        )
        if not cache_exists(cache_path):
            if not self._online_text_embeddings:
                raise FileNotFoundError(
                    f"Missing action-only RoboTwin text embedding cache: {cache_path}."
                )
            cached_online = self._online_text_cache.get(prompt)
            if cached_online is not None:
                return cached_online
            if self._text_encoder is None or self._tokenizer is None:
                raise RuntimeError("Online text embeddings are enabled but encoder is unavailable")
            with torch.inference_mode():
                ids, mask = self._tokenizer(
                    [prompt], return_mask=True, add_special_tokens=True
                )
                ids = ids.to(self.device)
                mask = mask.to(self.device, dtype=torch.bool)
                context = self._text_encoder(ids, mask)[0]
                context[~mask[0]] = 0.0
            language_mask = mask[0].contiguous()
            # ActionDiT historically receives an all-ones mask after padded
            # language embeddings have been zeroed.  Keep that behavior while
            # returning the original mask separately for va2 language attention.
            result = (context.contiguous(), torch.ones_like(language_mask), language_mask)
            if self._online_text_cache_size > 0:
                if len(self._online_text_cache) >= self._online_text_cache_size:
                    self._online_text_cache.pop(next(iter(self._online_text_cache)))
                self._online_text_cache[prompt] = result
            if self._write_text_embedding_cache and self._text_embedding_cache_dir is not None:
                self._text_embedding_cache_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "context": context.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
                    "mask": mask[0].detach().to("cpu", dtype=torch.bool).contiguous(),
                }
                tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
                try:
                    torch.save(payload, tmp_path)
                    os.replace(tmp_path, cache_path)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
            return result
        cache_key = cache_path.resolve()
        cached = _TEXT_CONTEXT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        payload = load_cache(cache_path)
        if not isinstance(payload, dict) or not {"context", "mask"} <= set(payload):
            raise ValueError(
                f"Invalid text embedding cache payload in {cache_path}; "
                "expected context/mask."
            )
        context = payload["context"]
        context_mask = payload["mask"]
        if not isinstance(context, torch.Tensor) or context.ndim != 2:
            raise ValueError(
                f"Cached context must be [L,D], got "
                f"{type(context)} {getattr(context, 'shape', None)}."
            )
        if not isinstance(context_mask, torch.Tensor) or context_mask.ndim != 1:
            raise ValueError(
                f"Cached mask must be [L], got "
                f"{type(context_mask)} {getattr(context_mask, 'shape', None)}."
            )
        if (
            context.shape[0] != self._context_len
            or context_mask.shape[0] != self._context_len
        ):
            raise ValueError(
                f"Cached context length mismatch in {cache_path}: expected "
                f"{self._context_len}, got context={context.shape[0]}, "
                f"mask={context_mask.shape[0]}."
            )
        context = context.clone()
        valid_mask = context_mask.to(dtype=torch.bool)
        context[~valid_mask] = 0
        result = (context, torch.ones_like(valid_mask), valid_mask)
        _TEXT_CONTEXT_CACHE[cache_key] = result
        return result

    def _prepare_reconstruction_ground_truth(
        self,
        observation: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self._reconstruction_visualization_include_gt_depth:
            return None
        camera_keys = self.processor.vggt_camera_keys
        if camera_keys is None:
            return None
        obs_data = observation["observation"]
        image_sequences = {}
        depths, intrinsics, extrinsics = [], [], []
        for key in camera_keys:
            simulator_key = _ROBOTWIN_CAMERA_MAP[key]
            camera = obs_data[simulator_key]
            if "depth" not in camera:
                logger.warning(
                    "GT reconstruction visualization requested, but RoboTwin "
                    "observation has no `%s.depth`; saving predictions only.",
                    simulator_key,
                )
                return None
            image_sequences[key] = (
                torch.from_numpy(np.ascontiguousarray(camera["rgb"]))
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            # RoboTwin exposes millimetres; reconstruction training targets use metres.
            depth = torch.as_tensor(np.asarray(camera["depth"]), dtype=torch.float32) / 1000.0
            depths.append(depth)
            intrinsics.append(torch.as_tensor(camera["intrinsic_cv"], dtype=torch.float32))
            extrinsic = torch.as_tensor(camera["extrinsic_cv"], dtype=torch.float32)
            extrinsics.append(extrinsic[:3, :4])
        depth = torch.stack(depths)
        return prepare_vggt_reconstruction_targets(
            image_sequences,
            list(camera_keys),
            depth,
            torch.isfinite(depth) & (depth > 0),
            torch.stack(intrinsics),
            torch.stack(extrinsics),
            int(self.processor.vggt_image_size),
        )

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        vggt_images = self._build_robotwin_vggt_tensor(observation)
        state_vector = self._raw_state(observation)
        proprio = self._normalize_state(state_vector)
        episode_index = max(0, self.episode_count - 1)
        capture_reconstruction = (
            self._reconstruction_visualizer is not None
            and episode_index < self._reconstruction_visualization_max_episodes
            and self._reconstruction_replan_count
            % self._reconstruction_visualization_every_n_replans
            == 0
        )

        prompt = DEFAULT_PROMPT.format(task=instruction)
        if getattr(self.model, "evaluation_mode", None) == "action_only":
            if vggt_images is None:
                raise ValueError(
                    "ActionVGGTFastWAM RoboTwin eval requires configured "
                    "VGGT camera inputs."
                )
            context, context_mask, language_context_mask = self._get_text_context(prompt)
            infer_kwargs = {
                "action_horizon": self.action_horizon,
                "vggt_images": vggt_images,
                "context": context,
                "context_mask": context_mask,
                "proprio": proprio,
                "num_inference_steps": self.num_inference_steps,
                "sigma_shift": self.sigma_shift,
                "seed": self.seed,
                "rand_device": self.rand_device,
            }
            # Language-conditioned reconstruction models consume the original
            # language padding mask. Keep the legacy action-only infer_action
            # ABI unchanged; existing models reject unknown keyword args.
            if getattr(self.model, "supports_language_context_mask", False):
                infer_kwargs["language_context_mask"] = language_context_mask
            if capture_reconstruction:
                infer_kwargs["return_reconstruction"] = True
        else:
            infer_kwargs = {
                "prompt": prompt,
                "input_image": image_tensor,
                "action_horizon": self.action_horizon,
                "proprio": proprio,
                "negative_prompt": self.negative_prompt,
                "text_cfg_scale": self.text_cfg_scale,
                "num_inference_steps": self.num_inference_steps,
                "sigma_shift": self.sigma_shift,
                "seed": self.seed,
                "rand_device": self.rand_device,
                "tiled": self.tiled,
            }
            if vggt_images is not None:
                infer_kwargs["vggt_images"] = vggt_images
            if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
                infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        if isinstance(pred, dict) and "reconstruction" in pred:
            if self._reconstruction_visualizer is not None:
                geometry = pred["reconstruction"]
                if hasattr(self.model, "reconstruction_streams"):
                    geometry = {name: dict(value) for name, value in geometry.items()}
                    for name, (stream, target, cameras) in self.model.reconstruction_streams().items():
                        geometry[name].update(stream=stream, target=target, camera_keys=list(cameras))
                frame_dir = self._reconstruction_visualizer.save(
                    episode_index=episode_index,
                    replan_index=self._reconstruction_replan_count,
                    vggt_images=vggt_images,
                    reconstruction=geometry,
                    ground_truth=self._prepare_reconstruction_ground_truth(observation),
                    input_camera_keys=self.processor.vggt_camera_keys,
                )
                logger.info("Saved VGGT reconstruction visualization to %s", frame_dir)
        if self._reconstruction_visualizer is not None:
            self._reconstruction_replan_count += 1

        action_tensor = pred if isinstance(pred, torch.Tensor) else pred["action"]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        if self._ee_delta_actions:
            base = torch.from_numpy(self._raw_state(observation)).unsqueeze(0)
            action_chunk = absolute_ee_action(torch.from_numpy(action_chunk), base).numpy()
        if not np.isfinite(action_chunk).all():
            raise ValueError("Inferred RoboTwin action contains non-finite values")
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="ee" if self._ee_delta_actions else "qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        if self._reconstruction_visualizer is not None:
            self._reconstruction_visualizer.close()
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self._reconstruction_replan_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )
    cfg = _restore_training_model_and_data(
        cfg,
        training_config_path=usr_args.get("training_config_path"),
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    visualization_cfg = cfg.EVALUATION.get("reconstruction_visualization", {})
    visualization_enabled = _parse_bool(
        usr_args.get(
            "reconstruction_visualization_enabled",
            visualization_cfg.get("enabled", False),
        )
    )
    visualization_output_dir = usr_args.get("reconstruction_visualization_output_dir")
    if _is_none_like(visualization_output_dir):
        eval_output_dir = usr_args.get("eval_output_dir")
        if not _is_none_like(eval_output_dir):
            visualization_output_dir = str(
                Path(str(eval_output_dir)) / "reconstruction_vis"
            )

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        text_embedding_cache_dir=cfg.data.train.get("text_embedding_cache_dir"),
        context_len=int(cfg.data.train.get("context_len", 128)),
        reconstruction_visualization_enabled=visualization_enabled,
        reconstruction_visualization_output_dir=visualization_output_dir,
        reconstruction_visualization_max_episodes=int(
            usr_args.get(
                "reconstruction_visualization_max_episodes",
                visualization_cfg.get("max_episodes", 3),
            )
        ),
        reconstruction_visualization_every_n_replans=int(
            usr_args.get(
                "reconstruction_visualization_every_n_replans",
                visualization_cfg.get("every_n_replans", 1),
            )
        ),
        reconstruction_visualization_include_gt_depth=_parse_bool(
            usr_args.get(
                "reconstruction_visualization_include_gt_depth",
                visualization_cfg.get("include_gt_depth", True),
            )
        ),
        reconstruction_visualization_save_panel=_parse_bool(
            usr_args.get("reconstruction_visualization_save_panel", visualization_cfg.get("save_panel", True))
        ),
        reconstruction_visualization_save_npz=_parse_bool(
            usr_args.get("reconstruction_visualization_save_npz", visualization_cfg.get("save_npz", True))
        ),
        reconstruction_visualization_save_pointcloud=_parse_bool(
            usr_args.get("reconstruction_visualization_save_pointcloud", visualization_cfg.get("save_pointcloud", True))
        ),
        reconstruction_visualization_save_video=_parse_bool(
            usr_args.get("reconstruction_visualization_save_video", visualization_cfg.get("save_video", True))
        ),
        reconstruction_visualization_video_fps=int(
            usr_args.get("reconstruction_visualization_video_fps", visualization_cfg.get("video_fps", 2))
        ),
        reconstruction_visualization_point_stride=int(
            usr_args.get("reconstruction_visualization_point_stride", visualization_cfg.get("point_stride", 4))
        ),
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
