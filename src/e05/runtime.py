import logging
import os
import inspect
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils import misc

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
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


def _create_action_vggt_language_recon_model(
    *,
    vggt_weights_path: str,
    aggregator_checkpoint_path: str | None = None,
    layer_pairs,
    projection_mode: str,
    projection_sharing: str = "shared",
    action_dit_config,
    action_scheduler,
    action_dit_pretrained_path: str | None = None,
    skip_action_dit_load_from_pretrain: bool = False,
    skip_vggt_load_from_pretrain: bool = False,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    vggt_width: int = 1024,
    vggt_num_heads: int = 16,
    vggt_camera_keys=(),
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    train_vggt: bool = True,
    reconstruction_loss_config=None,
    lambda_vggt_recon: float = 0.05,
    vggt_head_checkpoint_path: str | None = None,
    load_text_encoder: bool = False,
    pretrained_checkpoint_path: str | None = None,
    vggt_backbone_pretrained_checkpoint_path: str | None = None,
    load_pretrained_action_expert: bool = True,
    load_pretrained_bridge: bool = True,
    reset_action_projections: bool = False,
    reset_proprio_encoder: bool = False,
    reconstruction_target: str = "future",
    **_deployment_options,
):
    """Build single-stream language-conditioned geometry reconstruction model."""
    if reconstruction_target not in {"current", "future"}:
        raise ValueError("reconstruction_target must be 'current' or 'future'.")
    unsupported = set(_deployment_options) - {"skip_dit_load_from_pretrain", "vggt_fp32_storage_bf16_compute"}
    if unsupported:
        raise TypeError(f"Unsupported E05 options: {sorted(unsupported)}")
    generic_skip_dit = _deployment_options.pop("skip_dit_load_from_pretrain", None)
    if generic_skip_dit is not None:
        skip_action_dit_load_from_pretrain = bool(generic_skip_dit)
    del load_text_encoder
    if vggt_backbone_pretrained_checkpoint_path:
        if pretrained_checkpoint_path:
            raise ValueError("Use only one of pretrained_checkpoint_path and vggt_backbone_pretrained_checkpoint_path.")
        if not Path(vggt_backbone_pretrained_checkpoint_path).is_file():
            raise FileNotFoundError(f"VGGT backbone checkpoint not found: {vggt_backbone_pretrained_checkpoint_path}")
        skip_vggt_load_from_pretrain = True
    if pretrained_checkpoint_path:
        if not Path(pretrained_checkpoint_path).is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_checkpoint_path}")
        # The full-model checkpoint contains both branches and geometry heads.
        skip_action_dit_load_from_pretrain = True
        skip_vggt_load_from_pretrain = True
    from .models.wan22.action_dit import ActionDiT
    from .models.wan22.e05 import E05
    from .models.wan22.mot import MoT
    from .models.wan22.vggt_bridge import VGGTSharedAttentionBridge
    from .models.wan22.vggt_lang_recon_backbone import (
        build_vggt_language_recon_backbone,
    )

    if reconstruction_loss_config is None:
        raise ValueError(
            "e05 requires reconstruction_loss_config."
        )
    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if not isinstance(action_dit_config, dict) or not isinstance(action_scheduler, dict):
        raise ValueError("action_dit_config and action_scheduler must resolve to dicts.")
    if "text_dim" not in action_dit_config:
        raise ValueError("`action_dit_config.text_dim` is required.")
    required_scheduler = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_scheduler = sorted(required_scheduler - set(action_scheduler))
    if missing_scheduler:
        raise ValueError(
            f"action_scheduler is missing required keys: {missing_scheduler}."
        )

    pairs = [tuple(int(value) for value in pair) for pair in layer_pairs]
    action = ActionDiT.from_pretrained(
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
    )
    bridge = VGGTSharedAttentionBridge(
        e05_width=int(action.num_heads * action.attn_head_dim),
        vggt_width=int(vggt_width),
        e05_num_heads=int(action.num_heads),
        vggt_num_heads=int(vggt_num_heads),
        vggt_num_layers=24,
        e05_num_layers=len(action.blocks),
        layer_pairs=pairs,
        target_experts=("action",),
        projection_mode=str(projection_mode),
        projection_sharing=str(projection_sharing),
        detach_vggt_memory=not bool(train_vggt),
    ).to(device=device, dtype=model_dtype)
    mot = MoT(
        mixtures={"action": action},
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        vggt_bridge=bridge,
    )
    backbone = build_vggt_language_recon_backbone(
        weights_path=vggt_weights_path,
        aggregator_checkpoint_path=aggregator_checkpoint_path,
        head_checkpoint_path=vggt_head_checkpoint_path,
        callback_layer_indices=tuple(pair[0] for pair in pairs),
        text_dim=int(action_dit_config["text_dim"]),
        trainable=bool(train_vggt),
        skip_vggt_load_from_pretrain=bool(skip_vggt_load_from_pretrain),
    )
    if vggt_backbone_pretrained_checkpoint_path:
        backbone.load_pretrained_checkpoint(vggt_backbone_pretrained_checkpoint_path)
        logger.info("Loaded complete VGGT backbone from %s", vggt_backbone_pretrained_checkpoint_path)
    model = E05(
        reconstruction_target=reconstruction_target,
        action_expert=action,
        mot=mot,
        vggt_backbone=backbone,
        text_dim=int(action_dit_config["text_dim"]),
        vggt_camera_keys=tuple(vggt_camera_keys),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        device=device,
        torch_dtype=model_dtype,
        train_vggt=bool(train_vggt),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        reconstruction_loss_config=reconstruction_loss_config,
        lambda_vggt_recon=float(lambda_vggt_recon),
    )
    if pretrained_checkpoint_path:
        model.load_pretrained_checkpoint(
            pretrained_checkpoint_path,
            load_pretrained_action_expert=load_pretrained_action_expert,
            load_pretrained_bridge=load_pretrained_bridge,
            reset_action_projections=reset_action_projections,
            reset_proprio_encoder=reset_proprio_encoder,
        )
    return model


def create_e05(**kwargs):
    """Build e05 with the original 109k-compatible architecture and parameter names."""
    from .checkpoint_compat import normalize_metadata
    if "projection_mode" in kwargs:
        kwargs["projection_mode"] = normalize_metadata(kwargs["projection_mode"])
    return _create_action_vggt_language_recon_model(**kwargs)


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


_VGGT_FP32_BF16_FACTORY_TARGETS = {"e05.create_e05", "e05.runtime.create_e05"}


def _instantiate_training_model(model_cfg, *, model_dtype, device, mixed_precision,
                                vggt_fp32_storage_bf16_compute=False):
    enabled = bool(vggt_fp32_storage_bf16_compute)
    if enabled and mixed_precision != "bf16":
        raise ValueError(
            "vggt_fp32_storage_bf16_compute requires mixed_precision=bf16."
        )
    kwargs = {"model_dtype": model_dtype, "device": device}
    target = str(model_cfg.get("_target_", ""))
    if enabled and target in _VGGT_FP32_BF16_FACTORY_TARGETS:
        kwargs["vggt_fp32_storage_bf16_compute"] = True
    return instantiate(model_cfg, **kwargs)


def run_training(cfg: DictConfig):
    if cfg.resume and "pretrained_checkpoint_path" in cfg.model.keys():
        # Clear unused initialization paths before *any* interpolation, including
        # config serialization: the source may reference an unset environment var.
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        cfg.model.pretrained_checkpoint_path = None
        if "vggt_backbone_pretrained_checkpoint_path" in cfg.model.keys():
            cfg.model.vggt_backbone_pretrained_checkpoint_path = None
        cfg.model.action_dit_pretrained_path = None
        cfg.model.skip_action_dit_load_from_pretrain = True
        cfg.model.skip_vggt_load_from_pretrain = True
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    model_target = str(cfg.model.get("_target_", ""))
    if (
        model_target.endswith("create_e05")
        and not cfg.model.get("aggregator_checkpoint_path")
        and not cfg.model.get("skip_vggt_load_from_pretrain", False)
        and not cfg.model.get("pretrained_checkpoint_path")
        and not cfg.model.get("vggt_backbone_pretrained_checkpoint_path")
    ):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            raise ValueError(
                "Distributed VGGT-backed training requires "
                "`model.aggregator_checkpoint_path`; "
                "do not make every rank load the full VGGT checkpoint."
            )
        logger.warning(
            "VGGT-backed training is falling back to the full VGGT checkpoint. "
            "Set `model.aggregator_checkpoint_path` before multi-rank deployment."
        )
    # Seed before constructing either the model or datasets so randomly
    # initialized modules and any stochastic dataset setup honor cfg.seed.
    set_global_seed(
        int(cfg.seed),
        get_worker_init_fn=False,
        rank_offset=False,
    )
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = _instantiate_training_model(
        cfg.model,
        model_dtype=model_dtype,
        device=model_device,
        mixed_precision=mixed_precision,
        vggt_fp32_storage_bf16_compute=cfg.get("vggt_fp32_storage_bf16_compute", False),
    )
    train_ds, val_ds = build_datasets(cfg.data)

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()
