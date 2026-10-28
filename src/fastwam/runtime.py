import logging
import os
import inspect
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.video_io import save_mp4
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


def create_wan22_model(
    model_id: str,
    tokenizer_model_id: str,
    dit_config,
    tokenizer_max_len: int = 512,
    train_shift: float = 5.0,
    infer_shift: float = 5.0,
    num_train_timesteps: int = 1000,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.wan22 import Wan22Core

    if isinstance(dit_config, DictConfig):
        dit_config = OmegaConf.to_container(dit_config, resolve=True)
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must resolve to a dict, got {type(dit_config)}")

    return Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        dit_config=dit_config,
        train_shift=float(train_shift),
        infer_shift=float(infer_shift),
        num_train_timesteps=int(num_train_timesteps),
    )


def create_fastwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    fastwam_cls=None,
):
    from .models.wan22.fastwam import FastWAM

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    model_class = FastWAM if fastwam_cls is None else fastwam_cls
    return model_class.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        stop_action_grad_to_video=bool(loss.get("stop_action_grad_to_video", False)),
    )


def create_vggt_fastwam(
    *,
    vggt_weights_path: str,
    aggregator_checkpoint_path: str | None = None,
    layer_pairs,
    target_experts,
    projection_mode: str,
    projection_sharing: str = "shared",
    vggt_width: int = 1024,
    vggt_num_heads: int = 16,
    train_vggt: bool = False,
    **fastwam_kwargs,
):
    """Build FastWAM plus a frozen or trainable Aggregator-only VGGT source."""
    from .models.wan22.vggt_bridge import VGGTSharedAttentionBridge
    from .models.wan22.vggt_fastwam import VGGTFastWAM
    from .models.wan22.vggt_mot import build_vggt_backbone

    pairs = [tuple(int(v) for v in pair) for pair in layer_pairs]
    base = create_fastwam(fastwam_cls=VGGTFastWAM, **fastwam_kwargs)
    bridge = VGGTSharedAttentionBridge(
        fastwam_width=int(base.mot.num_heads * base.mot.attn_head_dim),
        vggt_width=int(vggt_width),
        fastwam_num_heads=int(base.mot.num_heads),
        vggt_num_heads=int(vggt_num_heads),
        layer_pairs=pairs,
        target_experts=tuple(target_experts),
        projection_mode=str(projection_mode),
        projection_sharing=str(projection_sharing),
        detach_vggt_memory=not bool(train_vggt),
    ).to(device=base.device, dtype=base.torch_dtype)
    base.mot.vggt_bridge = bridge
    callback_layers = tuple(pair[0] for pair in pairs)
    backbone = build_vggt_backbone(
        weights_path=vggt_weights_path,
        aggregator_checkpoint_path=aggregator_checkpoint_path,
        callback_layer_indices=callback_layers,
        trainable=bool(train_vggt),
    )
    return base.attach_vggt_backbone(backbone, train_vggt=bool(train_vggt))


def create_action_vggt_fastwam(
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
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    vggt_width: int = 1024,
    vggt_num_heads: int = 16,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    train_vggt: bool = False,
    skip_vggt_load_from_pretrain: bool = False,
    vggt_fp32_storage_bf16_compute: bool = False,
    reconstruction_loss_config: dict | None = None,
    vggt_freeze_mode: str = "full",
    vggt_unfreeze_last_n: int | None = None,
    vggt_head_checkpoint_path: str | None = None,
    lambda_vggt_recon: float = 0.05,
    num_vggt_forwards: int | None = None,
):
    """Build the action-only model without constructing Wan Video DiT or VAE."""
    from .models.wan22.action_dit import ActionDiT
    from .models.wan22.action_vggt_fastwam import ActionVGGTFastWAM
    from .models.wan22.action_vggt_loop_fastwam import ActionVGGTLoopFastWAM
    from .models.wan22.action_vggt_recon_fastwam import ActionVGGTReconFastWAM
    from .models.wan22.mot import MoT
    from .models.wan22.vggt_bridge import VGGTSharedAttentionBridge
    from .models.wan22.vggt_mot import build_loop_vggt_backbone, build_vggt_backbone

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if not isinstance(action_dit_config, dict):
        raise ValueError(
            f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}."
        )
    if "text_dim" not in action_dit_config:
        raise ValueError("`action_dit_config.text_dim` is required.")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if not isinstance(action_scheduler, dict):
        raise ValueError(
            f"`action_scheduler` must resolve to a dict, got {type(action_scheduler)}."
        )
    required_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_scheduler_keys = sorted(required_scheduler_keys - set(action_scheduler))
    if missing_scheduler_keys:
        raise ValueError(
            f"`action_scheduler` is missing required keys: {missing_scheduler_keys}."
        )

    if str(vggt_freeze_mode) not in {"full", "heads", "lastN"}:
        raise ValueError(f"Unsupported vggt_freeze_mode: {vggt_freeze_mode}")
    if vggt_unfreeze_last_n is not None and int(vggt_unfreeze_last_n) < 0:
        raise ValueError("vggt_unfreeze_last_n must be non-negative.")
    if vggt_head_checkpoint_path is not None and str(vggt_head_checkpoint_path) != str(vggt_weights_path):
        raise ValueError("Separate VGGT head checkpoint loading is not implemented.")

    pairs = [tuple(int(v) for v in pair) for pair in layer_pairs]
    action_expert = ActionDiT.from_pretrained(
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
    )
    bridge = VGGTSharedAttentionBridge(
        fastwam_width=int(action_expert.num_heads * action_expert.attn_head_dim),
        vggt_width=int(vggt_width),
        fastwam_num_heads=int(action_expert.num_heads),
        vggt_num_heads=int(vggt_num_heads),
        vggt_num_layers=24,
        fastwam_num_layers=len(action_expert.blocks),
        layer_pairs=pairs,
        target_experts=("action",),
        projection_mode=str(projection_mode),
        projection_sharing=str(projection_sharing),
        detach_vggt_memory=not bool(train_vggt),
    ).to(device=device, dtype=model_dtype)
    mot = MoT(
        mixtures={"action": action_expert},
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        vggt_bridge=bridge,
    )
    callback_layers = tuple(pair[0] for pair in pairs)
    backbone_kwargs = {
        "weights_path": vggt_weights_path,
        "aggregator_checkpoint_path": aggregator_checkpoint_path,
        "callback_layer_indices": callback_layers,
        "trainable": bool(train_vggt),
        "skip_vggt_load_from_pretrain": bool(skip_vggt_load_from_pretrain),
    }
    if num_vggt_forwards is None:
        backbone = build_vggt_backbone(
            **backbone_kwargs,
            enable_reconstruction_heads=reconstruction_loss_config is not None,
        )
        model_cls = (
            ActionVGGTReconFastWAM
            if reconstruction_loss_config is not None
            else ActionVGGTFastWAM
        )
    else:
        if reconstruction_loss_config is None:
            raise ValueError(
                "ActionVGGTLoopFastWAM requires `reconstruction_loss_config`."
            )
        backbone = build_loop_vggt_backbone(
            **backbone_kwargs,
            num_vggt_forwards=num_vggt_forwards,
        )
        model_cls = ActionVGGTLoopFastWAM
    model_kwargs = dict(
        action_expert=action_expert,
        mot=mot,
        vggt_backbone=backbone,
        text_dim=int(action_dit_config["text_dim"]),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        device=device,
        torch_dtype=model_dtype,
        train_vggt=bool(train_vggt),
        vggt_fp32_storage_bf16_compute=bool(vggt_fp32_storage_bf16_compute),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        reconstruction_loss_config=reconstruction_loss_config,
        lambda_vggt_recon=float(lambda_vggt_recon),
    )
    if num_vggt_forwards is not None:
        model_kwargs["num_vggt_forwards"] = num_vggt_forwards
    return model_cls(**model_kwargs)


def create_action_vggt_moe_fastwam(
    *,
    time_route_boundary: float = 0.8333333333333334,
    **kwargs,
):
    """Build the action-only hard time-route model with two independent VGGTs.

    A single base VGGT is loaded through the established action-only factory,
    then deep-copied before training.  This guarantees identical initialization
    values while keeping high/low route parameter storage independent.
    """
    import copy

    from .models.wan22.action_vggt_moe_fastwam import (
        ActionVGGTMoEFastWAM,
        TimeRoutedVGGTBackbone,
    )

    # Evaluation simulator configs carry these generic construction switches.
    # The restored action-only training config does not need a text encoder.
    kwargs.pop("load_text_encoder", None)
    generic_skip_dit = kwargs.pop("skip_dit_load_from_pretrain", None)
    if generic_skip_dit is not None:
        kwargs["skip_action_dit_load_from_pretrain"] = bool(generic_skip_dit)
    if kwargs.get("reconstruction_loss_config") is not None:
        raise ValueError(
            "ActionVGGTMoEFastWAM v1 does not support reconstruction heads/loss."
        )
    if kwargs.get("num_vggt_forwards") is not None:
        raise ValueError("ActionVGGTMoEFastWAM v1 does not support VGGT loops.")

    layer_pairs = tuple(
        (int(pair[0]), int(pair[1])) for pair in kwargs["layer_pairs"]
    )
    base = create_action_vggt_fastwam(**kwargs)
    high_noise_vggt = base.vggt_backbone
    low_noise_vggt = copy.deepcopy(high_noise_vggt)
    routed_backbone = TimeRoutedVGGTBackbone(
        high_noise_vggt=high_noise_vggt,
        low_noise_vggt=low_noise_vggt,
    )
    model = ActionVGGTMoEFastWAM(
        action_expert=base.action_expert,
        mot=base.mot,
        vggt_backbone=routed_backbone,
        text_dim=base.text_dim,
        layer_pairs=layer_pairs,
        time_route_boundary=float(time_route_boundary),
        proprio_dim=base.proprio_dim,
        device=str(base.device),
        torch_dtype=base.torch_dtype,
        train_vggt=base.train_vggt,
        action_train_shift=base.train_action_scheduler.shift,
        action_infer_shift=base.infer_action_scheduler.shift,
        action_num_train_timesteps=base.train_action_scheduler.num_train_timesteps,
    )
    # Keep the already seeded proprio initialization from the base factory;
    # ActionDiT and the shared bridge are also reused rather than reconstructed.
    if base.proprio_encoder is not None:
        model.proprio_encoder = base.proprio_encoder
    return model


def create_action_vggt_moe_recon_fastwam(
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
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    vggt_width: int = 1024,
    vggt_num_heads: int = 16,
    vggt_camera_keys=(),
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    train_vggt: bool = True,
    vggt_fp32_storage_bf16_compute: bool = False,
    time_route_boundary: float = 0.8333333333333334,
    low_noise_use_language_condition: bool = True,
    high_noise_use_language_condition: bool = False,
    reconstruction_loss_config=None,
    lambda_vggt_recon: float = 0.05,
    share_patch_embedding: bool = True,
    share_camera_register_tokens: bool = True,
    share_geometry_heads: bool = True,
    share_bridge: bool = True,
    checkpoint_sharing_policy: str = "strict",
    high_noise_reconstruction_target: str = "current",
    low_noise_reconstruction_target: str = "future",
    checkpoint_target_policy: str = "strict",
    checkpoint_language_policy: str = "strict",
    balance_route_action_weights: bool = False,
    load_text_encoder: bool = False,
    **_deployment_options,
):
    """Build timestep-routed reconstruction ActionDiT.

    High-noise samples optionally use language and have a configurable geometry
    target. Low-noise samples use language with an independently configurable geometry target. The two Aggregator
    streams optionally share DINO, token prefixes, FP32 heads and the bridge.
    """
    if not low_noise_use_language_condition:
        raise ValueError(
            "ActionVGGTMoEReconFastWAM requires language conditioning on the low route."
        )
    if reconstruction_loss_config is None:
        raise ValueError(
            "ActionVGGTMoEReconFastWAM requires reconstruction_loss_config."
        )
    from .models.wan22.action_dit import ActionDiT
    from .models.wan22.action_vggt_moe_recon_fastwam import ActionVGGTMoEReconFastWAM
    from .models.wan22.mot import MoT
    from .models.wan22.vggt_bridge import VGGTSharedAttentionBridge
    from .models.wan22.vggt_moe_recon_backbone import build_vggt_moe_recon_backbone

    # Evaluation simulator configs may include generic FastWAM construction
    # switches that are not part of the action-only factory contract.
    _ = load_text_encoder
    generic_skip_dit = _deployment_options.pop("skip_dit_load_from_pretrain", None)
    if generic_skip_dit is not None:
        skip_action_dit_load_from_pretrain = bool(generic_skip_dit)
    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if not isinstance(action_dit_config, dict) or not isinstance(action_scheduler, dict):
        raise ValueError("action_dit_config and action_scheduler must resolve to dicts.")

    pairs = [tuple(int(value) for value in pair) for pair in layer_pairs]
    action = ActionDiT.from_pretrained(
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
    )
    bridge = VGGTSharedAttentionBridge(
        fastwam_width=int(action.num_heads * action.attn_head_dim),
        vggt_width=int(vggt_width),
        fastwam_num_heads=int(action.num_heads),
        vggt_num_heads=int(vggt_num_heads),
        vggt_num_layers=24,
        fastwam_num_layers=len(action.blocks),
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
    backbone = build_vggt_moe_recon_backbone(
        weights_path=vggt_weights_path,
        aggregator_checkpoint_path=aggregator_checkpoint_path,
        callback_layer_indices=tuple(pair[0] for pair in pairs),
        input_camera_keys=tuple(str(key) for key in vggt_camera_keys),
        share_patch_embedding=bool(share_patch_embedding),
        share_camera_register_tokens=bool(share_camera_register_tokens),
        share_geometry_heads=bool(share_geometry_heads),
        high_noise_use_language_condition=bool(high_noise_use_language_condition),
        trainable=bool(train_vggt),
        text_dim=int(action_dit_config["text_dim"]),
    )
    return ActionVGGTMoEReconFastWAM(
        action_expert=action,
        mot=mot,
        vggt_backbone=backbone,
        text_dim=int(action_dit_config["text_dim"]),
        layer_pairs=pairs,
        time_route_boundary=float(time_route_boundary),
        share_bridge=bool(share_bridge),
        checkpoint_sharing_policy=str(checkpoint_sharing_policy),
        high_noise_reconstruction_target=str(high_noise_reconstruction_target),
        low_noise_reconstruction_target=str(low_noise_reconstruction_target),
        checkpoint_target_policy=str(checkpoint_target_policy),
        checkpoint_language_policy=str(checkpoint_language_policy),
        balance_route_action_weights=balance_route_action_weights,
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        device=device,
        torch_dtype=model_dtype,
        train_vggt=bool(train_vggt),
        vggt_fp32_storage_bf16_compute=bool(vggt_fp32_storage_bf16_compute),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        reconstruction_loss_config=reconstruction_loss_config,
        lambda_vggt_recon=float(lambda_vggt_recon),
    )


def create_action_vggt_multistage_recon_fastwam(
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
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    vggt_width: int = 1024,
    vggt_num_heads: int = 16,
    vggt_camera_keys=(),
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    train_vggt: bool = True,
    vggt_fp32_storage_bf16_compute: bool = False,
    num_stages: int = 3,
    routing_partition: str = "equal_probability",
    use_language_condition: bool = True,
    reconstruction_target: str = "future",
    stage_use_language_condition=None,
    stage_reconstruction_targets=None,
    reconstruction_loss_config=None,
    lambda_vggt_recon: float = 0.05,
    share_patch_embedding: bool = True,
    share_camera_register_tokens: bool = True,
    share_geometry_heads: bool = True,
    share_bridge: bool = True,
    balance_route_action_weights: bool = False,
    load_text_encoder: bool = False,
    **_deployment_options,
):
    """Construct independent, equally probable coarse1..N VGGT experts."""
    from .models.wan22.action_vggt_multistage_recon_fastwam import resolve_stage_options
    language, targets = resolve_stage_options(num_stages, use_language_condition, reconstruction_target,
                                              stage_use_language_condition, stage_reconstruction_targets)
    if routing_partition != "equal_probability":
        raise ValueError("Only equal_probability routing is supported.")
    if reconstruction_loss_config is None:
        raise ValueError(
            "ActionVGGTMultiStageReconFastWAM requires reconstruction_loss_config."
        )
    from .models.wan22.action_dit import ActionDiT
    from .models.wan22.action_vggt_multistage_recon_fastwam import ActionVGGTMultiStageReconFastWAM
    from .models.wan22.mot import MoT
    from .models.wan22.vggt_bridge import VGGTSharedAttentionBridge
    from .models.wan22.vggt_multistage_recon_backbone import build_vggt_multistage_recon_backbone

    # Evaluation simulator configs may include generic FastWAM construction
    # switches that are not part of the action-only factory contract.
    _ = load_text_encoder
    generic_skip_dit = _deployment_options.pop("skip_dit_load_from_pretrain", None)
    if generic_skip_dit is not None:
        skip_action_dit_load_from_pretrain = bool(generic_skip_dit)
    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if not isinstance(action_dit_config, dict) or not isinstance(action_scheduler, dict):
        raise ValueError("action_dit_config and action_scheduler must resolve to dicts.")

    pairs = [tuple(int(value) for value in pair) for pair in layer_pairs]
    action = ActionDiT.from_pretrained(
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
    )
    bridge = VGGTSharedAttentionBridge(
        fastwam_width=int(action.num_heads * action.attn_head_dim),
        vggt_width=int(vggt_width),
        fastwam_num_heads=int(action.num_heads),
        vggt_num_heads=int(vggt_num_heads),
        vggt_num_layers=24,
        fastwam_num_layers=len(action.blocks),
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
    backbone = build_vggt_multistage_recon_backbone(
        weights_path=vggt_weights_path,
        aggregator_checkpoint_path=aggregator_checkpoint_path,
        callback_layer_indices=tuple(pair[0] for pair in pairs),
        input_camera_keys=tuple(str(key) for key in vggt_camera_keys),
        share_patch_embedding=bool(share_patch_embedding),
        share_camera_register_tokens=bool(share_camera_register_tokens),
        share_geometry_heads=bool(share_geometry_heads),
        num_stages=num_stages,
        stage_use_language_condition=language,
        trainable=bool(train_vggt),
        text_dim=int(action_dit_config["text_dim"]),
    )
    return ActionVGGTMultiStageReconFastWAM(
        action_expert=action,
        mot=mot,
        vggt_backbone=backbone,
        text_dim=int(action_dit_config["text_dim"]),
        layer_pairs=pairs,
        num_stages=num_stages,
        routing_partition=routing_partition,
        stage_use_language_condition=language,
        stage_reconstruction_targets=targets,
        share_bridge=bool(share_bridge),
        balance_route_action_weights=balance_route_action_weights,
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        device=device,
        torch_dtype=model_dtype,
        train_vggt=bool(train_vggt),
        vggt_fp32_storage_bf16_compute=bool(vggt_fp32_storage_bf16_compute),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        reconstruction_loss_config=reconstruction_loss_config,
        lambda_vggt_recon=float(lambda_vggt_recon),
    )


def create_action_vggt_recon_fastwam(**kwargs):
    """Construct the explicit reconstruction variant via the shared factory."""
    # ``sim_robotwin.yaml`` carries the generic FastWAM construction switches.
    # The action-only factory has no text encoder and names the DiT switch more
    # explicitly, so translate/drop those fields when evaluation composes the
    # training task with the simulator config.
    kwargs.pop("load_text_encoder", None)
    generic_skip_dit = kwargs.pop("skip_dit_load_from_pretrain", None)
    if generic_skip_dit is not None:
        kwargs["skip_action_dit_load_from_pretrain"] = bool(generic_skip_dit)
    if kwargs.get("reconstruction_loss_config") is None:
        raise ValueError("ActionVGGTReconFastWAM requires reconstruction_loss_config.")
    return create_action_vggt_fastwam(**kwargs)


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
    num_vggt_forwards: int | None = None,
    reconstruction_target: str = "future",
    **_deployment_options,
):
    """Build single-stream language-conditioned geometry reconstruction model."""
    if reconstruction_target not in {"current", "future"}:
        raise ValueError("reconstruction_target must be 'current' or 'future'.")
    if num_vggt_forwards is not None and (
        isinstance(num_vggt_forwards, bool) or not isinstance(num_vggt_forwards, int)
        or num_vggt_forwards < 1
    ):
        raise ValueError("num_vggt_forwards must be a positive integer.")
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

    from .models.wan22.action_vggt_loop_recon_fastwam import ActionVGGTLoopReconFastWAM
    model_class = (E05 if num_vggt_forwards is None
                   else ActionVGGTLoopReconFastWAM)
    loop_options = {} if num_vggt_forwards is None else {"num_vggt_forwards": num_vggt_forwards}

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
        fastwam_width=int(action.num_heads * action.attn_head_dim),
        vggt_width=int(vggt_width),
        fastwam_num_heads=int(action.num_heads),
        vggt_num_heads=int(vggt_num_heads),
        vggt_num_layers=24,
        fastwam_num_layers=len(action.blocks),
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
        **loop_options,
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
    model = model_class(
        **loop_options,
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
    return _create_action_vggt_language_recon_model(**kwargs)


def create_action_vggt_lang_recon_fastwam(**kwargs):
    """Compatibility alias for configurations saved before the public e05 name."""
    return create_e05(**kwargs)


def create_action_vggt_loop_recon_fastwam(*, num_vggt_forwards: int = 2, **kwargs):
    """Build shared VGGT recurrence with full supervision on every pass."""
    if (isinstance(num_vggt_forwards, bool)
            or not isinstance(num_vggt_forwards, int) or num_vggt_forwards < 1):
        raise ValueError("num_vggt_forwards must be a positive integer.")
    return _create_action_vggt_language_recon_model(
        num_vggt_forwards=num_vggt_forwards, **kwargs,
    )


def create_action_vggt_loop_fastwam(*, num_vggt_forwards: int = 4, **kwargs):
    """Construct the reconstruction-first recurrent VGGT action model."""
    kwargs.pop("load_text_encoder", None)
    generic_skip_dit = kwargs.pop("skip_dit_load_from_pretrain", None)
    if generic_skip_dit is not None:
        kwargs["skip_action_dit_load_from_pretrain"] = bool(generic_skip_dit)
    if kwargs.get("reconstruction_loss_config") is None:
        raise ValueError("ActionVGGTLoopFastWAM requires reconstruction_loss_config.")
    return create_action_vggt_fastwam(
        **kwargs,
        num_vggt_forwards=num_vggt_forwards,
    )


def create_action_vggt_mot_fastwam(
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
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    vggt_width: int = 1024,
    vggt_num_heads: int = 16,
    vggt_camera_keys=(),
    va1_camera_keys=(),
    va2_camera_keys=(),
    isolate_vggt_streams: bool = False,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    train_vggt: bool = True,
):
    """Build ActionDiT with one-way memory from coupled or isolated VGGT streams."""
    from .models.wan22.action_dit import ActionDiT
    from .models.wan22.action_vggt_mot_fastwam import ActionVGGTMoTFastWAM
    from .models.wan22.mot import MoT
    from .models.wan22.vggt_bridge import VGGTSharedAttentionBridge
    from .models.wan22.vggt_mot_backbone import build_vggt_mot_backbone

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if not isinstance(action_dit_config, dict):
        raise ValueError(
            f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}."
        )
    if int(action_dit_config.get("num_layers", -1)) != 30:
        raise ValueError("ActionVGGTMoTFastWAM requires a 30-layer ActionDiT.")
    if "text_dim" not in action_dit_config:
        raise ValueError("`action_dit_config.text_dim` is required.")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if not isinstance(action_scheduler, dict):
        raise ValueError(
            f"`action_scheduler` must resolve to a dict, got {type(action_scheduler)}."
        )
    required_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_scheduler_keys = sorted(required_scheduler_keys - set(action_scheduler))
    if missing_scheduler_keys:
        raise ValueError(
            f"`action_scheduler` is missing required keys: {missing_scheduler_keys}."
        )

    pairs = [tuple(int(value) for value in pair) for pair in layer_pairs]
    action_expert = ActionDiT.from_pretrained(
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
    )
    bridge = VGGTSharedAttentionBridge(
        fastwam_width=int(action_expert.num_heads * action_expert.attn_head_dim),
        vggt_width=int(vggt_width),
        fastwam_num_heads=int(action_expert.num_heads),
        vggt_num_heads=int(vggt_num_heads),
        layer_pairs=pairs,
        target_experts=("action",),
        projection_mode=str(projection_mode),
        projection_sharing=str(projection_sharing),
        detach_vggt_memory=not bool(train_vggt),
    ).to(device=device, dtype=model_dtype)
    mot = MoT(
        mixtures={"action": action_expert},
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        vggt_bridge=bridge,
    )
    callback_layers = tuple(pair[0] for pair in pairs)
    backbone = build_vggt_mot_backbone(
        weights_path=vggt_weights_path,
        aggregator_checkpoint_path=aggregator_checkpoint_path,
        callback_layer_indices=callback_layers,
        input_camera_keys=tuple(str(key) for key in vggt_camera_keys),
        va1_camera_keys=tuple(str(key) for key in va1_camera_keys),
        va2_camera_keys=tuple(str(key) for key in va2_camera_keys),
        trainable=bool(train_vggt),
        isolate_vggt_streams=bool(isolate_vggt_streams),
    )
    return ActionVGGTMoTFastWAM(
        action_expert=action_expert,
        mot=mot,
        vggt_backbone=backbone,
        text_dim=int(action_dit_config["text_dim"]),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        device=device,
        torch_dtype=model_dtype,
        train_vggt=bool(train_vggt),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
    )


def create_action_vggt_mot_recon_fastwam(
    *, vggt_weights_path: str, aggregator_checkpoint_path: str | None = None,
    layer_pairs, projection_mode: str, projection_sharing: str = "shared",
    action_dit_config, action_scheduler, action_dit_pretrained_path: str | None = None,
    skip_action_dit_load_from_pretrain: bool = False, proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True, vggt_width: int = 1024,
    vggt_num_heads: int = 16, vggt_camera_keys=(), va1_camera_keys=(), va2_camera_keys=(),
    isolate_vggt_streams: bool = True, model_dtype: torch.dtype = torch.bfloat16,
    vggt_stream_attention: str | None = None,
    va1_reconstruction_target: str = "current",
    va1_use_language_condition: bool = False,
    va2_use_language_condition: bool = True,
    action_vggt_source: str = "both",
    share_patch_embedding: bool = True, share_camera_register_tokens: bool = True,
    share_geometry_heads: bool = True, share_bridge: bool = True,
    device: str = "cuda", train_vggt: bool = True, reconstruction_loss_config=None,
    lambda_vggt_recon: float = 0.05, vggt_head_checkpoint_path: str | None = None,
    load_text_encoder: bool = False, **_deployment_options,
):
    """Build the language-conditioned, dual-geometry ActionVGGT MoT model."""
    from .models.wan22.action_dit import ActionDiT
    from .models.wan22.action_vggt_mot_recon_fastwam import ActionVGGTMoTReconFastWAM
    from .models.wan22.mot import MoT
    from .models.wan22.vggt_bridge import VGGTSharedAttentionBridge
    from .models.wan22.vggt_mot_recon_backbone import build_vggt_mot_recon_backbone
    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if not share_bridge and projection_mode != "vggt_to_fastwam":
        raise ValueError("share_bridge=false requires projection_mode=vggt_to_fastwam.")
    pairs = [tuple(int(v) for v in pair) for pair in layer_pairs]
    action = ActionDiT.from_pretrained(action_dit_config=action_dit_config, action_dit_pretrained_path=action_dit_pretrained_path, skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain), device=device, torch_dtype=model_dtype)
    bridge = VGGTSharedAttentionBridge(fastwam_width=int(action.num_heads * action.attn_head_dim), vggt_width=int(vggt_width), fastwam_num_heads=int(action.num_heads), vggt_num_heads=int(vggt_num_heads), vggt_num_layers=24, fastwam_num_layers=len(action.blocks), layer_pairs=pairs, target_experts=("action",), projection_mode=str(projection_mode), projection_sharing=str(projection_sharing), detach_vggt_memory=not bool(train_vggt)).to(device=device, dtype=model_dtype)
    mot = MoT(mixtures={"action": action}, mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn), vggt_bridge=bridge)
    backbone = build_vggt_mot_recon_backbone(weights_path=vggt_weights_path, aggregator_checkpoint_path=aggregator_checkpoint_path, callback_layer_indices=tuple(p[0] for p in pairs), input_camera_keys=tuple(str(k) for k in vggt_camera_keys), va1_camera_keys=tuple(str(k) for k in va1_camera_keys), va2_camera_keys=tuple(str(k) for k in va2_camera_keys), trainable=bool(train_vggt), isolate_vggt_streams=bool(isolate_vggt_streams), vggt_stream_attention=vggt_stream_attention, text_dim=int(action_dit_config["text_dim"]), va1_use_language_condition=bool(va1_use_language_condition), va2_use_language_condition=bool(va2_use_language_condition), action_vggt_source=str(action_vggt_source), share_patch_embedding=share_patch_embedding, share_camera_register_tokens=share_camera_register_tokens, share_geometry_heads=share_geometry_heads, head_checkpoint_path=vggt_head_checkpoint_path)
    return ActionVGGTMoTReconFastWAM(action_expert=action, mot=mot, vggt_backbone=backbone, text_dim=int(action_dit_config["text_dim"]), proprio_dim=None if proprio_dim is None else int(proprio_dim), device=device, torch_dtype=model_dtype, train_vggt=bool(train_vggt), action_train_shift=float(action_scheduler["train_shift"]), action_infer_shift=float(action_scheduler["infer_shift"]), action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]), reconstruction_loss_config=reconstruction_loss_config, lambda_vggt_recon=float(lambda_vggt_recon), share_bridge=share_bridge, va1_reconstruction_target=va1_reconstruction_target)


def create_action_vggt_logit_diff_fastwam(
    *,
    vggt_weights_path: str,
    aggregator_checkpoint_path: str | None = None,
    layer_pairs,
    projection_mode: str,
    projection_sharing: str = "shared",
    contrastive_lambda: float = 0.1,
    contrastive_temperature: float = 1.0,
    action_dit_config,
    action_scheduler,
    action_dit_pretrained_path: str | None = None,
    skip_action_dit_load_from_pretrain: bool = False,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    vggt_width: int = 1024,
    vggt_num_heads: int = 16,
    vggt_camera_keys=(),
    va1_camera_keys=(),
    va2_camera_keys=(),
    isolate_vggt_streams: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    train_vggt: bool = True,
):
    """Build ActionDiT with joint softmax over action and logit-difference VGGT memory."""
    from .models.wan22.action_dit import ActionDiT
    from .models.wan22.action_vggt_logit_diff_fastwam import ActionVGGTLogitDiffFastWAM
    from .models.wan22.mot import MoT
    from .models.wan22.vggt_logit_diff_bridge import VGGTLogitDiffAttentionBridge
    from .models.wan22.vggt_mot_backbone import build_vggt_mot_backbone

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}.")
    if int(action_dit_config.get("num_layers", -1)) != 30:
        raise ValueError("ActionVGGTLogitDiffFastWAM requires a 30-layer ActionDiT.")
    if "text_dim" not in action_dit_config:
        raise ValueError("`action_dit_config.text_dim` is required.")
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must resolve to a dict, got {type(action_scheduler)}.")
    required_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_scheduler_keys = sorted(required_scheduler_keys - set(action_scheduler))
    if missing_scheduler_keys:
        raise ValueError(f"`action_scheduler` is missing required keys: {missing_scheduler_keys}.")

    input_cameras = tuple(str(key) for key in vggt_camera_keys)
    va1_cameras = tuple(str(key) for key in va1_camera_keys)
    va2_cameras = tuple(str(key) for key in va2_camera_keys)
    if va1_cameras != input_cameras or va2_cameras != input_cameras:
        raise ValueError(
            "ActionVGGTLogitDiffFastWAM requires va1_camera_keys and va2_camera_keys "
            "to exactly equal vggt_camera_keys."
        )
    if not isolate_vggt_streams:
        raise ValueError("ActionVGGTLogitDiffFastWAM requires isolate_vggt_streams=true.")

    pairs = [tuple(int(value) for value in pair) for pair in layer_pairs]
    action_expert = ActionDiT.from_pretrained(
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
    )
    bridge = VGGTLogitDiffAttentionBridge(
        fastwam_width=int(action_expert.num_heads * action_expert.attn_head_dim),
        vggt_width=int(vggt_width),
        fastwam_num_heads=int(action_expert.num_heads),
        vggt_num_heads=int(vggt_num_heads),
        layer_pairs=pairs,
        target_experts=("action",),
        projection_mode=str(projection_mode),
        projection_sharing=str(projection_sharing),
        contrastive_lambda=float(contrastive_lambda),
        contrastive_temperature=float(contrastive_temperature),
        detach_vggt_memory=not bool(train_vggt),
    ).to(device=device, dtype=model_dtype)
    mot = MoT(
        mixtures={"action": action_expert},
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        vggt_bridge=bridge,
    )
    callback_layers = tuple(pair[0] for pair in pairs)
    backbone = build_vggt_mot_backbone(
        weights_path=vggt_weights_path,
        aggregator_checkpoint_path=aggregator_checkpoint_path,
        callback_layer_indices=callback_layers,
        input_camera_keys=input_cameras,
        va1_camera_keys=va1_cameras,
        va2_camera_keys=va2_cameras,
        trainable=bool(train_vggt),
        isolate_vggt_streams=True,
    )
    return ActionVGGTLogitDiffFastWAM(
        action_expert=action_expert,
        mot=mot,
        vggt_backbone=backbone,
        text_dim=int(action_dit_config["text_dim"]),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        device=device,
        torch_dtype=model_dtype,
        train_vggt=bool(train_vggt),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
    )


def create_action_vggt_mod_fastwam(
    *,
    vggt_weights_path: str,
    aggregator_checkpoint_path: str | None = None,
    layer_pairs=None,
    projection_mode: str = "vggt_to_fastwam",
    projection_sharing: str = "shared",
    action_dit_config,
    action_scheduler,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    vggt_width: int = 1024,
    vggt_num_heads: int = 16,
    va2_input_mode: str = "block0_plus_final_tokens",
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    train_vggt: bool = True,
    load_text_encoder: bool = False,
    redirect_common_files: bool = True,
    **_unused,
):
    """Build ActionVGGTModFastWAM with serial 24+24-layer VGGT blocks."""
    from .models.wan22.action_dit import ActionDiT
    from .models.wan22.action_vggt_mod_fastwam import ActionVGGTModFastWAM
    from .models.wan22.mot import MoT
    from .models.wan22.serial_vggt import build_serial_vggt_backbone
    from .models.wan22.vggt_bridge import VGGTSharedAttentionBridge

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    action_dit_config = dict(action_dit_config)
    action_dit_config["num_layers"] = 48
    action_expert = ActionDiT(**action_dit_config).to(device=device, dtype=model_dtype)
    pairs = layer_pairs
    if pairs is None:
        pairs = [(i, i) for i in range(48)]
    bridge = VGGTSharedAttentionBridge(
        fastwam_width=int(action_expert.num_heads * action_expert.attn_head_dim),
        vggt_width=int(vggt_width),
        fastwam_num_heads=int(action_expert.num_heads),
        vggt_num_heads=int(vggt_num_heads),
        vggt_num_layers=48,
        fastwam_num_layers=48,
        layer_pairs=[tuple(int(v) for v in pair) for pair in pairs],
        target_experts=("action",),
        projection_mode=str(projection_mode),
        projection_sharing=str(projection_sharing),
        detach_vggt_memory=not bool(train_vggt),
    ).to(device=device, dtype=model_dtype)
    mot = MoT(
        mixtures={"action": action_expert},
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        vggt_bridge=bridge,
    )
    backbone = build_serial_vggt_backbone(
        weights_path=vggt_weights_path,
        aggregator_checkpoint_path=aggregator_checkpoint_path,
        va2_input_mode=str(va2_input_mode),
        trainable=bool(train_vggt),
    )
    required_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing = required_scheduler_keys - set(action_scheduler)
    if missing:
        raise ValueError(f"`action_scheduler` is missing required keys: {sorted(missing)}.")
    return ActionVGGTModFastWAM(
        action_expert=action_expert,
        mot=mot,
        vggt_backbone=backbone,
        text_dim=int(action_dit_config["text_dim"]),
        proprio_dim=proprio_dim,
        device=device,
        torch_dtype=model_dtype,
        train_vggt=bool(train_vggt),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
    )


def create_action_vggt_fastwam_4d(
    *,
    vggt_weights_path: str,
    aggregator_checkpoint_path: str | None = None,
    layer_pairs,
    projection_mode: str,
    projection_sharing: str = "shared",
    action_dit_config,
    four_d_dit_config,
    action_scheduler,
    four_d_scheduler,
    action_dit_pretrained_path: str | None = None,
    skip_action_dit_load_from_pretrain: bool = False,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    vggt_width: int = 1024,
    vggt_num_heads: int = 16,
    feature_dim: int = 2048,
    feature_layer: int = 23,
    expected_num_views: int = 2,
    loss=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    train_vggt: bool = True,
    use_precomputed_teacher_features: bool = False,
    action_only_inference: bool = False,
):
    """Build isolated action/4D experts and the requested teacher feature source."""
    from .models.wan22.action_dit import ActionDiT
    from .models.wan22.action_vggt_fastwam_4d import ActionVGGTFastWAM4D
    from .models.wan22.geoaware_projector import build_vggt_feature_backbone
    from .models.wan22.mot import MoT
    from .models.wan22.vggt_bridge import VGGTSharedAttentionBridge
    from .models.wan22.vggt_mot import build_vggt_backbone

    def resolved_dict(value, name: str) -> dict:
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if not isinstance(value, dict):
            raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}.")
        return dict(value)

    action_cfg = resolved_dict(action_dit_config, "action_dit_config")
    four_d_cfg = resolved_dict(four_d_dit_config, "four_d_dit_config")
    action_sched = resolved_dict(action_scheduler, "action_scheduler")
    four_d_sched = resolved_dict(four_d_scheduler, "four_d_scheduler")
    loss_cfg = {} if loss is None else resolved_dict(loss, "loss")

    for name, scheduler_cfg in (
        ("action_scheduler", action_sched),
        ("four_d_scheduler", four_d_sched),
    ):
        required = {"train_shift", "infer_shift", "num_train_timesteps"}
        missing = sorted(required - set(scheduler_cfg))
        if missing:
            raise ValueError(f"`{name}` is missing required keys: {missing}.")
    if action_cfg.get("text_dim") != four_d_cfg.get("text_dim"):
        raise ValueError(
            "Action and 4D experts must use the same `text_dim`: "
            f"{action_cfg.get('text_dim')} vs {four_d_cfg.get('text_dim')}."
        )
    if int(four_d_cfg.get("action_dim", -1)) != int(feature_dim):
        raise ValueError(
            "`four_d_dit_config.action_dim` must equal `feature_dim`: "
            f"{four_d_cfg.get('action_dim')} vs {feature_dim}."
        )

    action_expert = ActionDiT.from_pretrained(
        action_dit_config=action_cfg,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
    )
    four_d_expert = ActionDiT.from_pretrained(
        action_dit_config=four_d_cfg,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
    )
    pairs = [tuple(int(v) for v in pair) for pair in layer_pairs]
    bridge = VGGTSharedAttentionBridge(
        fastwam_width=int(action_expert.num_heads * action_expert.attn_head_dim),
        vggt_width=int(vggt_width),
        fastwam_num_heads=int(action_expert.num_heads),
        vggt_num_heads=int(vggt_num_heads),
        layer_pairs=pairs,
        target_experts=("four_d", "action"),
        projection_mode=str(projection_mode),
        projection_sharing=str(projection_sharing),
        detach_vggt_memory=not bool(train_vggt),
    ).to(device=device, dtype=model_dtype)
    mot = MoT(
        mixtures={"four_d": four_d_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        vggt_bridge=bridge,
    )

    callback_layers = tuple(pair[0] for pair in pairs)
    backbone = build_vggt_backbone(
        weights_path=vggt_weights_path,
        aggregator_checkpoint_path=aggregator_checkpoint_path,
        callback_layer_indices=callback_layers,
        trainable=bool(train_vggt),
    )
    teacher = None
    if not bool(use_precomputed_teacher_features) and not bool(action_only_inference):
        teacher = build_vggt_feature_backbone(
            weights_path=vggt_weights_path,
            aggregator_checkpoint_path=aggregator_checkpoint_path,
            selected_layers=(int(feature_layer),),
            trainable=False,
        )
    return ActionVGGTFastWAM4D(
        action_expert=action_expert,
        four_d_expert=four_d_expert,
        mot=mot,
        vggt_backbone=backbone,
        vggt_teacher=teacher,
        text_dim=int(action_cfg["text_dim"]),
        feature_dim=int(feature_dim),
        feature_layer=int(feature_layer),
        expected_num_views=int(expected_num_views),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        device=device,
        torch_dtype=model_dtype,
        train_vggt=bool(train_vggt),
        use_precomputed_teacher_features=bool(use_precomputed_teacher_features),
        action_only_inference=bool(action_only_inference),
        loss_lambda_4d=float(loss_cfg.get("lambda_4d", 1.0)),
        action_train_shift=float(action_sched["train_shift"]),
        action_infer_shift=float(action_sched["infer_shift"]),
        action_num_train_timesteps=int(action_sched["num_train_timesteps"]),
        four_d_train_shift=float(four_d_sched["train_shift"]),
        four_d_infer_shift=float(four_d_sched["infer_shift"]),
        four_d_num_train_timesteps=int(four_d_sched["num_train_timesteps"]),
    )


def create_action_vggt_mod_depth_fastwam(
    *,
    vggt_weights_path: str,
    aggregator_checkpoint_path: str | None = None,
    depth_teacher_checkpoint_path: str | None = None,
    layer_pairs=None,
    projection_mode: str = "vggt_to_fastwam",
    projection_sharing: str = "shared",
    action_dit_config,
    depth_dit_config,
    action_scheduler,
    depth_scheduler,
    depth_future_offsets,
    depth_patch_size: int = 16,
    depth_height: int = 256,
    depth_width: int = 256,
    depth_reads_action: bool = True,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    vggt_width: int = 1024,
    vggt_num_heads: int = 16,
    expected_num_views: int = 2,
    va2_input_mode: str = "block0_plus_final_tokens",
    loss=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    train_vggt: bool = True,
    action_only_inference: bool = False,
    **_unused,
):
    """Build serial-VGGT joint action/depth experts (48 layers each).

    Action and depth use one shared flow timestep but independent noisy tokens;
    serial VGGT emits callback memories 0..47 consumed by matching DiT layers.
    """
    del depth_teacher_checkpoint_path
    from .models.wan22.action_dit import ActionDiT
    from .models.wan22.action_vggt_mod_depth_fastwam import ActionVGGTModDepthFastWAM
    from .models.wan22.depth_dit import DepthDiT
    from .models.wan22.mot import MoT
    from .models.wan22.serial_vggt import build_serial_vggt_backbone
    from .models.wan22.vggt_bridge import VGGTSharedAttentionBridge

    def resolved(value, name):
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if not isinstance(value, dict):
            raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}.")
        return dict(value)

    action_cfg = resolved(action_dit_config, "action_dit_config")
    depth_cfg = resolved(depth_dit_config, "depth_dit_config")
    action_sched = resolved(action_scheduler, "action_scheduler")
    depth_sched = resolved(depth_scheduler, "depth_scheduler")
    loss_cfg = {} if loss is None else resolved(loss, "loss")
    required = {"train_shift", "infer_shift", "num_train_timesteps"}
    for name, sched in (("action_scheduler", action_sched), ("depth_scheduler", depth_sched)):
        missing = sorted(required - set(sched))
        if missing:
            raise ValueError(f"`{name}` is missing required keys: {missing}.")
    for key in required:
        if float(action_sched[key]) != float(depth_sched[key]):
            raise ValueError(f"Action and depth must share `{key}`.")
    if action_cfg.get("text_dim") != depth_cfg.get("text_dim"):
        raise ValueError("Action and depth experts must use the same `text_dim`.")
    patch = int(depth_patch_size)
    height, width = int(depth_height), int(depth_width)
    if patch <= 0 or height % patch or width % patch:
        raise ValueError("Depth height/width must be positive and divisible by depth_patch_size.")
    if int(depth_cfg.get("action_dim", -1)) != patch * patch:
        raise ValueError("depth_dit_config.action_dim must equal depth_patch_size^2.")
    action_cfg["num_layers"] = 48
    depth_cfg.update({
        "num_layers": 48,
        "depth_patch_size": patch,
        "future_offsets": tuple(int(v) for v in depth_future_offsets),
        "num_views": int(expected_num_views),
        "patch_grid_height": height // patch,
        "patch_grid_width": width // patch,
    })
    action = ActionDiT(**action_cfg).to(device=device, dtype=model_dtype)
    depth = DepthDiT(**depth_cfg).to(device=device, dtype=model_dtype)
    pairs = [(i, i) for i in range(48)] if layer_pairs is None else [tuple(int(v) for v in p) for p in layer_pairs]
    bridge = VGGTSharedAttentionBridge(
        fastwam_width=int(action.num_heads * action.attn_head_dim),
        vggt_width=int(vggt_width), vggt_num_heads=int(vggt_num_heads),
        fastwam_num_heads=int(action.num_heads), vggt_num_layers=48,
        fastwam_num_layers=48, layer_pairs=pairs,
        target_experts=("depth", "action"), projection_mode=str(projection_mode),
        projection_sharing=str(projection_sharing), detach_vggt_memory=not bool(train_vggt),
    ).to(device=device, dtype=model_dtype)
    mot = MoT(mixtures={"depth": depth, "action": action},
              mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn), vggt_bridge=bridge)
    backbone = build_serial_vggt_backbone(
        weights_path=vggt_weights_path,
        aggregator_checkpoint_path=aggregator_checkpoint_path,
        va2_input_mode=str(va2_input_mode), trainable=bool(train_vggt),
    )
    return ActionVGGTModDepthFastWAM(
        action_expert=action, depth_expert=depth, mot=mot, vggt_backbone=backbone,
        text_dim=int(action_cfg["text_dim"]), expected_num_views=int(expected_num_views),
        depth_reads_action=bool(depth_reads_action), proprio_dim=proprio_dim,
        device=device, torch_dtype=model_dtype, train_vggt=bool(train_vggt),
        action_only_inference=bool(action_only_inference),
        loss_lambda_depth=float(loss_cfg.get("lambda_depth", 1.0)),
        train_shift=float(action_sched["train_shift"]), infer_shift=float(action_sched["infer_shift"]),
        num_train_timesteps=int(action_sched["num_train_timesteps"]),
    )


def create_action_vggt_latent_fastwam(
    *, vggt_weights_path: str, aggregator_checkpoint_path: str | None = None,
    geometry_adapter_checkpoint_path: str | None = None, layer_pairs,
    projection_mode: str, projection_sharing: str = "shared",
    action_dit_config, vggt_dit_config, action_scheduler, vggt_scheduler,
    expected_num_views: int = 2, proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True, vggt_width: int = 1024,
    vggt_num_heads: int = 16, latent_frames: int = 3, latent_height: int = 24,
    latent_width: int = 20, loss=None, model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda", train_vggt: bool = True,
    action_dit_pretrained_path: str | None = None,
    skip_action_dit_load_from_pretrain: bool = False,
    action_only_inference: bool = False, use_online_latent_targets: bool = True,
    vggt_repo_path: str | None = None, **_ignored,
):
    """Build Action + adapter-latent experts; adapter targets are cache-side only."""
    from .models.wan22.action_dit import ActionDiT
    from .models.wan22.action_vggt_latent_fastwam import ActionVGGTLatentFastWAM
    from .models.wan22.vggt_latent_dit import VGGTLatentDiT
    from .models.wan22.mot import MoT
    from .models.wan22.vggt_bridge import VGGTSharedAttentionBridge
    from .models.wan22.vggt_mot import build_vggt_backbone

    def cfg(x):
        return dict(OmegaConf.to_container(x, resolve=True) if isinstance(x, DictConfig) else x)
    action_cfg, latent_cfg = cfg(action_dit_config), cfg(vggt_dit_config)
    action_sched, latent_sched = cfg(action_scheduler), cfg(vggt_scheduler)
    if action_cfg.get("text_dim") != latent_cfg.get("text_dim"):
        raise ValueError("Action and VGGT latent experts must share text_dim.")
    if int(latent_cfg.get("action_dim", -1)) != 48:
        raise ValueError("VGGT latent action_dim must be 48.")
    if (int(latent_frames), int(latent_height), int(latent_width)) != (3, 24, 20):
        raise ValueError("First-version VGGT latent contract is [48,3,24,20].")
    for key in ("train_shift", "infer_shift", "num_train_timesteps"):
        if float(action_sched[key]) != float(latent_sched[key]):
            raise ValueError("Action and VGGT latent schedulers must match.")
    action = ActionDiT.from_pretrained(action_dit_config=action_cfg, action_dit_pretrained_path=action_dit_pretrained_path, skip_dit_load_from_pretrain=skip_action_dit_load_from_pretrain, device=device, torch_dtype=model_dtype)
    latent_cfg.update({"latent_frames": 3, "latent_height": 24, "latent_width": 20, "max_seq_len": max(int(latent_cfg.get("max_seq_len", 0)), 1440)})
    latent = VGGTLatentDiT(**latent_cfg).to(device=device, dtype=model_dtype)
    pairs = [tuple(int(v) for v in pair) for pair in layer_pairs]
    bridge = VGGTSharedAttentionBridge(fastwam_width=int(action.num_heads * action.attn_head_dim), vggt_width=int(vggt_width), fastwam_num_heads=int(action.num_heads), vggt_num_heads=int(vggt_num_heads), layer_pairs=pairs, target_experts=("vggt", "action"), projection_mode=str(projection_mode), projection_sharing=str(projection_sharing), detach_vggt_memory=not bool(train_vggt)).to(device=device, dtype=model_dtype)
    mot = MoT(mixtures={"vggt": latent, "action": action}, mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn), vggt_bridge=bridge)
    backbone = build_vggt_backbone(weights_path=vggt_weights_path, aggregator_checkpoint_path=aggregator_checkpoint_path, callback_layer_indices=tuple(pair[0] for pair in pairs), trainable=bool(train_vggt))
    teacher = adapter = None
    if bool(use_online_latent_targets) and not bool(action_only_inference):
        from .models.vggt_gen3r_rect_features import FrozenVGGTOmegaGen3RRectFeatureEncoder
        from .models.vggt_gen3r_geometry_adapter_rect import FastWAMExactRectGeometryAdapter
        if not geometry_adapter_checkpoint_path:
            raise ValueError("geometry_adapter_checkpoint_path is required for online latent targets.")
        adapter = FastWAMExactRectGeometryAdapter()
        adapter.load_pretrained(geometry_adapter_checkpoint_path)
        # The teacher and adapter are frozen, but both must reside on the
        # same device as the online future RGB tensor before convolution.
        # Geometry features are emitted in float32 by the frozen Omega
        # encoder; keep adapter weights float32 to avoid mixed-type conv
        # failures (the resulting latent is cast to model dtype downstream).
        adapter = adapter.to(device=device, dtype=torch.float32)
        adapter.eval().requires_grad_(False)
        teacher = FrozenVGGTOmegaGen3RRectFeatureEncoder(
            checkpoint_path=vggt_weights_path,
            repo_path=vggt_repo_path or str(Path(vggt_weights_path).resolve().parents[2] / "vggt-omega"),
            device=device,
            last_layers=4,
            # The Omega aggregator contains dynamic callback-index set logic
            # that is incompatible with TorchDynamo on PyTorch 2.10.  The
            # frozen online teacher is used only for target generation, so
            # keep it eager (the trainable current-image backbone remains
            # separately configurable).
            compile_backbone=False,
        )
    return ActionVGGTLatentFastWAM(action_expert=action, vggt_expert=latent, mot=mot, vggt_backbone=backbone, vggt_geometry_teacher=teacher, geometry_adapter=adapter, text_dim=int(action_cfg["text_dim"]), expected_num_views=int(expected_num_views), proprio_dim=proprio_dim, device=device, torch_dtype=model_dtype, train_vggt=bool(train_vggt), action_only_inference=bool(action_only_inference), loss_lambda_vggt=float((cfg(loss) if loss else {}).get("lambda_vggt", 1.0)), train_shift=float(action_sched["train_shift"]), infer_shift=float(latent_sched["infer_shift"]), num_train_timesteps=int(action_sched["num_train_timesteps"]))


def create_action_vggt_depth_fastwam(
    *,
    vggt_weights_path: str,
    aggregator_checkpoint_path: str | None = None,
    depth_teacher_checkpoint_path: str | None = None,
    layer_pairs,
    projection_mode: str,
    projection_sharing: str = "shared",
    action_dit_config,
    depth_dit_config,
    action_scheduler,
    depth_scheduler,
    depth_future_offsets,
    depth_patch_size: int = 16,
    depth_height: int = 256,
    depth_width: int = 256,
    depth_reads_action: bool = True,
    action_dit_pretrained_path: str | None = None,
    skip_action_dit_load_from_pretrain: bool = False,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    vggt_width: int = 1024,
    vggt_num_heads: int = 16,
    expected_num_views: int = 2,
    loss=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    train_vggt: bool = True,
    action_only_inference: bool = False,
):
    """Build joint Action/Depth experts with current-frame VGGT K/V memory."""
    del depth_teacher_checkpoint_path  # Used only by the offline cache builder.
    from .models.wan22.action_dit import ActionDiT
    from .models.wan22.action_vggt_depth_fastwam import ActionVGGTDepthFastWAM
    from .models.wan22.depth_dit import DepthDiT
    from .models.wan22.mot import MoT
    from .models.wan22.vggt_bridge import VGGTSharedAttentionBridge
    from .models.wan22.vggt_mot import build_vggt_backbone

    def resolved_dict(value, name: str) -> dict:
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if not isinstance(value, dict):
            raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}.")
        return dict(value)

    action_cfg = resolved_dict(action_dit_config, "action_dit_config")
    depth_cfg = resolved_dict(depth_dit_config, "depth_dit_config")
    action_sched = resolved_dict(action_scheduler, "action_scheduler")
    depth_sched = resolved_dict(depth_scheduler, "depth_scheduler")
    loss_cfg = {} if loss is None else resolved_dict(loss, "loss")
    required_scheduler = {"train_shift", "infer_shift", "num_train_timesteps"}
    for name, scheduler_cfg in (
        ("action_scheduler", action_sched),
        ("depth_scheduler", depth_sched),
    ):
        missing = sorted(required_scheduler - set(scheduler_cfg))
        if missing:
            raise ValueError(f"`{name}` is missing required keys: {missing}.")
    for key in required_scheduler:
        if float(action_sched[key]) != float(depth_sched[key]):
            raise ValueError(
                "Action and depth must share one flow schedule; "
                f"`{key}` differs: {action_sched[key]} vs {depth_sched[key]}."
            )
    if action_cfg.get("text_dim") != depth_cfg.get("text_dim"):
        raise ValueError(
            "Action and depth experts must use the same `text_dim`: "
            f"{action_cfg.get('text_dim')} vs {depth_cfg.get('text_dim')}."
        )
    patch_size = int(depth_patch_size)
    height = int(depth_height)
    width = int(depth_width)
    if patch_size <= 0 or height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            "Depth height/width must be positive and divisible by depth_patch_size: "
            f"{(height, width)} / {patch_size}."
        )
    if int(depth_cfg.get("action_dim", -1)) != patch_size * patch_size:
        raise ValueError(
            "`depth_dit_config.action_dim` must equal depth_patch_size^2: "
            f"{depth_cfg.get('action_dim')} vs {patch_size * patch_size}."
        )
    offsets = tuple(int(value) for value in depth_future_offsets)
    depth_cfg.update(
        {
            "depth_patch_size": patch_size,
            "future_offsets": offsets,
            "num_views": int(expected_num_views),
            "patch_grid_height": height // patch_size,
            "patch_grid_width": width // patch_size,
        }
    )

    action_expert = ActionDiT.from_pretrained(
        action_dit_config=action_cfg,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
    )
    depth_expert = DepthDiT.from_pretrained(
        action_dit_config=depth_cfg,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
    )
    pairs = [tuple(int(value) for value in pair) for pair in layer_pairs]
    bridge = VGGTSharedAttentionBridge(
        fastwam_width=int(action_expert.num_heads * action_expert.attn_head_dim),
        vggt_width=int(vggt_width),
        fastwam_num_heads=int(action_expert.num_heads),
        vggt_num_heads=int(vggt_num_heads),
        layer_pairs=pairs,
        target_experts=("depth", "action"),
        projection_mode=str(projection_mode),
        projection_sharing=str(projection_sharing),
        detach_vggt_memory=not bool(train_vggt),
    ).to(device=device, dtype=model_dtype)
    mot = MoT(
        mixtures={"depth": depth_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        vggt_bridge=bridge,
    )
    callback_layers = tuple(pair[0] for pair in pairs)
    backbone = build_vggt_backbone(
        weights_path=vggt_weights_path,
        aggregator_checkpoint_path=aggregator_checkpoint_path,
        callback_layer_indices=callback_layers,
        trainable=bool(train_vggt),
    )
    return ActionVGGTDepthFastWAM(
        action_expert=action_expert,
        depth_expert=depth_expert,
        mot=mot,
        vggt_backbone=backbone,
        text_dim=int(action_cfg["text_dim"]),
        expected_num_views=int(expected_num_views),
        depth_reads_action=bool(depth_reads_action),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        device=device,
        torch_dtype=model_dtype,
        train_vggt=bool(train_vggt),
        action_only_inference=bool(action_only_inference),
        loss_lambda_depth=float(loss_cfg.get("lambda_depth", 1.0)),
        train_shift=float(action_sched["train_shift"]),
        infer_shift=float(action_sched["infer_shift"]),
        num_train_timesteps=int(action_sched["num_train_timesteps"]),
    )


def create_action_geoaware_fastwam(
    *,
    vggt_weights_path: str,
    aggregator_checkpoint_path: str | None = None,
    selected_layers,
    expected_num_views: int,
    action_dit_config,
    action_scheduler,
    action_dit_pretrained_path: str | None = None,
    skip_action_dit_load_from_pretrain: bool = False,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    vggt_feature_dim: int = 2048,
    projector_layer_dim: int = 512,
    projector_output_dim: int = 512,
    visual_context_dim: int = 1024,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Build the action-only GeoAware hidden-feature baseline."""
    from .models.wan22.action_dit import ActionDiT
    from .models.wan22.action_geoaware_fastwam import ActionGeoAwareFastWAM
    from .models.wan22.geoaware_projector import (
        GeoAwareProjector,
        VisualContextAdapter,
        build_vggt_feature_backbone,
    )
    from .models.wan22.mot import MoT

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if not isinstance(action_dit_config, dict):
        raise ValueError(
            f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}."
        )
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if not isinstance(action_scheduler, dict):
        raise ValueError(
            f"`action_scheduler` must resolve to a dict, got {type(action_scheduler)}."
        )
    required_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_scheduler_keys = sorted(required_scheduler_keys - set(action_scheduler))
    if missing_scheduler_keys:
        raise ValueError(
            f"`action_scheduler` is missing required keys: {missing_scheduler_keys}."
        )

    layers = tuple(int(layer) for layer in selected_layers)
    action_expert = ActionDiT.from_pretrained(
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
    )
    if int(visual_context_dim) != int(action_expert.hidden_dim):
        raise ValueError(
            "`visual_context_dim` must equal action_dit_config.hidden_dim: "
            f"{visual_context_dim} vs {action_expert.hidden_dim}."
        )
    mot = MoT(
        mixtures={"action": action_expert},
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        vggt_bridge=None,
    )
    projector = GeoAwareProjector(
        selected_layers=layers,
        input_dim=int(vggt_feature_dim),
        layer_dim=int(projector_layer_dim),
        output_dim=int(projector_output_dim),
    ).to(device=device, dtype=model_dtype)
    visual_adapter = VisualContextAdapter(
        input_dim=int(projector_output_dim),
        output_dim=int(visual_context_dim),
    ).to(device=device, dtype=model_dtype)
    backbone = build_vggt_feature_backbone(
        weights_path=vggt_weights_path,
        aggregator_checkpoint_path=aggregator_checkpoint_path,
        selected_layers=layers,
    )
    return ActionGeoAwareFastWAM(
        action_expert=action_expert,
        mot=mot,
        vggt_backbone=backbone,
        geo_projector=projector,
        visual_adapter=visual_adapter,
        text_dim=int(action_dit_config["text_dim"]),
        expected_num_views=int(expected_num_views),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        device=device,
        torch_dtype=model_dtype,
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
    )


def create_action_ca_vggt_fastwam(
    *,
    vggt_weights_path: str,
    aggregator_checkpoint_path: str | None = None,
    layer_pairs,
    expected_num_views: int,
    action_dit_config,
    action_scheduler,
    action_dit_pretrained_path: str | None = None,
    skip_action_dit_load_from_pretrain: bool = False,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    vggt_feature_dim: int = 2048,
    visual_context_dim: int = 1024,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    train_vggt: bool = False,
):
    """Build ActionCAVGGTFastWAM without Video DiT, VAE, or VGGT K/V bridge."""
    from .models.wan22.action_ca_vggt_fastwam import ActionCAVGGTFastWAM
    from .models.wan22.action_dit import ActionDiT
    from .models.wan22.geoaware_projector import build_vggt_feature_backbone
    from .models.wan22.layerwise_vggt_context import (
        LayerwiseVGGTContextAdapter,
    )
    from .models.wan22.mot import MoT

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if not isinstance(action_dit_config, dict):
        raise ValueError(
            f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}."
        )
    if "text_dim" not in action_dit_config:
        raise ValueError("`action_dit_config.text_dim` is required.")
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if not isinstance(action_scheduler, dict):
        raise ValueError(
            f"`action_scheduler` must resolve to a dict, got {type(action_scheduler)}."
        )
    required_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_scheduler_keys = sorted(required_scheduler_keys - set(action_scheduler))
    if missing_scheduler_keys:
        raise ValueError(
            f"`action_scheduler` is missing required keys: {missing_scheduler_keys}."
        )

    pairs = tuple(tuple(pair) for pair in layer_pairs)
    action_expert = ActionDiT.from_pretrained(
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
    )
    if int(visual_context_dim) != int(action_expert.hidden_dim):
        raise ValueError(
            "`visual_context_dim` must equal action_dit_config.hidden_dim: "
            f"{visual_context_dim} vs {action_expert.hidden_dim}."
        )
    visual_context_adapter = LayerwiseVGGTContextAdapter(
        layer_pairs=pairs,
        input_dim=int(vggt_feature_dim),
        output_dim=int(visual_context_dim),
        action_num_layers=len(action_expert.blocks),
    ).to(device=device, dtype=model_dtype)
    mot = MoT(
        mixtures={"action": action_expert},
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        vggt_bridge=None,
    )
    backbone = build_vggt_feature_backbone(
        weights_path=vggt_weights_path,
        aggregator_checkpoint_path=aggregator_checkpoint_path,
        selected_layers=visual_context_adapter.selected_vggt_layers,
        trainable=bool(train_vggt),
    )
    return ActionCAVGGTFastWAM(
        action_expert=action_expert,
        mot=mot,
        vggt_backbone=backbone,
        visual_context_adapter=visual_context_adapter,
        text_dim=int(action_dit_config["text_dim"]),
        expected_num_views=int(expected_num_views),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        device=device,
        torch_dtype=model_dtype,
        train_vggt=bool(train_vggt),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
    )


def create_fastwam_joint(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.fastwam_joint import FastWAMJoint

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAMJoint.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def create_fastwam_idm(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.fastwam_idm import (
        FastWAMIDM,
    )

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAMIDM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


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


_VGGT_FP32_BF16_FACTORY_TARGETS = {
    "e05.create_e05",
    "fastwam.runtime.create_e05",
    "fastwam.runtime.create_action_vggt_fastwam",
    "fastwam.runtime.create_action_vggt_recon_fastwam",
    "fastwam.runtime.create_action_vggt_lang_recon_fastwam",
    "fastwam.runtime.create_action_vggt_moe_recon_fastwam",
    "fastwam.runtime.create_action_vggt_multistage_recon_fastwam",
}


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
        model_target.endswith(
            (
                "create_vggt_fastwam",
                "create_action_vggt_fastwam",
                "create_action_vggt_moe_fastwam",
                "create_action_vggt_moe_recon_fastwam",
                "create_action_vggt_multistage_recon_fastwam",
                "create_action_vggt_recon_fastwam",
                "create_e05",
                "create_action_vggt_lang_recon_fastwam",
                "create_action_vggt_loop_recon_fastwam",
                "create_action_vggt_loop_fastwam",
                "create_action_vggt_mot_fastwam",
                "create_action_vggt_mot_recon_fastwam",
                "create_action_vggt_mod_fastwam",
                "create_action_vggt_fastwam_4d",
                "create_action_vggt_latent_fastwam",
                "create_action_vggt_depth_fastwam",
                "create_action_vggt_mod_depth_fastwam",
                "create_action_geoaware_fastwam",
                "create_action_ca_vggt_fastwam",
            )
        )
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

def run_inference(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    inference_cfg = cfg.inference
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    checkpoint_path = inference_cfg.get("checkpoint_path")
    if checkpoint_path and not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Inference checkpoint not found: {checkpoint_path}")
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False)).model
    if checkpoint_path and "pretrained_checkpoint_path" in model_cfg.keys():
        model_cfg.pretrained_checkpoint_path = None
        if "vggt_backbone_pretrained_checkpoint_path" in model_cfg.keys():
            model_cfg.vggt_backbone_pretrained_checkpoint_path = None
        model_cfg.skip_action_dit_load_from_pretrain = True
        model_cfg.skip_vggt_load_from_pretrain = True
    model = _instantiate_training_model(
        model_cfg,
        model_dtype=model_dtype,
        device=str(inference_cfg.device),
        mixed_precision=mixed_precision,
        vggt_fp32_storage_bf16_compute=cfg.get("vggt_fp32_storage_bf16_compute", False),
    )
    if checkpoint_path:
        logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
        model.load_checkpoint(checkpoint_path)
    model.eval()
    
    def center_crop_resize(img: Image, width: int, height: int) -> Image.Image:
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        return resized.crop((left, top, left + width, top + height))

    input_image = Image.open(str(inference_cfg.input_image_path)).convert("RGB")
    input_image = center_crop_resize(input_image, width=inference_cfg.width, height=inference_cfg.height)
    arr = np.array(input_image, dtype=np.float32)
    x = torch.from_numpy(arr)
    x = x.to(device=model.device, dtype=model.torch_dtype)
    x = x * (2.0 / 255.0) - 1.0
    x = repeat(x, "H W C -> B C H W", B=1)
    output_mp4 = str(inference_cfg.output_mp4)

    action_horizon = inference_cfg.get("action_horizon")
    if isinstance(action_horizon, bool) or not isinstance(action_horizon, int) or action_horizon <= 0:
        raise ValueError("`inference.action_horizon` must be a positive integer.")

    infer_kwargs = {
        "prompt": str(inference_cfg.prompt),
        "negative_prompt": str(inference_cfg.negative_prompt),
        "text_cfg_scale": float(inference_cfg.text_cfg_scale),
        "action_cfg_scale": float(inference_cfg.action_cfg_scale),
        "input_image": x,
        "num_frames": int(inference_cfg.num_frames),
        "action_horizon": action_horizon,
        "num_inference_steps": int(inference_cfg.num_inference_steps),
        "sigma_shift": None if inference_cfg.get("sigma_shift") is None else float(inference_cfg.sigma_shift),
        "seed": int(inference_cfg.seed),
        "rand_device": str(inference_cfg.rand_device),
        "tiled": bool(inference_cfg.tiled),
    }

    infer_out = model.infer(**infer_kwargs)
    video = infer_out["video"]
    save_mp4(video, output_mp4, fps=15)
    logger.info("Saved inference video to %s", output_mp4)
    return output_mp4
