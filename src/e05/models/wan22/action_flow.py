from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_dit import ActionDiT
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .vggt_mot import FrozenVGGTBackbone, validate_vggt_images
from .vggt_reconstruction_loss import compute_vggt_reconstruction_loss, canonical_reconstruction_config


class ActionFlowBase(nn.Module):
    """Action-only flow model backed by frozen or trainable VGGT K/V memory."""

    evaluation_mode = "action_only"

    def __init__(
        self,
        *,
        action_expert: ActionDiT,
        mot: MoT,
        vggt_backbone: FrozenVGGTBackbone,
        text_dim: int,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        train_vggt: bool = False,
        vggt_fp32_storage_bf16_compute: bool = False,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        reconstruction_loss_config: Optional[dict] = None,
        lambda_vggt_recon: float = 0.05,
    ) -> None:
        super().__init__()
        if tuple(mot.expert_order) != ("action",):
            raise ValueError(
                "ActionFlowBase requires an action-only MoT with expert_order=('action',)."
            )
        if mot.mixtures["action"] is not action_expert:
            raise ValueError("`action_expert` must be the action module owned by `mot`.")
        if mot.vggt_bridge is None:
            raise ValueError("ActionFlowBase requires `mot.vggt_bridge`.")
        if mot.vggt_bridge.target_experts != frozenset({"action"}):
            raise ValueError("ActionFlowBase bridge must target only the action expert.")

        self.mot = mot
        # Trainer compatibility: action-only MoT contains exactly ActionDiT + bridge.
        self.dit = self.mot
        self.train_vggt = bool(train_vggt)
        self.vggt_fp32_storage_bf16_compute = bool(vggt_fp32_storage_bf16_compute)
        vggt_backbone.fp32_storage_bf16_compute = self.vggt_fp32_storage_bf16_compute
        self.reconstruction_loss_config = canonical_reconstruction_config(reconstruction_loss_config)
        self.lambda_vggt_recon = float(lambda_vggt_recon)
        self.mot.vggt_bridge.detach_vggt_memory = not self.train_vggt
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.proprio_encoder = (
            None
            if self.proprio_dim is None
            else nn.Linear(self.proprio_dim, self.text_dim, dtype=torch_dtype)
        )

        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )

        # Preserve the historical external ownership while frozen. A trainable
        # backbone must be registered for Accelerate/DeepSpeed synchronization.
        if self.train_vggt:
            self.vggt_backbone = vggt_backbone
        else:
            object.__setattr__(self, "vggt_backbone", vggt_backbone)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        if self.vggt_fp32_storage_bf16_compute:
            requested_model_dtype = torch_dtype
            # The VGGT checkpoint is loaded in FP32; avoid a BF16 round trip.
            # Action/MoT modules were constructed in model_dtype already.
            self.to(device=self.device)
            # `_apply` refreshes `self.torch_dtype` from model parameters. In
            # this opt-in path the first registered parameter may still be
            # FP32 before the Action/MoT conversion below, so retain the
            # caller's requested dtype explicitly.
            self.torch_dtype = requested_model_dtype
            self.mot.to(dtype=self.torch_dtype)
            if self.proprio_encoder is not None:
                self.proprio_encoder.to(dtype=self.torch_dtype)
        else:
            self.to(device=self.device, dtype=self.torch_dtype)
        self.vggt_backbone.train(self.training and self.train_vggt)

    @property
    def action_expert(self) -> ActionDiT:
        return self.mot.mixtures["action"]

    def reconstruction_loss_contract(self):
        """Official checkpoints encode scale semantics and the pinned objective."""
        config = getattr(self, "reconstruction_loss_config", None) or {}
        if config.get("backend", "legacy") != "official":
            return None  # Historical checkpoints have no loss contract.
        from .vggt_official_loss_adapter import OFFICIAL_COMMIT, official_config
        return {"backend": "official", "commit": OFFICIAL_COMMIT,
                "config": official_config(config.get("official")),
                "lambda_vggt_recon": self.lambda_vggt_recon}

    def _reconstruction_output_metadata(self):
        contract = self.reconstruction_loss_contract()
        if contract is None:
            return {}
        config = contract["config"]
        convention = ("arbitrary" if config["normalize_predictions"] else
                      "scene_normalized" if config["normalize_gt"] else "metric")
        return {"scale_convention": convention}

    def set_reconstruction_loss_progress(self, progress):
        """Set optimizer-step progress; no training_loss ABI changes for other models."""
        self._reconstruction_loss_progress = float(progress)

    def _reconstruction_loss_kwargs(self, sample, valid_mask=None):
        config = dict(self.reconstruction_loss_config)
        if config.get("backend", "legacy") == "official":
            synthetic = sample.get("vggt_recon_is_synthetic", False)
            if valid_mask is not None and isinstance(synthetic, torch.Tensor) and synthetic.ndim:
                synthetic = synthetic[valid_mask.to(synthetic.device)]
            config.update(schedule_progress=getattr(self, "_reconstruction_loss_progress", 0.0),
                          is_synthetic=synthetic)
        return config

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        backbone = getattr(self, "vggt_backbone", None)
        if backbone is not None and self._modules.get("vggt_backbone") is not backbone:
            backbone._apply(fn, recurse=recurse)
            backbone.eval()
        if backbone is not None and self.vggt_fp32_storage_bf16_compute:
            for name in ("aggregator", "va1", "va2"):
                module = getattr(backbone, name, None)
                if isinstance(module, nn.Module):
                    module.float()
        # CameraHead and DenseHead are deliberately kept in FP32.  Calling
        # ``self.to(dtype=bf16)`` during construction (or later through a
        # distributed wrapper) otherwise converts their parameters while the
        # upstream VGGT heads promote aggregated tokens to FP32, which makes
        # LayerNorm fail with a mixed input/parameter dtype.
        if self.reconstruction_loss_config is not None and backbone is not None:
            for head_name in ("camera_head", "dense_head"):
                head = getattr(backbone, head_name, None)
                if head is not None:
                    head.float()
        parameter = next(self.mot.parameters(), None)
        if parameter is None and backbone is not None:
            parameter = next(backbone.parameters(), None)
        if parameter is not None:
            self.device = parameter.device
            if parameter.is_floating_point():
                self.torch_dtype = parameter.dtype
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_vggt:
            self.vggt_backbone.eval()
        return self


    def _prepare_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
        *,
        batch_size: int,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim == 2 and not training:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1 and not training:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                "`context/context_mask` must be [B,L,D]/[B,L], got "
                f"{tuple(context.shape)} and {tuple(context_mask.shape)}."
            )
        if context.shape[0] != batch_size or context_mask.shape != context.shape[:2]:
            raise ValueError(
                "`context/context_mask` batch or sequence shape mismatch: "
                f"context={tuple(context.shape)}, mask={tuple(context_mask.shape)}, "
                f"batch_size={batch_size}."
            )
        if context.shape[2] != self.text_dim:
            raise ValueError(
                f"`context` last dim must be {self.text_dim}, got {context.shape[2]}."
            )
        context = context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        context_mask = context_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )

        if self.proprio_encoder is None:
            if proprio is not None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` disables proprio encoding."
                )
            return context, context_mask
        if proprio is None:
            raise ValueError("`proprio` is required when `proprio_dim` is enabled.")
        if training and proprio.ndim == 3:
            proprio = proprio[:, 0]
        elif not training and proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or tuple(proprio.shape) != (batch_size, self.proprio_dim):
            raise ValueError(
                f"`proprio` must resolve to [{batch_size},{self.proprio_dim}], "
                f"got {tuple(proprio.shape)}."
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=self.torch_dtype)
        ).unsqueeze(1)
        proprio_mask = torch.ones(
            (batch_size, 1), dtype=torch.bool, device=self.device
        )
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _predict_action(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        vggt_kv_by_layer,
    ) -> torch.Tensor:
        # Reconstruction backbones may promote their geometry features to
        # FP32 while retaining FP32 VGGT weights. The bridge itself consumes
        # the ActionDiT compute dtype, so normalize only its K/V boundary.
        if vggt_kv_by_layer is not None:
            vggt_kv_by_layer = {
                layer: {
                    **entry,
                    "k": entry["k"].to(dtype=self.torch_dtype),
                    "v": entry["v"].to(dtype=self.torch_dtype),
                }
                for layer, entry in vggt_kv_by_layer.items()
            }
        pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        seq_len = int(pre["tokens"].shape[1])
        attention_mask = torch.ones(
            (seq_len, seq_len), dtype=torch.bool, device=pre["tokens"].device
        )
        output = self.mot(
            embeds_all={"action": pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"action": pre["freqs"]},
            context_all={
                "action": {
                    "context": pre["context"],
                    "mask": pre["context_mask"],
                }
            },
            t_mod_all={"action": pre["t_mod"]},
            vggt_kv_by_layer=vggt_kv_by_layer,
        )
        return self.action_expert.post_dit(output["action"], pre)





    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)


__all__ = ["ActionFlowBase"]
