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


class ActionVGGTFastWAM(nn.Module):
    """Action-only flow model backed by frozen or trainable VGGT K/V memory."""

    CHECKPOINT_FORMAT = "action_vggt_fastwam"
    CHECKPOINT_VERSION = 2
    LEGACY_CHECKPOINT_VERSION = 1
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
                "ActionVGGTFastWAM requires an action-only MoT with expert_order=('action',)."
            )
        if mot.mixtures["action"] is not action_expert:
            raise ValueError("`action_expert` must be the action module owned by `mot`.")
        if mot.vggt_bridge is None:
            raise ValueError("ActionVGGTFastWAM requires `mot.vggt_bridge`.")
        if mot.vggt_bridge.target_experts != frozenset({"action"}):
            raise ValueError("ActionVGGTFastWAM bridge must target only the action expert.")

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

    def _capture_vggt(
        self,
        vggt_images: torch.Tensor,
        *,
        batch_size: Optional[int] = None,
    ):
        if vggt_images is None:
            raise ValueError("ActionVGGTFastWAM requires `vggt_images`.")
        validate_vggt_images(vggt_images, batch_size=batch_size)
        images = vggt_images.to(
            device=self.device,
            dtype=(torch.float32 if self.vggt_fp32_storage_bf16_compute else self.torch_dtype),
            non_blocking=True,
        )
        aggregator = getattr(self.vggt_backbone, "aggregator", self.vggt_backbone)
        parameter = next(aggregator.parameters(), None)
        if parameter is not None and images.dtype != parameter.dtype:
            images = images.to(dtype=parameter.dtype)
        return self.vggt_backbone(images)

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

    def training_loss(self, sample, vggt_kv_by_layer=None):
        if self.train_vggt and vggt_kv_by_layer is not None:
            raise ValueError(
                "Trainable VGGT requires live backbone capture during training; "
                "precomputed `vggt_kv_by_layer` is not supported."
            )
        required = {"action", "context", "context_mask", "vggt_images"}
        if self.reconstruction_loss_config is not None:
            required.update({
                "vggt_recon_depth", "vggt_recon_depth_valid",
                "vggt_recon_intrinsics", "vggt_recon_extrinsics",
            })
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(f"ActionVGGTFastWAM training sample is missing {missing}.")
        action = sample["action"]
        if action.ndim != 3 or action.shape[2] != self.action_expert.action_dim:
            raise ValueError(
                "`action` must be [B,T,action_dim], got "
                f"{tuple(action.shape)} with action_dim={self.action_expert.action_dim}."
            )
        action = action.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        batch_size, action_horizon, _ = action.shape
        context, context_mask = self._prepare_context(
            sample["context"],
            sample["context_mask"],
            sample.get("proprio"),
            batch_size=batch_size,
            training=True,
        )

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            if tuple(action_is_pad.shape) != (batch_size, action_horizon):
                raise ValueError(
                    f"`action_is_pad` must be [{batch_size},{action_horizon}], "
                    f"got {tuple(action_is_pad.shape)}."
                )
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )

        reconstruction = None
        if self.reconstruction_loss_config is not None:
            if vggt_kv_by_layer is not None:
                raise ValueError("Reconstruction training requires live VGGT capture.")
            reconstruction = self.vggt_backbone.forward_with_reconstruction(
                sample["vggt_images"].to(self.device, self.torch_dtype),
            )
            memory = reconstruction["kv_by_layer"]
        else:
            memory = (
                vggt_kv_by_layer
                if vggt_kv_by_layer is not None
                else self._capture_vggt(sample["vggt_images"], batch_size=batch_size)
            )
        noise = torch.randn_like(action)
        timestep = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise, timestep
        )
        target_action = self.train_action_scheduler.training_target(
            action, noise, timestep
        )
        pred_action = self._predict_action(
            noisy_action, timestep, context, context_mask, memory
        )

        token_loss = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is None:
            per_sample_loss = token_loss.mean(dim=1)
        else:
            valid = (~action_is_pad).to(dtype=token_loss.dtype)
            per_sample_loss = (token_loss * valid).sum(dim=1) / valid.sum(
                dim=1
            ).clamp(min=1.0)
        weight = self.train_action_scheduler.training_weight(timestep).to(
            device=per_sample_loss.device, dtype=per_sample_loss.dtype
        )
        loss_action = (per_sample_loss * weight).mean()
        loss_dict = {"loss_action": loss_action.detach()}
        if reconstruction is not None:
            recon_loss, recon_dict = compute_vggt_reconstruction_loss(
                depth=reconstruction["depth"],
                depth_conf=reconstruction["depth_conf"],
                pose_enc=reconstruction["pose_enc"],
                gt_depth=sample["vggt_recon_depth"].to(self.device),
                gt_valid=sample["vggt_recon_depth_valid"].to(self.device),
                gt_intrinsics=sample["vggt_recon_intrinsics"].to(self.device),
                gt_extrinsics=sample["vggt_recon_extrinsics"].to(self.device),
                **self._reconstruction_loss_kwargs(sample),
            )
            recon_loss = self.lambda_vggt_recon * recon_loss
            loss_action = loss_action + recon_loss
            loss_dict.update(recon_dict)
            loss_dict["loss_vggt_recon"] = recon_loss.detach()
        return loss_action, loss_dict

    @torch.no_grad()
    def infer_action(
        self,
        *,
        action_horizon: int,
        vggt_images: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        return_reconstruction: bool = False,
    ) -> torch.Tensor | dict[str, object]:
        self.eval()
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}.")
        context_batch = 1 if context.ndim == 2 else int(context.shape[0])
        context, context_mask = self._prepare_context(
            context,
            context_mask,
            proprio,
            batch_size=context_batch,
            training=False,
        )
        reconstruction = None
        if return_reconstruction:
            if self.reconstruction_loss_config is None:
                raise ValueError(
                    "Reconstruction output was requested from a model without "
                    "reconstruction heads."
                )
            validate_vggt_images(vggt_images, batch_size=context_batch)
            images = vggt_images.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            reconstruction = self.vggt_backbone.forward_with_reconstruction(images)
            memory = reconstruction["kv_by_layer"]
        else:
            memory = self._capture_vggt(
                vggt_images,
                batch_size=context_batch,
            )

        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        action = torch.randn(
            (context_batch, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        for timestep, delta in zip(timesteps, deltas):
            batch_timestep = timestep.expand(context_batch)
            pred_action = self._predict_action(
                action, batch_timestep, context, context_mask, memory
            )
            action = self.infer_action_scheduler.step(pred_action, delta, action)
        action = action.detach().to(device="cpu", dtype=torch.float32)
        if reconstruction is None:
            return action
        # Do not expose the large aggregated-token/KV tensors to the rollout
        # process.  Moving only head outputs to CPU keeps visualization from
        # retaining GPU memory across replans.
        return {
            "action": action,
            "reconstruction": {
                **{key: reconstruction[key].detach().to(device="cpu", dtype=torch.float32)
                   for key in ("depth", "depth_conf", "pose_enc")},
                **self._reconstruction_output_metadata(),
            },
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        if optimizer is not None:
            raise ValueError(
                "ActionVGGTFastWAM weight checkpoints do not contain optimizer state; "
                "use the Accelerate training-state checkpoint."
            )
        checkpoint_version = (
            self.CHECKPOINT_VERSION
            if self.train_vggt
            else self.LEGACY_CHECKPOINT_VERSION
        )
        payload = {
            "format": self.CHECKPOINT_FORMAT,
            "version": checkpoint_version,
            "action_expert": self.action_expert.state_dict(),
            "vggt_bridge": self.mot.vggt_bridge.state_dict(),
            "proprio_encoder": (
                None
                if self.proprio_encoder is None
                else self.proprio_encoder.state_dict()
            ),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.train_vggt:
            payload["vggt_aggregator"] = self.vggt_backbone.aggregator.state_dict()
        if getattr(self, "reconstruction_loss_config", None) and getattr(self.vggt_backbone, "camera_head", None) is not None:
            payload["camera_head"] = self.vggt_backbone.camera_head.state_dict()
            payload["dense_head"] = self.vggt_backbone.dense_head.state_dict()
        if self.reconstruction_loss_contract() is not None:
            payload["reconstruction_loss"] = self.reconstruction_loss_contract()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        if optimizer is not None:
            raise ValueError(
                "ActionVGGTFastWAM weight checkpoints do not contain optimizer state; "
                "use the Accelerate training-state checkpoint."
            )
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid ActionVGGTFastWAM checkpoint: {path}.")
        if payload.get("reconstruction_loss") != self.reconstruction_loss_contract():
            raise ValueError("Reconstruction loss contract mismatch; legacy/official weights must use their matching config.")
        base_expected_keys = {
            "format",
            "version",
            "action_expert",
            "vggt_bridge",
            "proprio_encoder",
            "step",
            "torch_dtype",
        }
        version = payload.get("version")
        expected_keys = set(base_expected_keys)
        if self.reconstruction_loss_contract() is not None:
            expected_keys.add("reconstruction_loss")
        if version == self.CHECKPOINT_VERSION:
            expected_keys.add("vggt_aggregator")
        if getattr(self, "reconstruction_loss_config", None):
            expected_keys.update({"camera_head", "dense_head"})
        missing = sorted(expected_keys - set(payload))
        unexpected = sorted(set(payload) - expected_keys)
        if missing or unexpected:
            raise ValueError(
                "ActionVGGTFastWAM checkpoint key mismatch: "
                f"missing={missing}, unexpected={unexpected}. Legacy FastWAM/VGGTFastWAM "
                "checkpoints are not supported."
            )
        if payload.get("format") != self.CHECKPOINT_FORMAT or version not in {
            self.LEGACY_CHECKPOINT_VERSION,
            self.CHECKPOINT_VERSION,
        }:
            raise ValueError(
                "Unsupported ActionVGGTFastWAM checkpoint format/version: "
                f"{payload['format']!r}/{payload['version']!r}."
            )
        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.mot.vggt_bridge.load_state_dict(payload["vggt_bridge"], strict=True)
        if version == self.CHECKPOINT_VERSION:
            try:
                self.vggt_backbone.aggregator.load_state_dict(
                    payload["vggt_aggregator"],
                    strict=True,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"vggt_aggregator checkpoint mismatch: {exc}"
                ) from exc
        elif self.train_vggt:
            warnings.warn(
                "Version-1 ActionVGGTFastWAM checkpoint has no VGGT weights; "
                "keeping the Aggregator initialization from model construction.",
                UserWarning,
                stacklevel=2,
            )
        if getattr(self, "reconstruction_loss_config", None):
            for name, module in (
                ("camera_head", self.vggt_backbone.camera_head),
                ("dense_head", self.vggt_backbone.dense_head),
            ):
                if module is None:
                    raise RuntimeError(
                        f"Checkpoint requires reconstruction module `{name}`."
                    )
                module.load_state_dict(payload[name], strict=True)
        proprio_state = payload["proprio_encoder"]
        if self.proprio_encoder is None:
            if proprio_state is not None:
                raise ValueError(
                    "Checkpoint contains proprio weights but the model disables proprio."
                )
        else:
            if not isinstance(proprio_state, dict):
                raise ValueError(
                    "Checkpoint is missing proprio weights required by this model."
                )
            self.proprio_encoder.load_state_dict(proprio_state, strict=True)
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)


__all__ = ["ActionVGGTFastWAM"]
