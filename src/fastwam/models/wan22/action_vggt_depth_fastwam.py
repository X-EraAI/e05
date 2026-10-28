from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_dit import ActionDiT
from .depth_dit import DepthDiT
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .vggt_mot import FrozenVGGTBackbone, validate_vggt_images


class ActionVGGTDepthFastWAM(nn.Module):
    """Joint action/depth flow model conditioned on current multi-view VGGT K/V."""

    CHECKPOINT_FORMAT = "action_vggt_depth_fastwam"
    CHECKPOINT_VERSION = 1
    DEPTH_NORMALIZATION_VERSION = 1
    evaluation_mode = "action_only"
    requires_vggt_future_images = False
    requires_vggt_teacher_features = False
    requires_depth_targets = True

    def __init__(
        self,
        *,
        action_expert: ActionDiT,
        depth_expert: DepthDiT,
        mot: MoT,
        vggt_backbone: FrozenVGGTBackbone,
        text_dim: int,
        expected_num_views: int,
        depth_reads_action: bool = True,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        train_vggt: bool = True,
        action_only_inference: bool = False,
        loss_lambda_depth: float = 1.0,
        train_shift: float = 5.0,
        infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        if tuple(mot.expert_order) != ("depth", "action"):
            raise ValueError(
                "ActionVGGTDepthFastWAM requires expert_order=('depth', 'action')."
            )
        if mot.mixtures["action"] is not action_expert:
            raise ValueError("`action_expert` must be the action module owned by `mot`.")
        if mot.mixtures["depth"] is not depth_expert:
            raise ValueError("`depth_expert` must be the depth module owned by `mot`.")
        if mot.vggt_bridge is None:
            raise ValueError("ActionVGGTDepthFastWAM requires `mot.vggt_bridge`.")
        if mot.vggt_bridge.target_experts != frozenset({"action", "depth"}):
            raise ValueError("The VGGT bridge must target both action and depth experts.")
        if expected_num_views <= 0 or expected_num_views != depth_expert.num_views:
            raise ValueError(
                "`expected_num_views` must match DepthDiT.num_views: "
                f"{expected_num_views} vs {depth_expert.num_views}."
            )
        if loss_lambda_depth < 0:
            raise ValueError("`loss_lambda_depth` must be non-negative.")

        self.mot = mot
        self.dit = mot
        self.train_vggt = bool(train_vggt)
        self.action_only_inference = bool(action_only_inference)
        self.depth_reads_action = bool(depth_reads_action)
        self.text_dim = int(text_dim)
        self.expected_num_views = int(expected_num_views)
        self.loss_lambda_depth = float(loss_lambda_depth)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.proprio_encoder = (
            None
            if self.proprio_dim is None
            else nn.Linear(self.proprio_dim, self.text_dim, dtype=torch_dtype)
        )
        self.mot.vggt_bridge.detach_vggt_memory = not self.train_vggt

        self.train_flow_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(num_train_timesteps), shift=float(train_shift)
        )
        self.infer_flow_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(num_train_timesteps), shift=float(infer_shift)
        )

        if self.train_vggt:
            self.vggt_backbone = vggt_backbone
        else:
            object.__setattr__(self, "vggt_backbone", vggt_backbone)

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.to(device=self.device, dtype=self.torch_dtype)
        self.vggt_backbone.train(self.training and self.train_vggt)

    @property
    def action_expert(self) -> ActionDiT:
        return self.mot.mixtures["action"]

    @property
    def depth_expert(self) -> DepthDiT:
        return self.mot.mixtures["depth"]

    @property
    def depth_future_offsets(self) -> tuple[int, ...]:
        return self.depth_expert.future_offsets

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        backbone = getattr(self, "vggt_backbone", None)
        if backbone is not None and not self.train_vggt:
            backbone._apply(fn, recurse=recurse)
            backbone.eval()
        parameter = next(self.mot.parameters(), None)
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

    def _validate_views(self, images: torch.Tensor, *, batch_size: int) -> None:
        validate_vggt_images(images, batch_size=batch_size)
        if int(images.shape[1]) != self.expected_num_views:
            raise ValueError(
                f"Expected {self.expected_num_views} VGGT views, got {images.shape[1]}."
            )
        if tuple(images.shape[-2:]) != (
            self.depth_expert.depth_height,
            self.depth_expert.depth_width,
        ):
            raise ValueError(
                "Current VGGT image layout must match the depth target cache: "
                f"expected {(self.depth_expert.depth_height, self.depth_expert.depth_width)}, "
                f"got {tuple(images.shape[-2:])}."
            )

    def _capture_vggt_kv(self, images: torch.Tensor, *, batch_size: int):
        self._validate_views(images, batch_size=batch_size)
        return self.vggt_backbone(
            images.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
        )

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
            raise ValueError("`context/context_mask` must be [B,L,D]/[B,L].")
        if context.shape[0] != batch_size or context_mask.shape != context.shape[:2]:
            raise ValueError("`context/context_mask` batch or sequence shape mismatch.")
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
                    "`proprio` was provided but `proprio_dim=None` disables encoding."
                )
            return context, context_mask
        if proprio is None:
            raise ValueError("`proprio` is required when `proprio_dim` is enabled.")
        if training and proprio.ndim == 3:
            proprio = proprio[:, 0]
        elif not training and proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or tuple(proprio.shape) != (
            batch_size,
            self.proprio_dim,
        ):
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

    @staticmethod
    def build_mot_mask(
        depth_seq_len: int,
        action_seq_len: int,
        *,
        depth_reads_action: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Build asymmetric visibility in registered ``[depth, action]`` order."""
        if depth_seq_len <= 0 or action_seq_len <= 0:
            raise ValueError("MoT expert sequence lengths must be positive.")
        total = depth_seq_len + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:depth_seq_len, :depth_seq_len] = True
        if depth_reads_action:
            mask[:depth_seq_len, depth_seq_len:] = True
        mask[depth_seq_len:, depth_seq_len:] = True
        return mask

    def _predict_joint_flow(
        self,
        *,
        noisy_action: torch.Tensor,
        noisy_depth: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        vggt_kv_by_layer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        depth_pre = self.depth_expert.pre_dit(
            action_tokens=noisy_depth,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        depth_seq_len = int(depth_pre["tokens"].shape[1])
        action_seq_len = int(action_pre["tokens"].shape[1])
        attention_mask = self.build_mot_mask(
            depth_seq_len,
            action_seq_len,
            depth_reads_action=self.depth_reads_action,
            device=depth_pre["tokens"].device,
        )
        output = self.mot(
            embeds_all={
                "depth": depth_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "depth": depth_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "depth": {
                    "context": depth_pre["context"],
                    "mask": depth_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "depth": depth_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            vggt_kv_by_layer=vggt_kv_by_layer,
            isolated_expert_attention=not self.depth_reads_action,
        )
        return (
            self.depth_expert.post_dit(output["depth"], depth_pre),
            self.action_expert.post_dit(output["action"], action_pre),
        )

    def _predict_action_flow(
        self,
        *,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        vggt_kv_by_layer,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        action_seq_len = int(action_pre["tokens"].shape[1])
        output = self.mot(
            embeds_all={"action": action_pre["tokens"]},
            attention_mask=torch.ones(
                (action_seq_len, action_seq_len),
                dtype=torch.bool,
                device=action_pre["tokens"].device,
            ),
            freqs_all={"action": action_pre["freqs"]},
            context_all={
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                }
            },
            t_mod_all={"action": action_pre["t_mod"]},
            vggt_kv_by_layer=vggt_kv_by_layer,
            isolated_expert_attention=True,
            active_experts=("action",),
        )
        return self.action_expert.post_dit(output["action"], action_pre)

    @staticmethod
    def _masked_action_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        token_loss = F.mse_loss(
            pred.float(), target.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is None:
            return token_loss.mean(dim=1)
        valid = (~action_is_pad).to(device=token_loss.device, dtype=token_loss.dtype)
        return (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def _validate_depth_batch(
        self,
        target: torch.Tensor,
        weight: torch.Tensor,
        future_is_pad: torch.Tensor,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (
            batch_size,
            len(self.depth_future_offsets),
            self.expected_num_views,
            self.depth_expert.patch_grid_height
            * self.depth_expert.patch_grid_width,
            self.depth_expert.action_dim,
        )
        if tuple(target.shape) != expected or tuple(weight.shape) != expected:
            raise ValueError(
                "Depth target/weight must match configured [B,K,V,N,P^2] layout "
                f"{expected}, got {tuple(target.shape)} and {tuple(weight.shape)}."
            )
        if tuple(future_is_pad.shape) != expected[:2]:
            raise ValueError(
                f"`depth_future_is_pad` must be {expected[:2]}, got {tuple(future_is_pad.shape)}."
            )
        target = target.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        ).flatten(1, 3)
        weight = weight.to(
            device=self.device, dtype=torch.float32, non_blocking=True
        ).flatten(1, 3)
        future_is_pad = future_is_pad.to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        if not bool(torch.isfinite(target).all().item()) or not bool(
            torch.isfinite(weight).all().item()
        ):
            raise ValueError("Depth target/weight contains non-finite values.")
        if bool((weight < 0).any().item()):
            raise ValueError("Depth confidence weights must be non-negative.")
        return target, weight, future_is_pad

    def training_loss(self, sample, vggt_kv_by_layer=None):
        if self.action_only_inference:
            raise RuntimeError("An action-only inference model cannot be used for training.")
        if self.train_vggt and vggt_kv_by_layer is not None:
            raise ValueError(
                "Trainable VGGT requires live K/V capture during training."
            )
        required = {
            "action",
            "context",
            "context_mask",
            "vggt_images",
            "depth_target",
            "depth_weight",
            "depth_future_is_pad",
        }
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(
                f"ActionVGGTDepthFastWAM training sample is missing {missing}."
            )
        action = sample["action"]
        if action.ndim != 3 or action.shape[2] != self.action_expert.action_dim:
            raise ValueError(
                "`action` must be [B,T,action_dim], got "
                f"{tuple(action.shape)}."
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
                    f"`action_is_pad` must be [{batch_size},{action_horizon}]."
                )
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        depth, depth_weight, depth_future_is_pad = self._validate_depth_batch(
            sample["depth_target"],
            sample["depth_weight"],
            sample["depth_future_is_pad"],
            batch_size=batch_size,
        )
        memory = (
            vggt_kv_by_layer
            if vggt_kv_by_layer is not None
            else self._capture_vggt_kv(sample["vggt_images"], batch_size=batch_size)
        )

        timestep = self.train_flow_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        action_noise = torch.randn_like(action)
        depth_noise = torch.randn_like(depth)
        noisy_action = self.train_flow_scheduler.add_noise(
            action, action_noise, timestep
        )
        noisy_depth = self.train_flow_scheduler.add_noise(
            depth, depth_noise, timestep
        )
        target_action = self.train_flow_scheduler.training_target(
            action, action_noise, timestep
        )
        target_depth = self.train_flow_scheduler.training_target(
            depth, depth_noise, timestep
        )
        pred_depth, pred_action = self._predict_joint_flow(
            noisy_action=noisy_action,
            noisy_depth=noisy_depth,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            vggt_kv_by_layer=memory,
        )

        flow_weight = self.train_flow_scheduler.training_weight(timestep).to(
            device=self.device, dtype=torch.float32
        )
        action_loss_per_sample = self._masked_action_loss(
            pred_action, target_action, action_is_pad
        )
        loss_action = (action_loss_per_sample * flow_weight).mean()

        tokens_per_horizon = (
            self.expected_num_views
            * self.depth_expert.patch_grid_height
            * self.depth_expert.patch_grid_width
        )
        horizon_valid = (~depth_future_is_pad).repeat_interleave(
            tokens_per_horizon, dim=1
        ).unsqueeze(-1)
        effective_weight = depth_weight * horizon_valid.to(depth_weight.dtype)
        depth_element_loss = F.mse_loss(
            pred_depth.float(), target_depth.float(), reduction="none"
        )
        weight_sum = effective_weight.sum(dim=(1, 2))
        depth_loss_per_sample = (
            (depth_element_loss * effective_weight).sum(dim=(1, 2))
            / weight_sum.clamp(min=1.0)
        )
        has_depth = (weight_sum > 0).to(depth_loss_per_sample.dtype)
        loss_depth = (
            depth_loss_per_sample * flow_weight * has_depth
        ).sum() / has_depth.sum().clamp(min=1.0)

        loss_total = loss_action + self.loss_lambda_depth * loss_depth
        return loss_total, {
            "loss_action": loss_action.detach(),
            "loss_depth": self.loss_lambda_depth * loss_depth.detach(),
        }

    @torch.no_grad()
    def infer_joint(
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
    ) -> dict[str, torch.Tensor]:
        if self.action_only_inference:
            raise RuntimeError("Joint inference is disabled in action-only mode.")
        self.eval()
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}.")
        batch_size = 1 if context.ndim == 2 else int(context.shape[0])
        context, context_mask = self._prepare_context(
            context,
            context_mask,
            proprio,
            batch_size=batch_size,
            training=False,
        )
        memory = self._capture_vggt_kv(vggt_images, batch_size=batch_size)
        action_generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        depth_generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed + 1)
        )
        action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        depth = torch.randn(
            (
                batch_size,
                self.depth_expert.depth_sequence_length,
                self.depth_expert.action_dim,
            ),
            generator=depth_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        timesteps, deltas = self.infer_flow_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        for timestep, delta in zip(timesteps, deltas):
            pred_depth, pred_action = self._predict_joint_flow(
                noisy_action=action,
                noisy_depth=depth,
                timestep=timestep.expand(batch_size),
                context=context,
                context_mask=context_mask,
                vggt_kv_by_layer=memory,
            )
            action = self.infer_flow_scheduler.step(pred_action, delta, action)
            depth = self.infer_flow_scheduler.step(pred_depth, delta, depth)
        return {
            "action": action.detach().to(device="cpu", dtype=torch.float32),
            "depth": self.depth_expert.unpatchify(depth)
            .detach()
            .to(device="cpu", dtype=torch.float32),
        }

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
        **_ignored,
    ) -> torch.Tensor:
        self.eval()
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}.")
        batch_size = 1 if context.ndim == 2 else int(context.shape[0])
        context, context_mask = self._prepare_context(
            context,
            context_mask,
            proprio,
            batch_size=batch_size,
            training=False,
        )
        memory = self._capture_vggt_kv(vggt_images, batch_size=batch_size)
        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        timesteps, deltas = self.infer_flow_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        for timestep, delta in zip(timesteps, deltas):
            pred_action = self._predict_action_flow(
                noisy_action=action,
                timestep=timestep.expand(batch_size),
                context=context,
                context_mask=context_mask,
                vggt_kv_by_layer=memory,
            )
            action = self.infer_flow_scheduler.step(pred_action, delta, action)
        return action.detach().to(device="cpu", dtype=torch.float32)

    def save_checkpoint(self, path, optimizer=None, step=None):
        if optimizer is not None:
            raise ValueError(
                "Weight checkpoints do not contain optimizer state; use Accelerate state."
            )
        payload = {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            "depth_normalization_version": self.DEPTH_NORMALIZATION_VERSION,
            "depth_future_offsets": self.depth_future_offsets,
            "depth_patch_size": self.depth_expert.depth_patch_size,
            "patch_grid_height": self.depth_expert.patch_grid_height,
            "patch_grid_width": self.depth_expert.patch_grid_width,
            "expected_num_views": self.expected_num_views,
            "depth_reads_action": self.depth_reads_action,
            "train_vggt": self.train_vggt,
            "action_expert": self.action_expert.state_dict(),
            "depth_expert": self.depth_expert.state_dict(),
            "vggt_bridge": self.mot.vggt_bridge.state_dict(),
            "vggt_aggregator": (
                self.vggt_backbone.aggregator.state_dict()
                if self.train_vggt
                else None
            ),
            "proprio_encoder": (
                None
                if self.proprio_encoder is None
                else self.proprio_encoder.state_dict()
            ),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        if optimizer is not None:
            raise ValueError(
                "Weight checkpoints do not contain optimizer state; use Accelerate state."
            )
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        expected_keys = {
            "format",
            "version",
            "depth_normalization_version",
            "depth_future_offsets",
            "depth_patch_size",
            "patch_grid_height",
            "patch_grid_width",
            "expected_num_views",
            "depth_reads_action",
            "train_vggt",
            "action_expert",
            "depth_expert",
            "vggt_bridge",
            "vggt_aggregator",
            "proprio_encoder",
            "step",
            "torch_dtype",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            actual = set(payload) if isinstance(payload, dict) else set()
            raise ValueError(
                "ActionVGGTDepthFastWAM checkpoint key mismatch: "
                f"missing={sorted(expected_keys - actual)}, "
                f"unexpected={sorted(actual - expected_keys)}."
            )
        if payload["format"] != self.CHECKPOINT_FORMAT or int(
            payload["version"]
        ) != self.CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported ActionVGGTDepthFastWAM checkpoint format/version: "
                f"{payload['format']!r}/{payload['version']!r}."
            )
        expected_metadata = {
            "depth_normalization_version": self.DEPTH_NORMALIZATION_VERSION,
            "depth_future_offsets": self.depth_future_offsets,
            "depth_patch_size": self.depth_expert.depth_patch_size,
            "patch_grid_height": self.depth_expert.patch_grid_height,
            "patch_grid_width": self.depth_expert.patch_grid_width,
            "expected_num_views": self.expected_num_views,
            "depth_reads_action": self.depth_reads_action,
            "train_vggt": self.train_vggt,
        }
        for key, expected in expected_metadata.items():
            actual = payload[key]
            if key == "depth_future_offsets":
                actual = tuple(int(value) for value in actual)
            if actual != expected:
                raise ValueError(
                    f"Checkpoint `{key}` mismatch: expected {expected}, got {actual}."
                )
        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.depth_expert.load_state_dict(payload["depth_expert"], strict=True)
        self.mot.vggt_bridge.load_state_dict(payload["vggt_bridge"], strict=True)
        if self.train_vggt:
            if not isinstance(payload["vggt_aggregator"], dict):
                raise ValueError("Trainable-VGGT checkpoint is missing Aggregator weights.")
            self.vggt_backbone.aggregator.load_state_dict(
                payload["vggt_aggregator"], strict=True
            )
        elif payload["vggt_aggregator"] is not None:
            raise ValueError("Frozen-VGGT model cannot load trainable Aggregator weights.")
        proprio_state = payload["proprio_encoder"]
        if self.proprio_encoder is None:
            if proprio_state is not None:
                raise ValueError("Checkpoint contains unexpected proprio weights.")
        else:
            if not isinstance(proprio_state, dict):
                raise ValueError("Checkpoint is missing required proprio weights.")
            self.proprio_encoder.load_state_dict(proprio_state, strict=True)
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)


__all__ = ["ActionVGGTDepthFastWAM"]
