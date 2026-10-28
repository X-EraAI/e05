from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_dit import ActionDiT
from .geoaware_projector import FrozenVGGTFeatureBackbone
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .vggt_mot import FrozenVGGTBackbone, validate_vggt_images


class ActionVGGTFastWAM4D(nn.Module):
    """Joint action/4D flow model with isolated MoT experts and shared VGGT memory.

    The registered ``vggt_backbone`` is the optionally trainable K/V source used by
    the shared-attention bridge. ``vggt_teacher`` is a separate, always-frozen
    Aggregator that produces stable layer-23 hidden features for the 4D objective.
    """

    CHECKPOINT_FORMAT = "action_vggt_fastwam_4d"
    CHECKPOINT_VERSION = 1
    evaluation_mode = "action_only"
    requires_vggt_future_images = True

    def __init__(
        self,
        *,
        action_expert: ActionDiT,
        four_d_expert: ActionDiT,
        mot: MoT,
        vggt_backbone: FrozenVGGTBackbone,
        vggt_teacher: Optional[FrozenVGGTFeatureBackbone],
        text_dim: int,
        feature_dim: int = 2048,
        feature_layer: int = 23,
        expected_num_views: int,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        train_vggt: bool = True,
        use_precomputed_teacher_features: bool = False,
        action_only_inference: bool = False,
        loss_lambda_4d: float = 1.0,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        four_d_train_shift: float = 5.0,
        four_d_infer_shift: float = 5.0,
        four_d_num_train_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        if tuple(mot.expert_order) != ("four_d", "action"):
            raise ValueError(
                "ActionVGGTFastWAM4D requires expert_order=('four_d', 'action')."
            )
        if mot.mixtures["action"] is not action_expert:
            raise ValueError("`action_expert` must be the action module owned by `mot`.")
        if mot.mixtures["four_d"] is not four_d_expert:
            raise ValueError("`four_d_expert` must be the four_d module owned by `mot`.")
        if mot.vggt_bridge is None:
            raise ValueError("ActionVGGTFastWAM4D requires `mot.vggt_bridge`.")
        if mot.vggt_bridge.target_experts != frozenset({"action", "four_d"}):
            raise ValueError("The VGGT bridge must target both action and four_d experts.")
        if feature_dim <= 0 or expected_num_views <= 0:
            raise ValueError("`feature_dim` and `expected_num_views` must be positive.")
        if four_d_expert.action_dim != int(feature_dim):
            raise ValueError(
                "4D expert input/output dim must match VGGT feature dim: "
                f"{four_d_expert.action_dim} vs {feature_dim}."
            )
        use_precomputed_teacher_features = bool(use_precomputed_teacher_features)
        action_only_inference = bool(action_only_inference)
        if action_only_inference:
            if vggt_teacher is not None:
                raise ValueError(
                    "Action-only inference mode must not construct a frozen VGGT teacher."
                )
        elif use_precomputed_teacher_features:
            if vggt_teacher is not None:
                raise ValueError(
                    "Precomputed teacher mode must not construct a frozen VGGT teacher."
                )
        else:
            if vggt_teacher is None:
                raise ValueError("Online teacher mode requires `vggt_teacher`.")
            if tuple(vggt_teacher.selected_layers) != (int(feature_layer),):
                raise ValueError(
                    "Frozen VGGT teacher must expose exactly the requested feature layer: "
                    f"selected={vggt_teacher.selected_layers}, requested={feature_layer}."
                )
            if vggt_teacher.trainable:
                raise ValueError("The g0/g32 VGGT feature teacher must remain frozen.")
        if loss_lambda_4d < 0:
            raise ValueError("`loss_lambda_4d` must be non-negative.")

        # Trainer compatibility: every trainable DiT/bridge parameter lives under
        # this single graph. The trainable VGGT backbone receives its own LR group.
        self.mot = mot
        self.dit = self.mot
        self.train_vggt = bool(train_vggt)
        self.use_precomputed_teacher_features = use_precomputed_teacher_features
        self.action_only_inference = action_only_inference
        self.requires_vggt_future_images = not use_precomputed_teacher_features
        self.requires_vggt_teacher_features = use_precomputed_teacher_features
        self.mot.vggt_bridge.detach_vggt_memory = not self.train_vggt
        self.text_dim = int(text_dim)
        self.feature_dim = int(feature_dim)
        self.feature_layer = int(feature_layer)
        self.expected_num_views = int(expected_num_views)
        self.loss_lambda_4d = float(loss_lambda_4d)
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
        self.train_four_d_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=four_d_num_train_timesteps,
            shift=four_d_train_shift,
        )
        self.infer_four_d_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=four_d_num_train_timesteps,
            shift=four_d_infer_shift,
        )

        # A trainable backbone must be registered for Accelerate/DeepSpeed. The
        # frozen teacher is deliberately kept outside the module tree so it is
        # absent from the optimizer and model checkpoint.
        if self.train_vggt:
            self.vggt_backbone = vggt_backbone
        else:
            object.__setattr__(self, "vggt_backbone", vggt_backbone)
        object.__setattr__(self, "vggt_teacher", vggt_teacher)

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.to(device=self.device, dtype=self.torch_dtype)
        self.vggt_backbone.train(self.training and self.train_vggt)
        if self.vggt_teacher is not None:
            self.vggt_teacher.eval()

    @property
    def action_expert(self) -> ActionDiT:
        return self.mot.mixtures["action"]

    @property
    def four_d_expert(self) -> ActionDiT:
        return self.mot.mixtures["four_d"]

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        backbone = getattr(self, "vggt_backbone", None)
        if backbone is not None and not self.train_vggt:
            backbone._apply(fn, recurse=recurse)
            backbone.eval()
        teacher = getattr(self, "vggt_teacher", None)
        if teacher is not None:
            teacher._apply(fn, recurse=recurse)
            teacher.eval()
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
        if self.vggt_teacher is not None:
            self.vggt_teacher.eval()
        return self

    def _validate_views(self, images: torch.Tensor, *, batch_size: int) -> None:
        validate_vggt_images(images, batch_size=batch_size)
        if int(images.shape[1]) != self.expected_num_views:
            raise ValueError(
                f"Expected {self.expected_num_views} VGGT views, got {images.shape[1]}."
            )

    def _capture_vggt_kv(self, images: torch.Tensor, *, batch_size: int):
        """Run the main VGGT on g0 images and return mapped K/V callbacks."""
        self._validate_views(images, batch_size=batch_size)
        images = images.to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        return self.vggt_backbone(images)

    def _teacher_features(self, images: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        """Return frozen layer-23 features as [B,V,S,2048]."""
        if self.vggt_teacher is None:
            raise RuntimeError(
                "Online VGGT teacher features are unavailable in precomputed or "
                "action-only inference mode."
            )
        self._validate_views(images, batch_size=batch_size)
        images = images.to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        features = self.vggt_teacher(images)[self.feature_layer]
        if features.ndim != 4 or features.shape[-1] != self.feature_dim:
            raise RuntimeError(
                "VGGT teacher feature must be [B,V,S,feature_dim], got "
                f"{tuple(features.shape)} with feature_dim={self.feature_dim}."
            )
        return features.detach()

    def _validate_cached_teacher_feature(
        self,
        feature: torch.Tensor,
        *,
        name: str,
        batch_size: int,
        allow_unbatched: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if not isinstance(feature, torch.Tensor) or not feature.is_floating_point():
            raise TypeError(f"`{name}` must be a floating-point tensor.")
        if allow_unbatched and feature.ndim == 2 and batch_size == 1:
            feature = feature.unsqueeze(0)
        if feature.ndim != 3 or feature.shape[0] != batch_size:
            raise ValueError(
                f"`{name}` must be [{batch_size},N,{self.feature_dim}], got "
                f"{tuple(feature.shape)}."
            )
        if feature.shape[2] != self.feature_dim:
            raise ValueError(
                f"`{name}` feature dim must be {self.feature_dim}, got {feature.shape[2]}."
            )
        num_tokens = int(feature.shape[1])
        if num_tokens <= 0 or num_tokens % self.expected_num_views != 0:
            raise ValueError(
                f"`{name}` token count {num_tokens} is not divisible by "
                f"expected_num_views={self.expected_num_views}."
            )
        shape = (self.expected_num_views, num_tokens // self.expected_num_views)
        return (
            feature.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            ).contiguous(),
            shape,
        )

    def _cached_teacher_pair(
        self,
        g0: torch.Tensor,
        g32: torch.Tensor,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        g0, shape = self._validate_cached_teacher_feature(
            g0, name="vggt_teacher_g0", batch_size=batch_size
        )
        g32, future_shape = self._validate_cached_teacher_feature(
            g32, name="vggt_teacher_g32", batch_size=batch_size
        )
        if g0.shape != g32.shape or shape != future_shape:
            raise ValueError(
                "Cached VGGT g0/g32 shapes must match, got "
                f"{tuple(g0.shape)} and {tuple(g32.shape)}."
            )
        return g0, g32, shape

    def _capture_teacher_pair(
        self,
        current_images: torch.Tensor,
        future_images: torch.Tensor,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Extract g0/g32 in one frozen teacher call.

        Input images are [B,V,3,H,W]. Concatenating on batch gives [2B,V,3,H,W];
        the teacher result [2B,V,S,2048] is split and flattened to
        [B,V*S,2048] for the 4D DiT token sequence.
        """
        self._validate_views(current_images, batch_size=batch_size)
        self._validate_views(future_images, batch_size=batch_size)
        if tuple(current_images.shape[1:]) != tuple(future_images.shape[1:]):
            raise ValueError(
                "Current/future VGGT image shapes must match, got "
                f"{tuple(current_images.shape)} and {tuple(future_images.shape)}."
            )
        combined = torch.cat([current_images, future_images], dim=0)
        features = self._teacher_features(combined, batch_size=2 * batch_size)
        g0, g32 = features.split(batch_size, dim=0)
        num_views, tokens_per_view = int(g0.shape[1]), int(g0.shape[2])
        return (
            g0.flatten(1, 2).contiguous(),
            g32.flatten(1, 2).contiguous(),
            (num_views, tokens_per_view),
        )

    def _capture_teacher_g0(
        self,
        current_images: torch.Tensor,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        features = self._teacher_features(current_images, batch_size=batch_size)
        shape = (int(features.shape[1]), int(features.shape[2]))
        return features.flatten(1, 2).contiguous(), shape

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
    def _build_isolated_mot_mask(
        four_d_seq_len: int,
        action_seq_len: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Build block-diagonal [four_d, action] visibility.

        Both diagonal blocks are fully visible. Off-diagonal blocks remain
        False, so neither expert can read the other expert's K/V tokens.
        """
        if four_d_seq_len <= 0 or action_seq_len <= 0:
            raise ValueError("MoT expert sequence lengths must be positive.")
        total = four_d_seq_len + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:four_d_seq_len, :four_d_seq_len] = True
        mask[four_d_seq_len:, four_d_seq_len:] = True
        return mask

    def _predict_joint_flow(
        self,
        *,
        noisy_action: torch.Tensor,
        action_timestep: torch.Tensor,
        noisy_g32: torch.Tensor,
        four_d_timestep: torch.Tensor,
        g0: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        vggt_kv_by_layer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict g32/action velocities through isolated experts.

        Tensor flow:
          g0, noisy_g32: [B,N,2048]
          four_d_input:  [B,2N,2048] = concat([g0, noisy_g32], seq)
          action_input:  [B,Ta,A]
          MoT output:    four_d [B,2N,2048], action [B,Ta,A]
          returned g32 velocity is the suffix [B,N,2048].
        """
        if g0.shape != noisy_g32.shape:
            raise ValueError(
                f"g0/noisy_g32 shapes must match, got {g0.shape} and {noisy_g32.shape}."
            )
        four_d_input = torch.cat([g0, noisy_g32], dim=1)
        four_d_pre = self.four_d_expert.pre_dit(
            action_tokens=four_d_input,
            timestep=four_d_timestep,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=action_timestep,
            context=context,
            context_mask=context_mask,
        )
        four_d_seq_len = int(four_d_pre["tokens"].shape[1])
        attention_mask = self._build_isolated_mot_mask(
            four_d_seq_len,
            int(action_pre["tokens"].shape[1]),
            device=four_d_pre["tokens"].device,
        )
        output = self.mot(
            embeds_all={
                "four_d": four_d_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "four_d": four_d_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "four_d": {
                    "context": four_d_pre["context"],
                    "mask": four_d_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "four_d": four_d_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            vggt_kv_by_layer=vggt_kv_by_layer,
            isolated_expert_attention=True,
        )
        num_feature_tokens = int(g0.shape[1])
        pred_g32 = self.four_d_expert.post_dit(
            output["four_d"][:, -num_feature_tokens:], four_d_pre
        )
        pred_action = self.action_expert.post_dit(output["action"], action_pre)
        return pred_g32, pred_action

    def _predict_action_flow(
        self,
        *,
        noisy_action: torch.Tensor,
        action_timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        vggt_kv_by_layer,
    ) -> torch.Tensor:
        """Predict action velocity without executing the isolated 4D expert."""
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=action_timestep,
            context=context,
            context_mask=context_mask,
        )
        action_seq_len = int(action_pre["tokens"].shape[1])
        attention_mask = torch.ones(
            (action_seq_len, action_seq_len),
            dtype=torch.bool,
            device=action_pre["tokens"].device,
        )
        output = self.mot(
            embeds_all={"action": action_pre["tokens"]},
            attention_mask=attention_mask,
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

    def training_loss(self, sample, vggt_kv_by_layer=None):
        if self.action_only_inference:
            raise RuntimeError(
                "An action-only inference model cannot be used for training."
            )
        if self.train_vggt and vggt_kv_by_layer is not None:
            raise ValueError(
                "Trainable VGGT requires live K/V capture during training; "
                "precomputed memory is unsupported."
            )
        required = {
            "action",
            "context",
            "context_mask",
            "vggt_images",
        }
        if self.use_precomputed_teacher_features:
            required.update({"vggt_teacher_g0", "vggt_teacher_g32"})
        else:
            required.add("vggt_future_images")
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(f"ActionVGGTFastWAM4D training sample is missing {missing}.")

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

        memory = (
            vggt_kv_by_layer
            if vggt_kv_by_layer is not None
            else self._capture_vggt_kv(sample["vggt_images"], batch_size=batch_size)
        )
        if self.use_precomputed_teacher_features:
            g0, g32, _ = self._cached_teacher_pair(
                sample["vggt_teacher_g0"],
                sample["vggt_teacher_g32"],
                batch_size=batch_size,
            )
        else:
            g0, g32, _ = self._capture_teacher_pair(
                sample["vggt_images"],
                sample["vggt_future_images"],
                batch_size=batch_size,
            )

        action_noise = torch.randn_like(action)
        action_timestep = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, action_noise, action_timestep
        )
        target_action = self.train_action_scheduler.training_target(
            action, action_noise, action_timestep
        )

        four_d_noise = torch.randn_like(g32)
        four_d_timestep = self.train_four_d_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=g32.dtype
        )
        noisy_g32 = self.train_four_d_scheduler.add_noise(
            g32, four_d_noise, four_d_timestep
        )
        target_g32 = self.train_four_d_scheduler.training_target(
            g32, four_d_noise, four_d_timestep
        )

        pred_g32, pred_action = self._predict_joint_flow(
            noisy_action=noisy_action,
            action_timestep=action_timestep,
            noisy_g32=noisy_g32,
            four_d_timestep=four_d_timestep,
            g0=g0,
            context=context,
            context_mask=context_mask,
            vggt_kv_by_layer=memory,
        )

        action_loss_per_sample = self._masked_action_loss(
            pred_action, target_action, action_is_pad
        )
        action_weight = self.train_action_scheduler.training_weight(
            action_timestep
        ).to(device=action.device, dtype=action_loss_per_sample.dtype)
        loss_action = (action_loss_per_sample * action_weight).mean()

        four_d_loss_per_sample = F.mse_loss(
            pred_g32.float(), target_g32.float(), reduction="none"
        ).mean(dim=(1, 2))
        four_d_weight = self.train_four_d_scheduler.training_weight(
            four_d_timestep
        ).to(device=g32.device, dtype=four_d_loss_per_sample.dtype)
        future_is_pad = sample.get("vggt_future_is_pad")
        if future_is_pad is None and sample.get("image_is_pad") is not None:
            future_is_pad = sample["image_is_pad"][:, -1]
        if future_is_pad is None:
            future_valid = torch.ones(
                (batch_size,), dtype=four_d_loss_per_sample.dtype, device=self.device
            )
        else:
            if future_is_pad.ndim == 2 and future_is_pad.shape[1] == 1:
                future_is_pad = future_is_pad[:, 0]
            if tuple(future_is_pad.shape) != (batch_size,):
                raise ValueError(
                    f"`vggt_future_is_pad` must resolve to [{batch_size}], "
                    f"got {tuple(future_is_pad.shape)}."
                )
            future_valid = (~future_is_pad.to(device=self.device, dtype=torch.bool)).to(
                dtype=four_d_loss_per_sample.dtype
            )
        loss_4d = (
            four_d_loss_per_sample * four_d_weight * future_valid
        ).sum() / future_valid.sum().clamp(min=1.0)

        loss_total = loss_action + self.loss_lambda_4d * loss_4d
        return loss_total, {
            "loss_action": loss_action.detach(),
            "loss_4d": self.loss_lambda_4d * loss_4d.detach(),
        }

    @torch.no_grad()
    def infer_joint(
        self,
        *,
        action_horizon: int,
        vggt_images: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        vggt_teacher_g0: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        four_d_sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
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
        if vggt_teacher_g0 is None:
            g0, (num_views, tokens_per_view) = self._capture_teacher_g0(
                vggt_images, batch_size=batch_size
            )
        else:
            g0, (num_views, tokens_per_view) = self._validate_cached_teacher_feature(
                vggt_teacher_g0,
                name="vggt_teacher_g0",
                batch_size=batch_size,
                allow_unbatched=True,
            )

        action_generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        four_d_generator = (
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
        g32 = torch.randn(
            g0.shape,
            generator=four_d_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        action_ts, action_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        four_d_ts, four_d_deltas = self.infer_four_d_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=g32.dtype,
            shift_override=four_d_sigma_shift,
        )
        for action_t, action_delta, four_d_t, four_d_delta in zip(
            action_ts, action_deltas, four_d_ts, four_d_deltas
        ):
            pred_g32, pred_action = self._predict_joint_flow(
                noisy_action=action,
                action_timestep=action_t.expand(batch_size),
                noisy_g32=g32,
                four_d_timestep=four_d_t.expand(batch_size),
                g0=g0,
                context=context,
                context_mask=context_mask,
                vggt_kv_by_layer=memory,
            )
            action = self.infer_action_scheduler.step(pred_action, action_delta, action)
            g32 = self.infer_four_d_scheduler.step(pred_g32, four_d_delta, g32)

        return {
            "action": action.detach().to(device="cpu", dtype=torch.float32),
            "g32": g32.unflatten(1, (num_views, tokens_per_view))
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
        vggt_teacher_g0: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        four_d_sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> torch.Tensor:
        del vggt_teacher_g0, four_d_sigma_shift
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
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        for timestep, delta in zip(timesteps, deltas):
            pred_action = self._predict_action_flow(
                noisy_action=action,
                action_timestep=timestep.expand(batch_size),
                context=context,
                context_mask=context_mask,
                vggt_kv_by_layer=memory,
            )
            action = self.infer_action_scheduler.step(pred_action, delta, action)
        return action.detach().to(device="cpu", dtype=torch.float32)

    def save_checkpoint(self, path, optimizer=None, step=None):
        if optimizer is not None:
            raise ValueError(
                "Weight checkpoints do not contain optimizer state; use Accelerate state."
            )
        payload = {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            "feature_dim": self.feature_dim,
            "feature_layer": self.feature_layer,
            "expected_num_views": self.expected_num_views,
            "train_vggt": self.train_vggt,
            "action_expert": self.action_expert.state_dict(),
            "four_d_expert": self.four_d_expert.state_dict(),
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
            "feature_dim",
            "feature_layer",
            "expected_num_views",
            "train_vggt",
            "action_expert",
            "four_d_expert",
            "vggt_bridge",
            "vggt_aggregator",
            "proprio_encoder",
            "step",
            "torch_dtype",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            actual = set(payload) if isinstance(payload, dict) else set()
            raise ValueError(
                "ActionVGGTFastWAM4D checkpoint key mismatch: "
                f"missing={sorted(expected_keys - actual)}, "
                f"unexpected={sorted(actual - expected_keys)}."
            )
        if (
            payload["format"] != self.CHECKPOINT_FORMAT
            or payload["version"] != self.CHECKPOINT_VERSION
        ):
            raise ValueError(
                "Unsupported ActionVGGTFastWAM4D checkpoint format/version: "
                f"{payload['format']!r}/{payload['version']!r}."
            )
        for key, expected in (
            ("feature_dim", self.feature_dim),
            ("feature_layer", self.feature_layer),
            ("expected_num_views", self.expected_num_views),
        ):
            if int(payload[key]) != int(expected):
                raise ValueError(
                    f"Checkpoint `{key}` mismatch: expected {expected}, got {payload[key]}."
                )
        if bool(payload["train_vggt"]) != self.train_vggt:
            raise ValueError(
                "Checkpoint/model `train_vggt` mismatch: "
                f"{payload['train_vggt']} vs {self.train_vggt}."
            )

        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.four_d_expert.load_state_dict(payload["four_d_expert"], strict=True)
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


__all__ = ["ActionVGGTFastWAM4D"]
