from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_dit import ActionDiT
from .geoaware_projector import FrozenVGGTFeatureBackbone
from .layerwise_vggt_context import LayerwiseVGGTContextAdapter
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .vggt_mot import validate_vggt_images


class ActionCAVGGTFastWAM(nn.Module):
    """Action-only policy with layer-mapped VGGT cross-attention context."""

    CHECKPOINT_FORMAT = "action_ca_vggt_fastwam"
    CHECKPOINT_VERSION = 2
    LEGACY_CHECKPOINT_VERSION = 1
    evaluation_mode = "action_only"

    def __init__(
        self,
        *,
        action_expert: ActionDiT,
        mot: MoT,
        vggt_backbone: FrozenVGGTFeatureBackbone,
        visual_context_adapter: LayerwiseVGGTContextAdapter,
        text_dim: int,
        expected_num_views: int,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        train_vggt: bool = False,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        if tuple(mot.expert_order) != ("action",):
            raise ValueError(
                "ActionCAVGGTFastWAM requires expert_order=('action',)."
            )
        if mot.mixtures["action"] is not action_expert:
            raise ValueError("`action_expert` must be the action module owned by `mot`.")
        if mot.vggt_bridge is not None:
            raise ValueError(
                "ActionCAVGGTFastWAM uses cross-attention context and must not "
                "contain a VGGT K/V bridge."
            )
        if visual_context_adapter.output_dim != action_expert.hidden_dim:
            raise ValueError(
                "Visual context width must equal ActionDiT hidden_dim: "
                f"{visual_context_adapter.output_dim} vs {action_expert.hidden_dim}."
            )
        if expected_num_views <= 0:
            raise ValueError("`expected_num_views` must be positive.")

        # Wan22Trainer optimizes model.dit plus the optional proprio encoder.
        # Keeping MoT and all pair-specific adapters here produces one optimizer
        # group; configurable VGGT ownership is handled separately below.
        self.dit = nn.ModuleDict(
            {
                "mot": mot,
                "visual_context_adapter": visual_context_adapter,
            }
        )
        self.train_vggt = bool(train_vggt)
        self.text_dim = int(text_dim)
        self.expected_num_views = int(expected_num_views)
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

        # Preserve external ownership while frozen; register trainable VGGT so
        # Accelerate/DeepSpeed can synchronize and checkpoint it.
        if self.train_vggt:
            self.vggt_backbone = vggt_backbone
        else:
            object.__setattr__(self, "vggt_backbone", vggt_backbone)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.to(device=self.device, dtype=self.torch_dtype)
        self.vggt_backbone.train(self.training and self.train_vggt)

    @property
    def mot(self) -> MoT:
        return self.dit["mot"]

    @property
    def visual_context_adapter(self) -> LayerwiseVGGTContextAdapter:
        return self.dit["visual_context_adapter"]

    @property
    def action_expert(self) -> ActionDiT:
        return self.mot.mixtures["action"]

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        backbone = getattr(self, "vggt_backbone", None)
        if backbone is not None and not self.train_vggt:
            backbone._apply(fn, recurse=recurse)
            backbone.eval()
        parameter = next(self.dit.parameters(), None)
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

    def _capture_visual_contexts(
        self,
        vggt_images: torch.Tensor,
        *,
        batch_size: int,
    ) -> dict[int, torch.Tensor]:
        """Run frozen VGGT once and project every mapped layer's full tokens.

        Tensor flow for VGGT layer ``g`` mapped to ActionDiT layer ``a``:
        ``images [B,V,3,H,W] -> feature_g [B,V,S,2048] ->
        adapter_a [B,V,S,1024] -> visual_context_a [B,V*S,1024]``.
        """
        validate_vggt_images(vggt_images, batch_size=batch_size)
        if int(vggt_images.shape[1]) != self.expected_num_views:
            raise ValueError(
                f"Expected {self.expected_num_views} VGGT views, "
                f"got {vggt_images.shape[1]}."
            )
        images = vggt_images.to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        features_by_layer = self.vggt_backbone(images)
        return self.visual_context_adapter(features_by_layer)

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
                    "`proprio` was provided but proprio encoding is disabled."
                )
            return context, context_mask
        if proprio is None:
            raise ValueError("`proprio` is required when proprio encoding is enabled.")
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

    def _predict_action(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        visual_context_by_action_layer: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Predict action flow with layer-dependent cross-attention context.

        ``ActionDiT.pre_dit`` first maps T5/state tokens to the ActionDiT hidden
        width. For a mapped layer, its projected VGGT tokens are appended to
        that base context and its allow-mask. Unmapped layers receive the base
        language/state payload unchanged.
        """
        pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        batch_size, action_len = pre["tokens"].shape[:2]
        context_by_layer: dict[int, dict[str, torch.Tensor]] = {}
        for action_layer, visual_context in visual_context_by_action_layer.items():
            if (
                visual_context.ndim != 3
                or visual_context.shape[0] != batch_size
                or visual_context.shape[2] != self.action_expert.hidden_dim
            ):
                raise ValueError(
                    f"Visual context for ActionDiT layer {action_layer} must be "
                    f"[B,N,{self.action_expert.hidden_dim}], got "
                    f"{tuple(visual_context.shape)}."
                )
            visual_mask = torch.ones(
                (batch_size, action_len, visual_context.shape[1]),
                dtype=torch.bool,
                device=pre["context_mask"].device,
            )
            context_by_layer[action_layer] = {
                "context": torch.cat(
                    [pre["context"], visual_context],
                    dim=1,
                ),
                "mask": torch.cat(
                    [pre["context_mask"], visual_mask],
                    dim=2,
                ),
            }

        attention_mask = torch.ones(
            (action_len, action_len),
            dtype=torch.bool,
            device=pre["tokens"].device,
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
            context_by_layer_all={"action": context_by_layer},
            t_mod_all={"action": pre["t_mod"]},
        )
        return self.action_expert.post_dit(output["action"], pre)

    def training_loss(self, sample):
        required = {"action", "context", "context_mask", "vggt_images"}
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(
                f"ActionCAVGGTFastWAM training sample is missing {missing}."
            )
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

        visual_contexts = self._capture_visual_contexts(
            sample["vggt_images"],
            batch_size=batch_size,
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
            noisy_action,
            timestep,
            context,
            context_mask,
            visual_contexts,
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
        return loss_action, {"loss_action": loss_action.detach()}

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
    ) -> torch.Tensor:
        self.eval()
        if action_horizon <= 0:
            raise ValueError("`action_horizon` must be positive.")
        context_batch = 1 if context.ndim == 2 else int(context.shape[0])
        context, context_mask = self._prepare_context(
            context,
            context_mask,
            proprio,
            batch_size=context_batch,
            training=False,
        )
        # VGGT and all pair adapters run once; their outputs are reused by
        # every flow-matching denoising step below.
        visual_contexts = self._capture_visual_contexts(
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
            pred_action = self._predict_action(
                action,
                timestep.expand(context_batch),
                context,
                context_mask,
                visual_contexts,
            )
            action = self.infer_action_scheduler.step(pred_action, delta, action)
        return action.detach().to(device="cpu", dtype=torch.float32)

    def save_checkpoint(self, path, optimizer=None, step=None):
        if optimizer is not None:
            raise ValueError(
                "ActionCAVGGTFastWAM weight checkpoints do not contain optimizer "
                "state; use the Accelerate training-state checkpoint."
            )
        checkpoint_version = (
            self.CHECKPOINT_VERSION
            if self.train_vggt
            else self.LEGACY_CHECKPOINT_VERSION
        )
        payload = {
            "format": self.CHECKPOINT_FORMAT,
            "version": checkpoint_version,
            "layer_pairs": [list(pair) for pair in self.visual_context_adapter.layer_pairs],
            "vggt_feature_dim": self.visual_context_adapter.input_dim,
            "visual_context_dim": self.visual_context_adapter.output_dim,
            "expected_num_views": self.expected_num_views,
            "action_expert": self.action_expert.state_dict(),
            "visual_context_adapter": self.visual_context_adapter.state_dict(),
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
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        if optimizer is not None:
            raise ValueError(
                "ActionCAVGGTFastWAM weight checkpoints do not contain optimizer "
                "state; use the Accelerate training-state checkpoint."
            )
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid ActionCAVGGTFastWAM checkpoint: {path}.")
        base_expected_keys = {
            "format",
            "version",
            "layer_pairs",
            "vggt_feature_dim",
            "visual_context_dim",
            "expected_num_views",
            "action_expert",
            "visual_context_adapter",
            "proprio_encoder",
            "step",
            "torch_dtype",
        }
        version = payload.get("version")
        expected_keys = set(base_expected_keys)
        if version == self.CHECKPOINT_VERSION:
            expected_keys.add("vggt_aggregator")
        missing = sorted(expected_keys - set(payload))
        unexpected = sorted(set(payload) - expected_keys)
        if missing or unexpected:
            raise ValueError(
                "ActionCAVGGTFastWAM checkpoint key mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if payload.get("format") != self.CHECKPOINT_FORMAT or version not in {
            self.LEGACY_CHECKPOINT_VERSION,
            self.CHECKPOINT_VERSION,
        }:
            raise ValueError(
                "Unsupported ActionCAVGGTFastWAM checkpoint format/version."
            )
        metadata = {
            "layer_pairs": [
                list(pair) for pair in self.visual_context_adapter.layer_pairs
            ],
            "vggt_feature_dim": self.visual_context_adapter.input_dim,
            "visual_context_dim": self.visual_context_adapter.output_dim,
            "expected_num_views": self.expected_num_views,
        }
        mismatches = {
            key: (payload[key], expected)
            for key, expected in metadata.items()
            if payload[key] != expected
        }
        if mismatches:
            raise ValueError(f"Checkpoint architecture mismatch: {mismatches}.")

        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.visual_context_adapter.load_state_dict(
            payload["visual_context_adapter"], strict=True
        )
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
                "Version-1 ActionCAVGGTFastWAM checkpoint has no VGGT weights; "
                "keeping the Aggregator weights loaded during model construction.",
                UserWarning,
                stacklevel=2,
            )
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


__all__ = ["ActionCAVGGTFastWAM"]
