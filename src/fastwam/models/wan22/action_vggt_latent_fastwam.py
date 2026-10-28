from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_dit import ActionDiT
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .vggt_mot import FrozenVGGTBackbone, validate_vggt_images
from .vggt_latent_dit import VGGTLatentDiT


class ActionVGGTLatentFastWAM(nn.Module):
    """Joint action/future-adapter-latent flow model.

    The latent expert receives only the current-frame VGGT K/V memory through
    the bridge.  It never receives an explicit current latent (``z0``).
    """

    CHECKPOINT_FORMAT = "action_vggt_latent_fastwam"
    CHECKPOINT_VERSION = 1
    evaluation_mode = "action_only"
    requires_vggt_future_images = True
    requires_vggt_geometry_latent_targets = True
    requires_vggt_teacher_features = False
    requires_depth_targets = False

    def __init__(self, *, action_expert: ActionDiT, vggt_expert: VGGTLatentDiT,
                 mot: MoT, vggt_backbone: FrozenVGGTBackbone, text_dim: int,
                 expected_num_views: int, proprio_dim: Optional[int] = None,
                 device: str = "cpu", torch_dtype: torch.dtype = torch.float32,
                 train_vggt: bool = True, action_only_inference: bool = False,
                 vggt_geometry_teacher=None, geometry_adapter=None,
                 loss_lambda_vggt: float = 1.0, train_shift: float = 5.0,
                 infer_shift: float = 5.0, num_train_timesteps: int = 1000) -> None:
        super().__init__()
        if tuple(mot.expert_order) != ("vggt", "action"):
            raise ValueError("ActionVGGTLatentFastWAM requires expert_order=('vggt','action').")
        if mot.mixtures["action"] is not action_expert or mot.mixtures["vggt"] is not vggt_expert:
            raise ValueError("MoT must own both supplied experts.")
        if mot.vggt_bridge is None or mot.vggt_bridge.target_experts != frozenset({"vggt", "action"}):
            raise ValueError("VGGT bridge must target both vggt and action experts.")
        if expected_num_views <= 0 or loss_lambda_vggt < 0:
            raise ValueError("Invalid expected_num_views/loss_lambda_vggt.")
        if action_expert.text_dim != vggt_expert.text_dim or action_expert.text_dim != int(text_dim):
            raise ValueError("Action/VGGT experts and model must use the same text_dim.")
        self.mot = mot
        self.dit = mot
        self.train_vggt = bool(train_vggt)
        self.requires_vggt_future_images = vggt_geometry_teacher is not None and geometry_adapter is not None and not bool(action_only_inference)
        self.requires_vggt_geometry_latent_targets = not self.requires_vggt_future_images and not bool(action_only_inference)
        self.action_only_inference = bool(action_only_inference)
        self.text_dim = int(text_dim)
        self.expected_num_views = int(expected_num_views)
        self.loss_lambda_vggt = float(loss_lambda_vggt)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.proprio_encoder = None if self.proprio_dim is None else nn.Linear(self.proprio_dim, self.text_dim, dtype=torch_dtype)
        self.train_flow_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=int(num_train_timesteps), shift=float(train_shift))
        self.infer_flow_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=int(num_train_timesteps), shift=float(infer_shift))
        self.vggt_expert = vggt_expert
        object.__setattr__(self, "vggt_geometry_teacher", vggt_geometry_teacher)
        object.__setattr__(self, "geometry_adapter", geometry_adapter)
        if geometry_adapter is not None:
            geometry_adapter.eval().requires_grad_(False)
        self.mot.vggt_bridge.detach_vggt_memory = not self.train_vggt
        if self.train_vggt:
            self.vggt_backbone = vggt_backbone
        else:
            object.__setattr__(self, "vggt_backbone", vggt_backbone)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.to(device=self.device, dtype=self.torch_dtype)
        self.vggt_backbone.train(self.training and self.train_vggt)

    @property
    def action_expert(self):
        return self.mot.mixtures["action"]

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if not self.train_vggt:
            self.vggt_backbone._apply(fn, recurse=recurse)
            self.vggt_backbone.eval()
        parameter = next(self.mot.parameters(), None)
        if parameter is not None:
            self.device = parameter.device
            if parameter.is_floating_point():
                self.torch_dtype = parameter.dtype
        return self

    def train(self, mode=True):
        super().train(mode)
        if not self.train_vggt:
            self.vggt_backbone.eval()
        return self

    def _validate_views(self, images, *, batch_size):
        validate_vggt_images(images, batch_size=batch_size)
        if int(images.shape[1]) != self.expected_num_views:
            raise ValueError(f"Expected {self.expected_num_views} VGGT views, got {images.shape[1]}.")

    def _capture_vggt_kv(self, images, *, batch_size):
        self._validate_views(images, batch_size=batch_size)
        return self.vggt_backbone(images.to(device=self.device, dtype=self.torch_dtype, non_blocking=True))

    def _capture_future_latent(self, images, *, batch_size):
        """Online target: [B,9,V,3,H,W] -> Omega geometry -> adapter -> [B,48,3,24,20]."""
        if self.vggt_geometry_teacher is None or self.geometry_adapter is None:
            raise RuntimeError("Online latent targets require frozen VGGT teacher and geometry adapter.")
        if images.ndim != 6 or int(images.shape[1]) != 9:
            raise ValueError(f"vggt_future_images must be [B,9,V,3,H,W], got {tuple(images.shape)}")
        self._validate_views(images[:, 0], batch_size=batch_size)
        with torch.no_grad():
            geometry = self.vggt_geometry_teacher.encode_clip_geometry_only(
                images.to(device=self.device, dtype=self.torch_dtype), output_dtype=torch.float32
            )
            latent = self.geometry_adapter.encode(geometry).mode()
        expected = (batch_size, 48, 3, 24, 20)
        if tuple(latent.shape) != expected:
            raise RuntimeError(f"Adapter latent must be {expected}, got {tuple(latent.shape)}")
        return latent.to(device=self.device, dtype=self.torch_dtype)

    def _prepare_context(self, context, context_mask, proprio, *, batch_size, training):
        if not training:
            if context.ndim == 2: context = context.unsqueeze(0)
            if context_mask.ndim == 1: context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2 or context.shape[0] != batch_size or context_mask.shape != context.shape[:2] or context.shape[2] != self.text_dim:
            raise ValueError("Invalid context/context_mask shape.")
        context = context.to(device=self.device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if self.proprio_encoder is None:
            if proprio is not None: raise ValueError("proprio_dim=None but proprio was provided.")
            return context, context_mask
        if proprio is None: raise ValueError("proprio is required.")
        if training and proprio.ndim == 3: proprio = proprio[:, 0]
        elif not training and proprio.ndim == 1: proprio = proprio.unsqueeze(0)
        if proprio.shape != (batch_size, self.proprio_dim): raise ValueError("Invalid proprio shape.")
        token = self.proprio_encoder(proprio.to(device=self.device, dtype=self.torch_dtype)).unsqueeze(1)
        return torch.cat([context, token], 1), torch.cat([context_mask, torch.ones((batch_size, 1), dtype=torch.bool, device=self.device)], 1)

    @staticmethod
    def _isolated_mask(vggt_len, action_len, device):
        mask = torch.zeros((vggt_len + action_len, vggt_len + action_len), dtype=torch.bool, device=device)
        mask[:vggt_len, :vggt_len] = True
        mask[vggt_len:, vggt_len:] = True
        return mask

    def _predict_joint_flow(self, *, noisy_latent, noisy_action, timestep, context, context_mask, vggt_kv_by_layer):
        # [B,48,3,24,20] -> [B,1440,48]; no z0 concatenation occurs here.
        latent_tokens = self.vggt_expert.flatten_latent(noisy_latent)
        vggt_pre = self.vggt_expert.pre_dit(latent_tokens, timestep, context, context_mask)
        action_pre = self.action_expert.pre_dit(noisy_action, timestep, context, context_mask)
        mask = self._isolated_mask(int(vggt_pre["tokens"].shape[1]), int(action_pre["tokens"].shape[1]), vggt_pre["tokens"].device)
        output = self.mot(
            embeds_all={"vggt": vggt_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=mask,
            freqs_all={"vggt": vggt_pre["freqs"], "action": action_pre["freqs"]},
            context_all={"vggt": {"context": vggt_pre["context"], "mask": vggt_pre["context_mask"]}, "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]}},
            t_mod_all={"vggt": vggt_pre["t_mod"], "action": action_pre["t_mod"]},
            vggt_kv_by_layer=vggt_kv_by_layer, isolated_expert_attention=True,
        )
        pred_latent = self.vggt_expert.unflatten_latent(self.vggt_expert.post_dit(output["vggt"], vggt_pre))
        pred_action = self.action_expert.post_dit(output["action"], action_pre)
        return pred_latent, pred_action

    def _predict_action_flow(self, *, noisy_action, timestep, context, context_mask, vggt_kv_by_layer):
        pre = self.action_expert.pre_dit(noisy_action, timestep, context, context_mask)
        n = int(pre["tokens"].shape[1])
        output = self.mot(embeds_all={"action": pre["tokens"]}, attention_mask=torch.ones((n, n), dtype=torch.bool, device=pre["tokens"].device), freqs_all={"action": pre["freqs"]}, context_all={"action": {"context": pre["context"], "mask": pre["context_mask"]}}, t_mod_all={"action": pre["t_mod"]}, vggt_kv_by_layer=vggt_kv_by_layer, isolated_expert_attention=True, active_experts=("action",))
        return self.action_expert.post_dit(output["action"], pre)

    def _validate_latent_batch(self, latent, *, batch_size):
        expected = (batch_size, self.vggt_expert.action_dim, self.vggt_expert.latent_frames, self.vggt_expert.latent_height, self.vggt_expert.latent_width)
        if tuple(latent.shape) != expected: raise ValueError(f"vggt_latent_future must be {expected}, got {tuple(latent.shape)}")
        latent = latent.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if not torch.isfinite(latent).all(): raise ValueError("vggt_latent_future contains non-finite values.")
        return latent

    def training_loss(self, sample, vggt_kv_by_layer=None):
        required = {"action", "context", "context_mask", "vggt_images"}
        if "vggt_latent_future" not in sample and "vggt_future_images" not in sample:
            required.add("vggt_latent_future")
        missing = sorted(required - set(sample))
        if missing: raise ValueError(f"ActionVGGTLatentFastWAM training sample is missing {missing}.")
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype)
        batch_size = int(action.shape[0])
        context, context_mask = self._prepare_context(sample["context"], sample["context_mask"], sample.get("proprio"), batch_size=batch_size, training=True)
        latent = self._validate_latent_batch(sample["vggt_latent_future"], batch_size=batch_size) if "vggt_latent_future" in sample else self._capture_future_latent(sample["vggt_future_images"], batch_size=batch_size)
        action_noise, latent_noise = torch.randn_like(action), torch.randn_like(latent)
        timestep = self.train_flow_scheduler.sample_training_t(batch_size=batch_size, device=self.device, dtype=action.dtype)
        noisy_action = self.train_flow_scheduler.add_noise(action, action_noise, timestep)
        noisy_latent = self.train_flow_scheduler.add_noise(latent, latent_noise, timestep)
        target_action = self.train_flow_scheduler.training_target(action, action_noise, timestep)
        target_latent = self.train_flow_scheduler.training_target(latent, latent_noise, timestep)
        memory = vggt_kv_by_layer if vggt_kv_by_layer is not None else self._capture_vggt_kv(sample["vggt_images"], batch_size=batch_size)
        pred_latent, pred_action = self._predict_joint_flow(noisy_latent=noisy_latent, noisy_action=noisy_action, timestep=timestep, context=context, context_mask=context_mask, vggt_kv_by_layer=memory)
        weight = self.train_flow_scheduler.training_weight(timestep).to(device=self.device, dtype=torch.float32)
        action_loss = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=(1, 2))
        latent_loss = F.mse_loss(pred_latent.float(), target_latent.float(), reduction="none").mean(dim=(1, 2, 3, 4))
        valid = sample.get("vggt_future_is_pad")
        valid_factor = torch.ones_like(latent_loss) if valid is None else (~valid.to(device=self.device, dtype=torch.bool)).reshape(batch_size, -1).all(dim=1).to(latent_loss.dtype)
        loss_action = (action_loss * weight).mean()
        loss_vggt = (latent_loss * weight * valid_factor).sum() / valid_factor.sum().clamp(min=1)
        return loss_action + self.loss_lambda_vggt * loss_vggt, {"loss_action": loss_action.detach(), "loss_vggt": (self.loss_lambda_vggt * loss_vggt).detach()}

    @torch.no_grad()
    def infer_action(self, *, action_horizon, vggt_images, context, context_mask, proprio=None, num_inference_steps=20, sigma_shift=None, seed=None, rand_device="cpu", **_ignored):
        self.eval(); batch_size = 1 if context.ndim == 2 else int(context.shape[0])
        context, context_mask = self._prepare_context(context, context_mask, proprio, batch_size=batch_size, training=False)
        memory = self._capture_vggt_kv(vggt_images, batch_size=batch_size)
        gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn((batch_size, action_horizon, self.action_expert.action_dim), generator=gen, device=rand_device).to(self.device, self.torch_dtype)
        ts, ds = self.infer_flow_scheduler.build_inference_schedule(num_inference_steps=num_inference_steps, device=self.device, dtype=action.dtype, shift_override=sigma_shift)
        for t, d in zip(ts, ds): action = self.infer_flow_scheduler.step(self._predict_action_flow(noisy_action=action, timestep=t.expand(batch_size), context=context, context_mask=context_mask, vggt_kv_by_layer=memory), d, action)
        return action.detach().cpu().float()

    @torch.no_grad()
    def infer_joint(self, *, action_horizon, vggt_images, context, context_mask, proprio=None, num_inference_steps=20, sigma_shift=None, seed=None, rand_device="cpu"):
        self.eval(); batch_size = 1 if context.ndim == 2 else int(context.shape[0])
        context, context_mask = self._prepare_context(context, context_mask, proprio, batch_size=batch_size, training=False)
        memory = self._capture_vggt_kv(vggt_images, batch_size=batch_size)
        gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn((batch_size, action_horizon, self.action_expert.action_dim), generator=gen, device=rand_device).to(self.device, self.torch_dtype)
        latent = torch.randn((batch_size, self.vggt_expert.action_dim, self.vggt_expert.latent_frames, self.vggt_expert.latent_height, self.vggt_expert.latent_width), generator=gen, device=rand_device).to(self.device, self.torch_dtype)
        ts, ds = self.infer_flow_scheduler.build_inference_schedule(num_inference_steps=num_inference_steps, device=self.device, dtype=action.dtype, shift_override=sigma_shift)
        for t, d in zip(ts, ds):
            pz, pa = self._predict_joint_flow(noisy_latent=latent, noisy_action=action, timestep=t.expand(batch_size), context=context, context_mask=context_mask, vggt_kv_by_layer=memory)
            latent = self.infer_flow_scheduler.step(pz, d, latent); action = self.infer_flow_scheduler.step(pa, d, action)
        return {"action": action.detach().cpu().float(), "vggt_latent": latent.detach().cpu().float()}

    def save_checkpoint(self, path, optimizer=None, step=None):
        if optimizer is not None: raise ValueError("Use Accelerate state for optimizer checkpoints.")
        torch.save({"format": self.CHECKPOINT_FORMAT, "version": self.CHECKPOINT_VERSION, "latent_shape": [self.vggt_expert.action_dim, self.vggt_expert.latent_frames, self.vggt_expert.latent_height, self.vggt_expert.latent_width], "expected_num_views": self.expected_num_views, "train_vggt": self.train_vggt, "action_expert": self.action_expert.state_dict(), "vggt_expert": self.vggt_expert.state_dict(), "vggt_bridge": self.mot.vggt_bridge.state_dict(), "vggt_aggregator": self.vggt_backbone.aggregator.state_dict() if self.train_vggt else None, "proprio_encoder": None if self.proprio_encoder is None else self.proprio_encoder.state_dict(), "step": step}, path)

    def load_checkpoint(self, path, optimizer=None):
        """Load the compact weight checkpoint used by RobotWin action-only eval."""
        if optimizer is not None:
            raise ValueError("Weight checkpoints do not contain optimizer state; use Accelerate state.")
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != self.CHECKPOINT_FORMAT or int(payload.get("version", -1)) != self.CHECKPOINT_VERSION:
            raise ValueError("Unsupported ActionVGGTLatentFastWAM checkpoint format/version.")
        shape = tuple(payload.get("latent_shape", ()))
        expected = (self.vggt_expert.action_dim, self.vggt_expert.latent_frames, self.vggt_expert.latent_height, self.vggt_expert.latent_width)
        if shape != expected:
            raise ValueError(f"Checkpoint latent shape {shape} does not match model {expected}.")
        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.vggt_expert.load_state_dict(payload["vggt_expert"], strict=True)
        self.mot.vggt_bridge.load_state_dict(payload["vggt_bridge"], strict=True)
        if self.train_vggt and payload.get("vggt_aggregator") is not None:
            self.vggt_backbone.aggregator.load_state_dict(payload["vggt_aggregator"], strict=True)
        if self.proprio_encoder is not None and payload.get("proprio_encoder") is not None:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        return int(payload.get("step") or 0)

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)


__all__ = ["ActionVGGTLatentFastWAM"]
