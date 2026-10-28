from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_dit import ActionDiT
from .action_vggt_fastwam import ActionVGGTFastWAM
from .mot import MoT
from .vggt_mot import FrozenVGGTBackbone, validate_vggt_images


VGGTMemory = dict[int, dict[str, torch.Tensor | int]]


class TimeRoutedVGGTBackbone(nn.Module):
    """Route image batches to independent high/low-noise VGGT backbones.

    The route is driven by action diffusion time, not by video-frame time.  A
    mixed batch is split before the expensive VGGT forwards and its per-layer
    K/V memories are restored to the original batch order afterwards.
    """

    VALID_ROUTES = frozenset({"high", "low"})

    def __init__(
        self,
        high_noise_vggt: FrozenVGGTBackbone,
        low_noise_vggt: FrozenVGGTBackbone,
    ) -> None:
        super().__init__()
        high_layers = tuple(int(v) for v in high_noise_vggt.callback_layer_indices)
        low_layers = tuple(int(v) for v in low_noise_vggt.callback_layer_indices)
        if high_layers != low_layers:
            raise ValueError(
                "Time-routed VGGT backbones must capture the same layers, got "
                f"high={high_layers} and low={low_layers}."
            )
        if bool(high_noise_vggt.trainable) != bool(low_noise_vggt.trainable):
            raise ValueError("High/low VGGT backbones must use the same trainable mode.")
        if high_noise_vggt is low_noise_vggt:
            raise ValueError("High/low VGGT backbones must be distinct modules.")

        self.high_noise_vggt = high_noise_vggt
        self.low_noise_vggt = low_noise_vggt
        self.callback_layer_indices = high_layers
        self.trainable = bool(high_noise_vggt.trainable)

    def train(self, mode: bool = True):
        effective_mode = bool(mode) if self.trainable else False
        super().train(effective_mode)
        return self

    def backbone_for(self, route: str) -> FrozenVGGTBackbone:
        """Return the VGGT assigned to ``route`` without running it."""
        if route == "high":
            return self.high_noise_vggt
        if route == "low":
            return self.low_noise_vggt
        raise ValueError(f"Unknown VGGT route {route!r}; expected 'high' or 'low'.")

    @staticmethod
    def _validate_memory(
        memory: Mapping[int, Mapping[str, torch.Tensor | int]],
        *,
        expected_layers: Sequence[int],
        expected_batch: int,
        route: str,
    ) -> None:
        expected = set(int(layer) for layer in expected_layers)
        actual = set(int(layer) for layer in memory)
        if actual != expected:
            raise RuntimeError(
                f"{route} VGGT memory layer mismatch: expected={sorted(expected)}, "
                f"actual={sorted(actual)}."
            )
        for layer in expected_layers:
            entry = memory[int(layer)]
            if set(entry) != {"layer_idx", "k", "v"}:
                raise RuntimeError(
                    f"{route} VGGT layer {layer} must contain layer_idx/k/v."
                )
            if int(entry["layer_idx"]) != int(layer):
                raise RuntimeError(
                    f"{route} VGGT memory key {layer} contains layer_idx="
                    f"{entry['layer_idx']}."
                )
            k, v = entry["k"], entry["v"]
            if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                raise TypeError(f"{route} VGGT layer {layer} K/V must be tensors.")
            if k.ndim != 4 or k.shape != v.shape:
                raise RuntimeError(
                    f"{route} VGGT layer {layer} K/V must have matching "
                    f"[B,H,S,Dh] shapes, got {tuple(k.shape)}/{tuple(v.shape)}."
                )
            if int(k.shape[0]) != int(expected_batch):
                raise RuntimeError(
                    f"{route} VGGT layer {layer} batch mismatch: "
                    f"expected={expected_batch}, actual={k.shape[0]}."
                )

    def forward_route(self, images: torch.Tensor, *, route: str) -> VGGTMemory:
        """Capture one route's memory.

        Tensor flow: ``images [Br,V,3,H,W]`` -> selected VGGT -> per-layer
        ``K/V [Br,Hv,Sv,Dh]``.  Empty route tensors are intentionally rejected;
        :meth:`forward_mixed` skips those forwards entirely.
        """
        validate_vggt_images(images)
        if images.shape[0] == 0:
            raise ValueError("Do not execute a VGGT backbone with an empty route batch.")
        memory = self.backbone_for(route)(images)
        self._validate_memory(
            memory,
            expected_layers=self.callback_layer_indices,
            expected_batch=int(images.shape[0]),
            route=route,
        )
        return memory

    @staticmethod
    def _memory_zero_anchor(
        memory: Mapping[int, Mapping[str, torch.Tensor | int]],
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Connect a routed VGGT forward to the loss with exactly zero weight.

        DeepSpeed ranks must execute matching expert graphs even when one rank's
        local batch has no samples for a route.  Anchoring only the inactive
        parameters is insufficient because its autograd hooks can be scheduled
        differently from ranks that actually execute that expert.  A one-sample
        dummy forward preserves the expert graph and collective ordering while
        contributing no gradient value.
        """
        anchor = reference.sum() * 0.0
        for entry in memory.values():
            k, v = entry["k"], entry["v"]
            if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                raise TypeError("VGGT memory K/V must be tensors.")
            anchor = anchor + k.sum() * 0.0 + v.sum() * 0.0
        return anchor

    @staticmethod
    def _merge_memories(
        high_memory: VGGTMemory,
        low_memory: VGGTMemory,
        high_indices: torch.Tensor,
        low_indices: torch.Tensor,
    ) -> VGGTMemory:
        """Concatenate routed memories and invert the route permutation."""
        if set(high_memory) != set(low_memory):
            raise RuntimeError("High/low VGGT memories contain different layers.")
        routed_order = torch.cat((high_indices, low_indices), dim=0)
        inverse_order = torch.argsort(routed_order)
        merged: VGGTMemory = {}
        for layer in sorted(high_memory):
            high_entry, low_entry = high_memory[layer], low_memory[layer]
            if int(high_entry["layer_idx"]) != int(low_entry["layer_idx"]):
                raise RuntimeError(f"High/low layer metadata differs at layer {layer}.")
            high_k, high_v = high_entry["k"], high_entry["v"]
            low_k, low_v = low_entry["k"], low_entry["v"]
            if high_k.shape[1:] != low_k.shape[1:] or high_v.shape[1:] != low_v.shape[1:]:
                raise RuntimeError(
                    f"High/low VGGT memory shape differs at layer {layer}: "
                    f"K={tuple(high_k.shape)}/{tuple(low_k.shape)}, "
                    f"V={tuple(high_v.shape)}/{tuple(low_v.shape)}."
                )
            if high_k.device != low_k.device or high_k.dtype != low_k.dtype:
                raise RuntimeError(f"High/low VGGT K dtype/device differs at layer {layer}.")
            if high_v.device != low_v.device or high_v.dtype != low_v.dtype:
                raise RuntimeError(f"High/low VGGT V dtype/device differs at layer {layer}.")
            # [Bh/Bl,Hv,Sv,Dh] -> [B,Hv,Sv,Dh], restored to caller order.
            merged[layer] = {
                "layer_idx": int(layer),
                "k": torch.cat((high_k, low_k), dim=0).index_select(0, inverse_order),
                "v": torch.cat((high_v, low_v), dim=0).index_select(0, inverse_order),
            }
        return merged

    def forward_mixed(
        self,
        images: torch.Tensor,
        high_noise_mask: torch.Tensor,
    ) -> tuple[VGGTMemory, torch.Tensor]:
        """Run only the selected VGGT for every sample and restore batch order.

        Tensor flow::

            images [B,V,3,H,W] + high_noise_mask [B]
              -> images_high [Bh,V,3,H,W] / images_low [Bl,V,3,H,W]
              -> routed K/V [Br,Hv,Sv,Dh]
              -> restored K/V [B,Hv,Sv,Dh]

        The second return value is the inactive-parameter zero anchor.
        """
        validate_vggt_images(images)
        if high_noise_mask.dtype != torch.bool or high_noise_mask.ndim != 1:
            raise ValueError("`high_noise_mask` must be a rank-1 boolean tensor [B].")
        if high_noise_mask.device != images.device:
            raise ValueError("`high_noise_mask` and images must be on the same device.")
        if high_noise_mask.shape[0] != images.shape[0]:
            raise ValueError(
                "Route mask batch mismatch: "
                f"mask={high_noise_mask.shape[0]}, images={images.shape[0]}."
            )

        high_indices = torch.nonzero(high_noise_mask, as_tuple=False).flatten()
        low_indices = torch.nonzero(~high_noise_mask, as_tuple=False).flatten()
        high_memory = low_memory = None
        anchor = images.sum() * 0.0

        if high_indices.numel() > 0:
            high_memory = self.forward_route(
                images.index_select(0, high_indices), route="high"
            )
        elif self.trainable:
            dummy_memory = self.forward_route(images[:1], route="high")
            anchor = anchor + self._memory_zero_anchor(
                dummy_memory, reference=images
            )
        if low_indices.numel() > 0:
            low_memory = self.forward_route(
                images.index_select(0, low_indices), route="low"
            )
        elif self.trainable:
            dummy_memory = self.forward_route(images[:1], route="low")
            anchor = anchor + self._memory_zero_anchor(
                dummy_memory, reference=images
            )

        if high_memory is None:
            assert low_memory is not None
            return low_memory, anchor
        if low_memory is None:
            return high_memory, anchor
        return (
            self._merge_memories(
                high_memory, low_memory, high_indices, low_indices
            ),
            anchor,
        )


class ActionVGGTMoEFastWAM(ActionVGGTFastWAM):
    """Action-only FastWAM with hard diffusion-time routing over two VGGTs."""

    CHECKPOINT_FORMAT = "action_vggt_moe_fastwam"
    CHECKPOINT_VERSION = 1
    ROUTE_RULE = "high_if_sigma_greater_than_or_equal_to_boundary"
    BACKBONE_TYPE = TimeRoutedVGGTBackbone

    def __init__(
        self,
        *,
        action_expert: ActionDiT,
        mot: MoT,
        vggt_backbone: TimeRoutedVGGTBackbone,
        text_dim: int,
        layer_pairs: Sequence[Sequence[int]],
        time_route_boundary: float = 0.8333333333333334,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        train_vggt: bool = True,
        vggt_fp32_storage_bf16_compute: bool = False,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
    ) -> None:
        boundary = float(time_route_boundary)
        if not 0.0 < boundary < 1.0:
            raise ValueError(
                f"`time_route_boundary` must be in (0, 1), got {boundary}."
            )
        if not isinstance(vggt_backbone, self.BACKBONE_TYPE):
            raise TypeError(
                "ActionVGGTMoEFastWAM requires a time-routed VGGT backbone."
            )
        raw_pairs = tuple(layer_pairs)
        if not raw_pairs or any(len(pair) != 2 for pair in raw_pairs):
            raise ValueError("`layer_pairs` must contain (vggt_layer, action_layer) pairs.")
        pairs = tuple((int(pair[0]), int(pair[1])) for pair in raw_pairs)
        if tuple(pair[0] for pair in pairs) != vggt_backbone.callback_layer_indices:
            raise ValueError(
                "VGGT callback layers must exactly match `layer_pairs` order."
            )

        self.time_route_boundary = boundary
        self.route_layer_pairs = pairs
        super().__init__(
            action_expert=action_expert,
            mot=mot,
            vggt_backbone=vggt_backbone,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            train_vggt=train_vggt,
            vggt_fp32_storage_bf16_compute=vggt_fp32_storage_bf16_compute,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            reconstruction_loss_config=None,
        )

    @property
    def routed_vggt_backbone(self) -> TimeRoutedVGGTBackbone:
        return self.vggt_backbone

    def high_noise_mask(self, timestep: torch.Tensor) -> torch.Tensor:
        """Return the hard route mask for action diffusion timesteps ``[B]``."""
        if not isinstance(timestep, torch.Tensor) or timestep.ndim != 1:
            raise ValueError("`timestep` must be a rank-1 tensor [B].")
        sigma = timestep.float() / float(
            self.train_action_scheduler.num_train_timesteps
        )
        return sigma >= self.time_route_boundary

    def _route_for_scalar_timestep(self, timestep: torch.Tensor) -> str:
        if not isinstance(timestep, torch.Tensor) or timestep.numel() != 1:
            raise ValueError("Inference route requires one scalar diffusion timestep.")
        sigma = float(timestep.float().item()) / float(
            self.infer_action_scheduler.num_train_timesteps
        )
        return "high" if sigma >= self.time_route_boundary else "low"

    def _capture_route(
        self,
        vggt_images: torch.Tensor,
        *,
        route: str,
        batch_size: int,
    ) -> VGGTMemory:
        validate_vggt_images(vggt_images, batch_size=batch_size)
        images = vggt_images.to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        return self.routed_vggt_backbone.forward_route(images, route=route)

    def training_loss(self, sample, vggt_kv_by_layer=None):
        """Compute action flow loss with per-sample hard VGGT routing.

        ``timestep [B]`` is sampled before visual capture.  It denotes action
        diffusion time (not a video-frame index), splits ``images
        [B,V,3,H,W]`` into route batches, and produces restored VGGT memory
        ``[B,Hv,Sv,Dh]``.  The shared bridge and sole ActionDiT then process
        ``noisy_action [B,Ta,A]`` exactly once.
        """
        if vggt_kv_by_layer is not None:
            raise ValueError(
                "ActionVGGTMoEFastWAM does not support precomputed VGGT K/V; "
                "the route-specific backbone must run live."
            )
        required = {"action", "context", "context_mask", "vggt_images"}
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(
                f"ActionVGGTMoEFastWAM training sample is missing {missing}."
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

        noise = torch.randn_like(action)
        timestep = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise, timestep)
        target_action = self.train_action_scheduler.training_target(
            action, noise, timestep
        )
        high_mask = self.high_noise_mask(timestep)

        images = sample["vggt_images"]
        validate_vggt_images(images, batch_size=batch_size)
        images = images.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        memory, inactive_anchor = self.routed_vggt_backbone.forward_mixed(
            images, high_mask
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
        weighted_loss = per_sample_loss * weight
        loss_action = weighted_loss.mean()
        loss = loss_action + inactive_anchor.to(dtype=loss_action.dtype)

        zero = weighted_loss.new_zeros(())
        loss_high = weighted_loss[high_mask].mean() if high_mask.any() else zero
        low_mask = ~high_mask
        loss_low = weighted_loss[low_mask].mean() if low_mask.any() else zero
        loss_dict = {
            "loss_action": loss_action.detach(),
            "loss_action_high": loss_high.detach(),
            "loss_action_low": loss_low.detach(),
            "route_high_fraction": high_mask.float().mean().detach(),
            "route_high_count": high_mask.sum().detach(),
            "route_low_count": low_mask.sum().detach(),
        }
        return loss, loss_dict

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
    ) -> torch.Tensor:
        """Denoise actions while retaining only the active route's VGGT K/V.

        Inference timestep is shared by the batch.  VGGT does not consume that
        timestep, so K/V ``[B,Hv,Sv,Dh]`` is captured only when the hard route
        changes; ActionDiT still runs once per denoising step.
        """
        self.eval()
        if return_reconstruction:
            raise ValueError("ActionVGGTMoEFastWAM v1 has no reconstruction heads.")
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
        validate_vggt_images(vggt_images, batch_size=context_batch)

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

        active_route: str | None = None
        memory: VGGTMemory | None = None
        for timestep, delta in zip(timesteps, deltas):
            route = self._route_for_scalar_timestep(timestep)
            if route != active_route:
                # Drop the old stage's potentially large K/V before capturing
                # the new stage, so only one route cache remains live.
                memory = None
                memory = self._capture_route(
                    vggt_images,
                    route=route,
                    batch_size=context_batch,
                )
                active_route = route
            assert memory is not None
            batch_timestep = timestep.expand(context_batch)
            pred_action = self._predict_action(
                action, batch_timestep, context, context_mask, memory
            )
            action = self.infer_action_scheduler.step(pred_action, delta, action)
        return action.detach().to(device="cpu", dtype=torch.float32)

    def _checkpoint_metadata(self) -> dict[str, object]:
        bridge = self.mot.vggt_bridge
        return {
            "time_route_boundary": self.time_route_boundary,
            "route_rule": self.ROUTE_RULE,
            "num_train_timesteps": self.train_action_scheduler.num_train_timesteps,
            "layer_pairs": self.route_layer_pairs,
            "train_vggt": self.train_vggt,
            "bridge_config": {
                "projection_mode": bridge.projection_mode,
                "projection_sharing": bridge.projection_sharing,
                "fastwam_width": bridge.fastwam_width,
                "vggt_width": bridge.vggt_width,
                "fastwam_num_heads": bridge.fastwam_num_heads,
                "vggt_num_heads": bridge.vggt_num_heads,
            },
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        """Save both route experts and all shared action modules."""
        if optimizer is not None:
            raise ValueError(
                "ActionVGGTMoEFastWAM weight checkpoints do not contain optimizer "
                "state; use the Accelerate training-state checkpoint."
            )
        payload = {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            "action_expert": self.action_expert.state_dict(),
            "vggt_bridge": self.mot.vggt_bridge.state_dict(),
            "proprio_encoder": (
                None
                if self.proprio_encoder is None
                else self.proprio_encoder.state_dict()
            ),
            "high_noise_vggt_aggregator": self.routed_vggt_backbone.high_noise_vggt.aggregator.state_dict(),
            "low_noise_vggt_aggregator": self.routed_vggt_backbone.low_noise_vggt.aggregator.state_dict(),
            "metadata": self._checkpoint_metadata(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        """Strictly load a native dual-VGGT checkpoint; no single-VGGT fallback."""
        if optimizer is not None:
            raise ValueError(
                "ActionVGGTMoEFastWAM weight checkpoints do not contain optimizer "
                "state; use the Accelerate training-state checkpoint."
            )
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid ActionVGGTMoEFastWAM checkpoint: {path}.")
        expected_keys = {
            "format",
            "version",
            "action_expert",
            "vggt_bridge",
            "proprio_encoder",
            "high_noise_vggt_aggregator",
            "low_noise_vggt_aggregator",
            "metadata",
            "step",
            "torch_dtype",
        }
        missing = sorted(expected_keys - set(payload))
        unexpected = sorted(set(payload) - expected_keys)
        if missing or unexpected:
            raise ValueError(
                "ActionVGGTMoEFastWAM checkpoint key mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if (
            payload["format"] != self.CHECKPOINT_FORMAT
            or payload["version"] != self.CHECKPOINT_VERSION
        ):
            raise ValueError(
                "Unsupported ActionVGGTMoEFastWAM checkpoint format/version: "
                f"{payload['format']!r}/{payload['version']!r}."
            )
        expected_metadata = self._checkpoint_metadata()
        if payload["metadata"] != expected_metadata:
            raise ValueError(
                "ActionVGGTMoEFastWAM checkpoint metadata mismatch: "
                f"expected={expected_metadata}, actual={payload['metadata']}."
            )

        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.mot.vggt_bridge.load_state_dict(payload["vggt_bridge"], strict=True)
        for name, module in (
            (
                "high_noise_vggt_aggregator",
                self.routed_vggt_backbone.high_noise_vggt.aggregator,
            ),
            (
                "low_noise_vggt_aggregator",
                self.routed_vggt_backbone.low_noise_vggt.aggregator,
            ),
        ):
            try:
                module.load_state_dict(payload[name], strict=True)
            except RuntimeError as exc:
                raise RuntimeError(f"{name} checkpoint mismatch: {exc}") from exc
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


__all__ = ["ActionVGGTMoEFastWAM", "TimeRoutedVGGTBackbone"]
