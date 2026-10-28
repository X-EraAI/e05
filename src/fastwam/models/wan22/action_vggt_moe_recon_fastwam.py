from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .action_vggt_moe_fastwam import ActionVGGTMoEFastWAM
from .vggt_moe_recon_backbone import VGGTMoEReconBackbone
from .vggt_reconstruction_loss import compute_vggt_reconstruction_loss, canonical_reconstruction_config


class ActionVGGTMoEReconFastWAM(ActionVGGTMoEFastWAM):
    """Time-routed action model with current/future VGGT reconstruction.

    High-noise samples optionally use language and supervise configurable
    current/future geometry; low-noise samples use the language-conditioned
    VGGT and supervise configurable current/future geometry. Input modules, FP32 heads and the bridge
    share by default; each group can also have independent route parameters.
    """

    CHECKPOINT_FORMAT = "action_vggt_moe_recon_fastwam"
    CHECKPOINT_VERSION = 4
    LEGACY_CHECKPOINT_VERSION = 2
    BACKBONE_TYPE = VGGTMoEReconBackbone
    evaluation_mode = "action_only"
    supports_language_context_mask = True

    def __init__(
        self,
        *,
        action_expert,
        mot,
        vggt_backbone: VGGTMoEReconBackbone,
        text_dim: int,
        layer_pairs,
        time_route_boundary: float = 0.8333333333333334,
        proprio_dim=None,
        device="cpu",
        torch_dtype=torch.float32,
        train_vggt=True,
        vggt_fp32_storage_bf16_compute=False,
        action_train_shift=5.0,
        action_infer_shift=5.0,
        action_num_train_timesteps=1000,
        reconstruction_loss_config=None,
        lambda_vggt_recon=0.05,
        share_bridge=True,
        checkpoint_sharing_policy="strict",
        high_noise_reconstruction_target="current",
        low_noise_reconstruction_target="future",
        checkpoint_target_policy="strict",
        checkpoint_language_policy="strict",
        balance_route_action_weights=False,
    ):
        if not isinstance(vggt_backbone, VGGTMoEReconBackbone):
            raise TypeError(
                "ActionVGGTMoEReconFastWAM requires VGGTMoEReconBackbone."
            )
        if checkpoint_sharing_policy not in ("strict", "expand_shared"):
            raise ValueError("checkpoint_sharing_policy must be strict or expand_shared.")
        if high_noise_reconstruction_target not in ("current", "future"):
            raise ValueError("high_noise_reconstruction_target must be current or future.")
        if low_noise_reconstruction_target not in ("current", "future"):
            raise ValueError("low_noise_reconstruction_target must be current or future.")
        if checkpoint_target_policy not in ("strict", "allow_high_target_change", "allow_target_change"):
            raise ValueError("checkpoint_target_policy must be strict, allow_high_target_change or allow_target_change.")
        if checkpoint_language_policy not in ("strict", "allow_high_language_change"):
            raise ValueError("checkpoint_language_policy must be strict or allow_high_language_change.")
        self.checkpoint_sharing_policy = checkpoint_sharing_policy
        self.high_noise_reconstruction_target = high_noise_reconstruction_target
        self.low_noise_reconstruction_target = low_noise_reconstruction_target
        self.checkpoint_target_policy = checkpoint_target_policy
        self.checkpoint_language_policy = checkpoint_language_policy
        self.share_bridge = bool(share_bridge)
        if not self.share_bridge:
            from .vggt_route_bridge import VGGTRouteBridge
            mot.vggt_bridge = VGGTRouteBridge(mot.vggt_bridge)
        super().__init__(
            action_expert=action_expert,
            mot=mot,
            vggt_backbone=vggt_backbone,
            text_dim=text_dim,
            layer_pairs=layer_pairs,
            time_route_boundary=time_route_boundary,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            train_vggt=train_vggt,
            vggt_fp32_storage_bf16_compute=vggt_fp32_storage_bf16_compute,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
        )
        self.reconstruction_loss_config = canonical_reconstruction_config(reconstruction_loss_config or {})
        self.lambda_vggt_recon = float(lambda_vggt_recon)
        if type(balance_route_action_weights) is not bool:
            raise ValueError("balance_route_action_weights must be a bool.")
        self.balance_route_action_weights = balance_route_action_weights
        self.route_action_weight_scales = self._compute_route_action_weight_scales()
        self.vggt_backbone.camera_head.float()
        self.vggt_backbone.dense_head.float()

    def _compute_route_action_weight_scales(self):
        """Normalize E[w(t) | route] using the scheduler's uniform base variable.

        Integrate each conditional interval separately, so rare routes still
        receive the same quadrature resolution. No batch statistics or RNG are
        used, and all distributed ranks compute identical fixed scales.
        """
        if not self.balance_route_action_weights:
            return {"high": 1.0, "low": 1.0}
        scheduler = self.train_action_scheduler
        boundary = self.time_route_boundary
        u_boundary = boundary / (scheduler.shift * (1.0 - boundary) + boundary)
        grid = (torch.arange(16384, dtype=torch.float64) + 0.5) / 16384
        scales = {}
        for route, start, end in (("low", 0.0, u_boundary), ("high", u_boundary, 1.0)):
            u = start + (end - start) * grid
            sigma = scheduler.shift * u / (1.0 + (scheduler.shift - 1.0) * u)
            mean = scheduler.training_weight(sigma * scheduler.num_train_timesteps).double().mean()
            if not torch.isfinite(mean) or mean <= 0:
                raise ValueError(f"Cannot normalize action weights for {route} route.")
            scales[route] = 1.0 / mean.item()
        return scales

    def _action_training_weight(self, timestep, high_mask):
        weight = self.train_action_scheduler.training_weight(timestep)
        if self.balance_route_action_weights:
            weight = weight * torch.where(
                high_mask,
                weight.new_tensor(self.route_action_weight_scales["high"]),
                weight.new_tensor(self.route_action_weight_scales["low"]),
            )
        return weight

    def route_action_weighting_config(self):
        # None preserves the contract of old runs with balancing disabled.
        if not self.balance_route_action_weights:
            return None
        return {"method": "conditional_mean_v1", "train_shift": self.train_action_scheduler.shift,
                "time_route_boundary": self.time_route_boundary,
                "num_train_timesteps": self.train_action_scheduler.num_train_timesteps}

    @property
    def routed_vggt_backbone(self):
        return self.vggt_backbone

    @staticmethod
    def _select_sample(sample, indices):
        """Select route samples while preserving scalar/non-batch fields."""
        batch_size = int(sample["action"].shape[0])
        return {
            key: (
                value.index_select(0, indices.to(value.device))
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and int(value.shape[0]) == batch_size
                else value
            )
            for key, value in sample.items()
        }

    def _recon_loss(self, prediction, sample, prefix, valid_mask=None, metric_prefix=None):
        """Compute geometry loss for one route-local prediction batch.

        Route predictions use generic ``depth/depth_conf/pose_enc`` keys;
        reconstruction targets keep the existing current/future field names.
        """
        target = f"vggt_recon_{prefix}_"
        metric_prefix = prefix if metric_prefix is None else metric_prefix
        pred = {
            f"{prefix}_depth": prediction["depth"],
            f"{prefix}_depth_conf": prediction["depth_conf"],
            f"{prefix}_pose": prediction["pose_enc"],
        }
        # Connect every head even when confidence loss is off or future rows
        # are all padding. Empty reductions stay finite for excluded NaNs.
        graph_zero = sum(value.reshape(-1)[:0].float().sum() for value in pred.values())
        if valid_mask is not None:
            valid_mask = valid_mask.to(self.device, dtype=torch.bool)
            if not bool(valid_mask.any()):
                zero = graph_zero
                return zero, {
                    f"loss_vggt_{metric_prefix}_camera": zero.detach(),
                    f"loss_vggt_{metric_prefix}_depth": zero.detach(),
                    f"loss_vggt_{metric_prefix}_point": zero.detach(),
                    f"vggt_{metric_prefix}_valid_depth_ratio": zero.detach(),
                    f"vggt_{metric_prefix}_scene_scale": zero.detach(),
                }
            keep = valid_mask
            pred = {key: value[keep] for key, value in pred.items()}
            sample = self._select_sample(sample, keep.nonzero().flatten())

        loss, metrics = compute_vggt_reconstruction_loss(
            depth=pred[f"{prefix}_depth"],
            depth_conf=pred[f"{prefix}_depth_conf"],
            pose_enc=pred[f"{prefix}_pose"],
            gt_depth=sample[f"{target}depth"].to(self.device),
            gt_valid=sample[f"{target}depth_valid"].to(self.device),
            gt_intrinsics=sample[f"{target}intrinsics"].to(self.device),
            gt_extrinsics=sample[f"{target}extrinsics"].to(self.device),
            **self._reconstruction_loss_kwargs(sample),
        )
        renamed = {}
        for key, value in metrics.items():
            if key.startswith("loss_vggt_"):
                key = key.replace("loss_vggt_", f"loss_vggt_{metric_prefix}_", 1)
            elif key.startswith("vggt_"):
                key = key.replace("vggt_", f"vggt_{metric_prefix}_", 1)
            renamed[key] = value
        return loss + graph_zero, renamed

    def training_loss(self, sample, vggt_kv_by_layer=None):
        if vggt_kv_by_layer is not None:
            raise ValueError(
                "ActionVGGTMoEReconFastWAM requires live routed VGGT capture."
            )
        required = {
            "action",
            "context",
            "context_mask",
            "language_context_mask",
            "vggt_images",
        }
        required.update(
            {
                f"vggt_recon_{prefix}_{suffix}"
                for prefix in {self.high_noise_reconstruction_target, self.low_noise_reconstruction_target}
                for suffix in ("depth", "depth_valid", "intrinsics", "extrinsics")
            }
        )
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(
                f"ActionVGGTMoEReconFastWAM sample is missing {missing}."
            )

        action = sample["action"].to(self.device, self.torch_dtype)
        batch_size, action_horizon, _ = action.shape
        context, context_mask = self._prepare_context(
            sample["context"],
            sample["context_mask"],
            sample.get("proprio"),
            batch_size=batch_size,
            training=True,
        )
        language_context = sample["context"].to(self.device)
        language_mask = sample["language_context_mask"].to(self.device)

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(self.device, dtype=torch.bool)

        noise = torch.randn_like(action)
        timestep = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise, timestep)
        target_action = self.train_action_scheduler.training_target(action, noise, timestep)
        high_mask = self.high_noise_mask(timestep)

        reconstruction = self.vggt_backbone.forward_mixed_with_reconstruction(
            sample["vggt_images"].to(self.device, self.torch_dtype),
            high_mask,
            context=language_context,
            language_context_mask=language_mask,
        )
        pred_action = self._predict_action(
            noisy_action,
            timestep,
            context,
            context_mask,
            reconstruction["kv_by_layer"],
        )
        token_loss = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is None:
            per_sample = token_loss.mean(dim=1)
        else:
            valid = (~action_is_pad).to(token_loss.dtype)
            per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        weight = self._action_training_weight(timestep, high_mask).to(per_sample)
        weighted_loss = per_sample * weight
        loss_action = weighted_loss.mean()

        high_indices = reconstruction["high_indices"]
        low_indices = reconstruction["low_indices"]
        zero = loss_action.new_zeros(())
        def empty_metrics(prefix):
            return {
                **{f"loss_vggt_{prefix}_{key}": zero for key in ("camera", "depth", "point")},
                f"vggt_{prefix}_valid_depth_ratio": zero,
                f"vggt_{prefix}_scene_scale": zero,
            }
        high_prefix = ("current" if self.high_noise_reconstruction_target == "current" else "high_future")
        if self.low_noise_reconstruction_target == "current":
            high_prefix = f"high_{self.high_noise_reconstruction_target}"
        low_prefix = "future" if self.low_noise_reconstruction_target == "future" else "low_current"
        high_metrics = empty_metrics(high_prefix)
        low_metrics = empty_metrics(low_prefix)
        high_valid_ratio = zero

        if reconstruction["high_result"] is None:
            loss_high = zero
        else:
            high_sample = self._select_sample(sample, high_indices)
            high_valid = None
            if self.high_noise_reconstruction_target == "future":
                high_valid = ~high_sample.get(
                    "vggt_recon_future_is_pad",
                    torch.zeros(high_indices.shape[0], dtype=torch.bool),
                ).to(self.device, dtype=torch.bool)
                high_valid_ratio = high_valid.float().mean()
            loss_high, values = self._recon_loss(
                reconstruction["high_result"], high_sample,
                self.high_noise_reconstruction_target, high_valid,
                metric_prefix=high_prefix,
            )
            high_metrics.update(values)

        if reconstruction["low_result"] is None:
            loss_low = zero
            low_valid_ratio = zero
        else:
            low_sample = self._select_sample(sample, low_indices)
            low_valid = None
            if self.low_noise_reconstruction_target == "future":
                low_valid = ~low_sample.get(
                    "vggt_recon_future_is_pad",
                    torch.zeros(low_indices.shape[0], dtype=torch.bool),
                ).to(self.device, dtype=torch.bool)
            loss_low, values = self._recon_loss(
                reconstruction["low_result"],
                low_sample,
                self.low_noise_reconstruction_target,
                low_valid,
                metric_prefix=low_prefix,
            )
            low_metrics.update(values)
            low_valid_ratio = (low_valid.float().mean() if low_valid is not None else zero.new_ones(()))

        loss_recon = self.lambda_vggt_recon * 0.5 * (loss_high + loss_low)
        loss = loss_action + loss_recon + reconstruction["inactive_anchor"].to(loss_action.dtype)
        high_weighted = weighted_loss[high_mask].mean() if high_mask.any() else zero
        low_mask = ~high_mask
        low_weighted = weighted_loss[low_mask].mean() if low_mask.any() else zero
        metrics = {
            "loss_action": loss_action.detach(),
            "loss_action_high": high_weighted.detach(),
            "loss_action_low": low_weighted.detach(),
            "loss_vggt_recon": loss_recon.detach(),
            "route_high_fraction": high_mask.float().mean().detach(),
            "route_high_count": high_mask.sum().detach(),
            "route_low_count": low_mask.sum().detach(),
            f"vggt_{low_prefix}_valid_ratio": low_valid_ratio.detach(),
            **({"vggt_high_future_valid_ratio": high_valid_ratio.detach()}
               if self.high_noise_reconstruction_target == "future" else {}),
            **high_metrics,
            **low_metrics,
        }
        return loss, metrics

    @torch.no_grad()
    def infer_action(
        self,
        *,
        action_horizon,
        vggt_images,
        context,
        context_mask,
        language_context_mask=None,
        proprio=None,
        num_inference_steps=20,
        sigma_shift=None,
        seed=None,
        rand_device="cpu",
        return_reconstruction=False,
    ):
        self.eval()
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        batch_size = 1 if context.ndim == 2 else int(context.shape[0])
        raw_language = context if context.ndim == 3 else context.unsqueeze(0)
        context, context_mask = self._prepare_context(
            context,
            context_mask,
            proprio,
            batch_size=batch_size,
            training=False,
        )
        if language_context_mask is None:
            raise ValueError("Language-conditioned inference requires language_context_mask.")
        elif language_context_mask.ndim == 1:
            language_context_mask = language_context_mask.unsqueeze(0)
        language_context_mask = language_context_mask.to(self.device, dtype=torch.bool)
        raw_language = raw_language.to(self.device)
        if raw_language.shape[:2] != language_context_mask.shape:
            raise ValueError("Language context and mask must have matching [B,L] dimensions.")
        images = vggt_images.to(self.device, self.torch_dtype)
        # Cache patch tokens only when their embedding is shared. Independent
        # embeddings run on first route use; only active-route K/V stay cached.
        patches = frame_rope = None
        if self.vggt_backbone.share_patch_embedding:
            patches, frame_rope = self.vggt_backbone._prepare_patch_tokens(
                images, self.vggt_backbone.high_noise_vggt)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(self.device, self.torch_dtype)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        active_route = None
        memory = None
        high_reconstruction = None
        low_reconstruction = None
        def capture(route, reconstruct):
            # Shared patches are reused; independent patches are computed only
            # when this route is first visited. Prefix tokens belong to route.
            tokens, rope = self.vggt_backbone.prepare_route_tokens(
                images, route, patch_tokens=patches, frame_rope=frame_rope)
            use_language = route == "low" or self.vggt_backbone.high_noise_use_language_condition
            result = self.vggt_backbone._run_visual_route(
                tokens, images,
                aggregator=(self.vggt_backbone.high_noise_vggt if route == "high"
                            else self.vggt_backbone.low_noise_vggt),
                frame_rope=rope,
                context=raw_language if use_language else None,
                language_context_mask=language_context_mask if use_language else None,
                use_language=use_language, reconstruct=reconstruct,
            )
            for entry in result["kv_by_layer"].values():
                entry["high_noise_mask"] = torch.full(
                    (batch_size,), route == "high", device=self.device, dtype=torch.bool)
            return result

        for timestep, delta in zip(timesteps, deltas):
            route = self._route_for_scalar_timestep(timestep)
            if route != active_route:
                memory = None
                captured = capture(route, return_reconstruction)
                memory = captured["kv_by_layer"]
                captured.pop("kv_by_layer")
                if return_reconstruction:
                    if route == "high":
                        high_reconstruction = captured
                    else:
                        low_reconstruction = captured
                active_route = route
            pred_action = self._predict_action(
                action,
                timestep.expand(batch_size),
                context,
                context_mask,
                memory,
            )
            action = self.infer_action_scheduler.step(pred_action, delta, action)

        action = action.detach().cpu().float()
        if not return_reconstruction:
            return action

        # A short denoising schedule can visit only high. Explicit visualization
        # requests still return both geometries, without changing action output.
        memory = None
        if high_reconstruction is None:
            high_reconstruction = capture("high", True)
            high_reconstruction.pop("kv_by_layer")
        if low_reconstruction is None:
            low_reconstruction = capture("low", True)
            low_reconstruction.pop("kv_by_layer")

        def public(result, route, target):
            if result is None:
                return None
            return {
                **self._reconstruction_output_metadata(),
                "route": route,
                "target": target,
                "depth": result["depth"].detach().cpu().float(),
                "confidence": result["depth_conf"].detach().cpu().float(),
                "pose": result["pose_enc"].detach().cpu().float(),
            }

        return {
            "action": action,
            "reconstruction": {
                (f"high_{self.high_noise_reconstruction_target}"
                 if self.low_noise_reconstruction_target == "current"
                 else ("current" if self.high_noise_reconstruction_target == "current" else "high_future")):
                    public(high_reconstruction, "high", self.high_noise_reconstruction_target),
                ("low_current" if self.low_noise_reconstruction_target == "current"
                 else ("future" if self.high_noise_reconstruction_target == "current" else "low_future")):
                    public(low_reconstruction, "low", self.low_noise_reconstruction_target),
            },
        }

    def sharing_config(self):
        return {
            "share_patch_embedding": self.vggt_backbone.share_patch_embedding,
            "share_camera_register_tokens": self.vggt_backbone.share_camera_register_tokens,
            "share_geometry_heads": self.vggt_backbone.share_geometry_heads,
            "share_bridge": self.share_bridge,
        }

    def _checkpoint_metadata(self):
        bridge = self.mot.vggt_bridge
        return {
            "sharing": self.sharing_config(),
            "time_route_boundary": self.time_route_boundary,
            "route_rule": self.ROUTE_RULE,
            "num_train_timesteps": self.train_action_scheduler.num_train_timesteps,
            "layer_pairs": self.route_layer_pairs,
            "train_vggt": self.train_vggt,
            "high_noise_reconstruction_target": self.high_noise_reconstruction_target,
            "high_noise_use_language_condition": self.vggt_backbone.high_noise_use_language_condition,
            "low_noise_reconstruction_target": self.low_noise_reconstruction_target,
            "low_noise_use_language_condition": True,
            "lambda_vggt_recon": self.lambda_vggt_recon,
            **({"route_action_weighting": self.route_action_weighting_config()}
               if self.balance_route_action_weights else {}),
            "reconstruction_loss_config": dict(self.reconstruction_loss_config),
            **({"reconstruction_loss": self.reconstruction_loss_contract()}
               if self.reconstruction_loss_contract() is not None else {}),
            "vggt_camera_keys": tuple(self.vggt_backbone.input_camera_keys),
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
        if optimizer is not None:
            raise ValueError("Weight checkpoints do not contain optimizer state.")
        metadata = self._checkpoint_metadata()
        version = self.CHECKPOINT_VERSION
        if self.low_noise_reconstruction_target == "future":
            version = 3
            if not self.vggt_backbone.high_noise_use_language_condition:
                # Preserve the v2 payload for the default visual-only high route.
                metadata.pop("high_noise_use_language_condition")
                version = self.LEGACY_CHECKPOINT_VERSION
        torch.save(
            {
                "format": self.CHECKPOINT_FORMAT,
                "version": version,
                "action_expert": self.action_expert.state_dict(),
                "vggt_bridge": self.mot.vggt_bridge.state_dict(),
                "proprio_encoder": None if self.proprio_encoder is None else self.proprio_encoder.state_dict(),
                "vggt_backbone": self.vggt_backbone.state_dict(),
                "metadata": metadata,
                "step": step,
                "torch_dtype": str(self.torch_dtype),
            },
            path,
        )

    def load_checkpoint(self, path, optimizer=None):
        if optimizer is not None:
            raise ValueError("Weight checkpoints do not contain optimizer state.")
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        expected_keys = {
            "format", "version", "action_expert", "vggt_bridge",
            "proprio_encoder", "vggt_backbone", "metadata", "step", "torch_dtype",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise ValueError("Invalid ActionVGGTMoEReconFastWAM checkpoint keys.")
        if payload["format"] != self.CHECKPOINT_FORMAT or int(payload["version"]) not in (1, 2, 3, 4):
            raise ValueError(f"Unsupported ActionVGGTMoEReconFastWAM checkpoint: {path}")
        source_meta = dict(payload["metadata"])
        if int(payload["version"]) < 4:
            source_meta.setdefault("low_noise_reconstruction_target", "future")
        if int(payload["version"]) < 3:
            source_meta.setdefault("high_noise_use_language_condition", False)
        expected_meta = self._checkpoint_metadata()
        target_sharing = expected_meta.pop("sharing")
        if int(payload["version"]) == 1:
            source_sharing = {key: True for key in target_sharing}
        else:
            source_sharing = source_meta.pop("sharing", None)
        if (not isinstance(source_sharing, dict) or set(source_sharing) != set(target_sharing)
                or any(type(value) is not bool for value in source_sharing.values())):
            raise ValueError("Checkpoint sharing metadata is missing or invalid.")
        if self.checkpoint_target_policy in ("allow_high_target_change", "allow_target_change"):
            source_target = source_meta.get("high_noise_reconstruction_target")
            if source_target not in ("current", "future"):
                raise ValueError("Checkpoint high-noise reconstruction target is missing or invalid.")
            source_meta["high_noise_reconstruction_target"] = expected_meta["high_noise_reconstruction_target"]
        if self.checkpoint_target_policy == "allow_target_change":
            if source_meta.get("low_noise_reconstruction_target") not in ("current", "future"):
                raise ValueError("Checkpoint low-noise reconstruction target is missing or invalid.")
            source_meta["low_noise_reconstruction_target"] = expected_meta["low_noise_reconstruction_target"]
        if self.checkpoint_language_policy == "allow_high_language_change":
            source_language = source_meta.get("high_noise_use_language_condition")
            if type(source_language) is not bool:
                raise ValueError("Checkpoint high-noise language condition is missing or invalid.")
            source_meta["high_noise_use_language_condition"] = expected_meta["high_noise_use_language_condition"]
        if source_meta != expected_meta:
            differences = sorted(k for k in source_meta.keys() | expected_meta.keys()
                                 if source_meta.get(k) != expected_meta.get(k))
            raise ValueError(f"ActionVGGTMoEReconFastWAM checkpoint metadata mismatch: {differences}")
        changed = [key for key in target_sharing if source_sharing[key] != target_sharing[key]]
        if changed and self.checkpoint_sharing_policy != "expand_shared":
            raise ValueError(f"Checkpoint sharing mismatch: {changed}; use expand_shared for weights-only splitting.")
        if any(not source_sharing[key] for key in changed):
            raise ValueError(f"Cannot merge independent checkpoint modules into shared modules: {changed}")

        # v1 already stores both va1/va2 input keys. Removing aliases during
        # construction makes load_state_dict copy these into separate storage.
        backbone_state = dict(payload["vggt_backbone"])
        if "share_geometry_heads" in changed:
            for key, value in list(backbone_state.items()):
                if key.startswith(("camera_head.", "dense_head.")):
                    backbone_state["low_" + key] = value
        bridge_state = payload["vggt_bridge"]
        if "share_bridge" in changed:
            bridge_state = {f"{route}.{key}": value for route in ("high", "low")
                            for key, value in bridge_state.items()}
        payload["vggt_backbone"] = backbone_state
        payload["vggt_bridge"] = bridge_state
        # Validate every module before changing any weights.
        modules = {"action_expert": self.action_expert, "vggt_bridge": self.mot.vggt_bridge,
                   "vggt_backbone": self.vggt_backbone, "proprio_encoder": self.proprio_encoder}
        for name, module in modules.items():
            state = payload[name]
            if module is None:
                if state is not None:
                    raise ValueError(f"Checkpoint {name} configuration mismatch.")
                continue
            expected = module.state_dict()
            if not isinstance(state, dict) or set(state) != set(expected):
                raise ValueError(f"Checkpoint {name} state keys mismatch.")
            for key, value in expected.items():
                if not isinstance(state[key], torch.Tensor) or state[key].shape != value.shape:
                    raise ValueError(f"Checkpoint {name}.{key} shape mismatch.")
        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.mot.vggt_bridge.load_state_dict(payload["vggt_bridge"], strict=True)
        self.vggt_backbone.load_state_dict(payload["vggt_backbone"], strict=True)
        proprio_state = payload["proprio_encoder"]
        if (self.proprio_encoder is None) != (proprio_state is None):
            raise ValueError("Checkpoint proprio configuration mismatch.")
        if self.proprio_encoder is not None:
            self.proprio_encoder.load_state_dict(proprio_state, strict=True)
        return payload


__all__ = ["ActionVGGTMoEReconFastWAM"]
