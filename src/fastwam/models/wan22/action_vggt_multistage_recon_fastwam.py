"""Equal-probability diffusion stages with independent VGGT experts."""
from __future__ import annotations

from pathlib import Path
import math

import torch
from torch.nn import functional as F

from .action_vggt_fastwam import ActionVGGTFastWAM
from .vggt_multistage_recon_backbone import VGGTMultiStageReconBackbone
from .vggt_multistage_route_bridge import VGGTMultiStageRouteBridge
from .vggt_reconstruction_loss import compute_vggt_reconstruction_loss


def resolve_stage_options(num_stages, use_language_condition=True, reconstruction_target="future",
                          stage_use_language_condition=None, stage_reconstruction_targets=None):
    if type(num_stages) is not int or num_stages < 2:
        raise ValueError("num_stages must be an integer >= 2.")
    language = list(stage_use_language_condition) if stage_use_language_condition is not None else [use_language_condition] * num_stages
    targets = list(stage_reconstruction_targets) if stage_reconstruction_targets is not None else [reconstruction_target] * num_stages
    if len(language) != num_stages or any(type(v) is not bool for v in language):
        raise ValueError("Expected one boolean language switch per stage.")
    if len(targets) != num_stages or any(v not in ("current", "future") for v in targets):
        raise ValueError("Expected one current/future reconstruction target per stage.")
    return language, targets


class ActionVGGTMultiStageReconFastWAM(ActionVGGTFastWAM):
    CHECKPOINT_FORMAT = "action_vggt_multistage_recon_fastwam"
    CHECKPOINT_VERSION = 1
    supports_language_context_mask = True

    def __init__(self, *, num_stages=3, routing_partition="equal_probability",
                 use_language_condition=True, reconstruction_target="future",
                 stage_use_language_condition=None, stage_reconstruction_targets=None,
                 layer_pairs, share_bridge=True, balance_route_action_weights=False, **kwargs):
        language, targets = resolve_stage_options(num_stages, use_language_condition, reconstruction_target,
                                                 stage_use_language_condition, stage_reconstruction_targets)
        if routing_partition != "equal_probability":
            raise ValueError("Only equal_probability routing is supported.")
        if type(balance_route_action_weights) is not bool:
            raise ValueError("balance_route_action_weights must be bool.")
        if kwargs.get("reconstruction_loss_config") is None:
            raise ValueError("MultiStageRecon requires reconstruction_loss_config.")
        backbone = kwargs["vggt_backbone"]
        if not isinstance(backbone, VGGTMultiStageReconBackbone) or len(backbone.stage_names) != num_stages:
            raise ValueError("Expected an N-stage reconstruction backbone.")
        if tuple(language) != backbone.stage_use_language_condition:
            raise ValueError("Model/backbone language switches differ.")
        self.num_stages = num_stages
        self.stage_names = backbone.stage_names
        self.stage_reconstruction_targets = tuple(targets)
        self.route_layer_pairs = tuple(tuple(int(x) for x in p) for p in layer_pairs)
        if tuple(p[0] for p in self.route_layer_pairs) != backbone.callback_layer_indices:
            raise ValueError("Backbone capture layers and layer_pairs differ.")
        self.share_bridge = bool(share_bridge)
        self.balance_route_action_weights = balance_route_action_weights
        if not share_bridge:
            kwargs["mot"].vggt_bridge = VGGTMultiStageRouteBridge(kwargs["mot"].vggt_bridge, self.stage_names)
        super().__init__(**kwargs)
        shift = self.train_action_scheduler.shift
        if not math.isfinite(shift):
            raise ValueError("train_shift must be finite.")
        self.time_route_boundaries = tuple(shift * (i / num_stages) / (1 + (shift - 1) * (i / num_stages))
                                          for i in range(1, num_stages))
        self.route_action_weight_scales = self._compute_route_action_weight_scales()

    def stage_ids_for(self, timestep):
        sigma = timestep.float() / self.train_action_scheduler.num_train_timesteps
        if not bool(torch.isfinite(sigma).all()) or bool(((sigma < 0) | (sigma > 1)).any()):
            raise ValueError("Diffusion timesteps must be finite and in [0,T].")
        boundaries = sigma.new_tensor(self.time_route_boundaries)
        return self.num_stages - 1 - torch.bucketize(sigma.contiguous(), boundaries, right=True)

    def _compute_route_action_weight_scales(self):
        if not self.balance_route_action_weights:
            return [1.] * self.num_stages
        scheduler = self.train_action_scheduler
        grid = (torch.arange(16384, dtype=torch.float64) + .5) / 16384
        scales = []
        for i in range(self.num_stages):
            u = (self.num_stages - 1 - i + grid) / self.num_stages
            sigma = scheduler.shift * u / (1 + (scheduler.shift - 1) * u)
            mean = scheduler.training_weight(sigma * scheduler.num_train_timesteps).double().mean()
            if not torch.isfinite(mean) or mean <= 0:
                raise ValueError("Cannot normalize stage action weights.")
            scales.append(1. / mean.item())
        return scales

    def route_action_weighting_config(self):
        return {"enabled": self.balance_route_action_weights, "method": "conditional_mean_v1",
                "train_shift": self.train_action_scheduler.shift, "boundaries": list(self.time_route_boundaries),
                "num_train_timesteps": self.train_action_scheduler.num_train_timesteps}

    def _action_training_weight(self, timestep, stage_ids):
        weight = self.train_action_scheduler.training_weight(timestep)
        return weight * weight.new_tensor(self.route_action_weight_scales)[stage_ids]

    def training_loss(self, sample, vggt_kv_by_layer=None):
        if vggt_kv_by_layer is not None:
            raise ValueError("MultiStageRecon requires live VGGT capture.")
        required = {"action", "context", "context_mask", "language_context_mask", "vggt_images"}
        required.update(f"vggt_recon_{target}_{key}" for target in set(self.stage_reconstruction_targets)
                        for key in ("depth", "depth_valid", "intrinsics", "extrinsics"))
        missing = required - sample.keys()
        if missing:
            raise ValueError(f"Missing sample fields: {sorted(missing)}")
        action = sample["action"].to(self.device, self.torch_dtype)
        context, context_mask = self._prepare_context(sample["context"], sample["context_mask"],
            sample.get("proprio"), batch_size=len(action), training=True)
        noise = torch.randn_like(action)
        timestep = self.train_action_scheduler.sample_training_t(batch_size=len(action), device=self.device, dtype=action.dtype)
        stage_ids = self.stage_ids_for(timestep)
        reconstruction = self.vggt_backbone.forward_mixed_with_reconstruction(
            sample["vggt_images"].to(self.device, self.torch_dtype), stage_ids,
            context=sample["context"].to(self.device), language_context_mask=sample["language_context_mask"].to(self.device))
        prediction = self._predict_action(self.train_action_scheduler.add_noise(action, noise, timestep), timestep,
                                          context, context_mask, reconstruction["kv_by_layer"])
        target = self.train_action_scheduler.training_target(action, noise, timestep)
        token_loss = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(2)
        pad = sample.get("action_is_pad")
        if pad is None:
            per_sample = token_loss.mean(1)
        else:
            valid = (~pad.to(self.device, dtype=torch.bool)).to(token_loss)
            per_sample = (token_loss * valid).sum(1) / valid.sum(1).clamp_min(1.)
        weighted = per_sample * self._action_training_weight(timestep, stage_ids)
        loss_action = weighted.mean()
        zero = loss_action.new_zeros(())
        loss_recon = zero
        # Detached numerators and denominators are globally summed once. Every
        # rank returns identical stage metrics, so the trainer's mean is inert.
        numerators, denominators, metric_names = [], [], []
        for i, name in enumerate(self.stage_names):
            index = reconstruction["indices"][name]
            count = zero.new_tensor(index.numel())
            prediction = reconstruction["results"][name]
            local_metrics = {f"loss_vggt_{name}_{k}": zero for k in ("camera", "depth", "point")}
            local_metrics.update({f"vggt_{name}_valid_depth_ratio": zero, f"vggt_{name}_scene_scale": zero})
            route_loss, valid_count = zero, count
            if prediction is not None:
                selected = self._select_sample(sample, index)
                target = self.stage_reconstruction_targets[i]
                valid = None
                if target == "future":
                    valid = ~selected.get("vggt_recon_future_is_pad", torch.zeros(len(index), dtype=torch.bool)).to(self.device, dtype=torch.bool)
                    valid_count = valid.sum().to(zero)
                route_loss, values = self._recon_loss(prediction, selected, target, valid_mask=valid, metric_prefix=name)
                local_metrics.update(values)
            loss_recon = loss_recon + route_loss / self.num_stages
            stage_metrics = {f"route_{name}_count": (count, zero.new_ones(())),
                             f"route_{name}_fraction": (count, zero.new_tensor(len(action))),
                             f"loss_action_{name}": (weighted.index_select(0, index).sum(), count),
                             f"route_{name}_valid_reconstruction_count": (valid_count, zero.new_ones(()))}
            stage_metrics.update({key: (value * valid_count, valid_count) for key, value in local_metrics.items()})
            for key, (num, den) in stage_metrics.items():
                metric_names.append(key); numerators.append(num.detach()); denominators.append(den.detach())
        loss_recon = self.lambda_vggt_recon * loss_recon
        stats = torch.stack([torch.stack(numerators), torch.stack(denominators)])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
        metrics = {}
        for i, key in enumerate(metric_names):
            metrics[key] = stats[0, i] if key.endswith("_count") else stats[0, i] / stats[1, i].clamp_min(1)
        metrics.update(loss_action=loss_action.detach(), loss_vggt_recon=loss_recon.detach())
        loss = loss_action + loss_recon + reconstruction["inactive_anchor"].to(loss_action)
        return loss, metrics

    @torch.no_grad()
    def infer_action(self, *, action_horizon, vggt_images, context, context_mask,
                     language_context_mask=None, proprio=None, num_inference_steps=20,
                     sigma_shift=None, seed=None, rand_device="cpu", return_reconstruction=False):
        self.eval()
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        raw_language = context.unsqueeze(0) if context.ndim == 2 else context
        batch_size = len(raw_language)
        context, context_mask = self._prepare_context(context, context_mask, proprio, batch_size=batch_size, training=False)
        if language_context_mask is None:
            if any(self.vggt_backbone.stage_use_language_condition):
                raise ValueError("Language-conditioned inference requires language_context_mask.")
            language_context_mask = torch.ones(raw_language.shape[:2], dtype=torch.bool)
        if language_context_mask.ndim == 1:
            language_context_mask = language_context_mask.unsqueeze(0)
        if language_context_mask.shape != raw_language.shape[:2]:
            raise ValueError("Language context/mask shape mismatch.")
        raw_language = raw_language.to(self.device)
        language_context_mask = language_context_mask.to(self.device, dtype=torch.bool)
        images = vggt_images.to(self.device, self.torch_dtype)
        backbone = self.vggt_backbone
        from .vggt_mot import validate_vggt_images
        validate_vggt_images(images, batch_size=batch_size, patch_size=int(backbone.aggregator.patch_size))
        patches = rope = None
        with backbone._autocast_context(images.device):
            if backbone.share_patch_embedding:
                patches, rope = backbone._prepare_patch_tokens(images, backbone.aggregator)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn((batch_size, action_horizon, self.action_expert.action_dim),
                             generator=generator, device=rand_device).to(self.device, self.torch_dtype)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps, device=self.device, dtype=action.dtype, shift_override=sigma_shift)
        stage_schedule = self.stage_ids_for(timesteps).tolist()
        reconstructions = {}
        visits = []
        def capture(i, reconstruct):
            name = self.stage_names[i]
            with backbone._autocast_context(images.device):
                tokens, frame_rope = backbone.prepare_route_tokens(images, name, patch_tokens=patches, frame_rope=rope)
                result = backbone._run_visual_route(tokens, images, route=name, frame_rope=frame_rope,
                    context=raw_language, language_context_mask=language_context_mask,
                    use_language=backbone.stage_use_language_condition[i], reconstruct=reconstruct)
            for entry in result["kv_by_layer"].values():
                entry["stage_ids"] = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            return result
        def public(i, result):
            return {**self._reconstruction_output_metadata(), "route": self.stage_names[i],
                    "target": self.stage_reconstruction_targets[i],
                    "depth": result["depth"].detach().cpu().float(),
                    "confidence": result["depth_conf"].detach().cpu().float(),
                    "pose": result["pose_enc"].detach().cpu().float()}
        memory, active = None, None
        for timestep, delta, i in zip(timesteps, deltas, stage_schedule):
            if active != i:
                memory = None
                result = capture(i, return_reconstruction)
                memory = result.pop("kv_by_layer")
                if return_reconstruction:
                    reconstructions[self.stage_names[i]] = public(i, result)
                del result
                active = i
                visits.append(self.stage_names[i])
            action = self.infer_action_scheduler.step(
                self._predict_action(action, timestep.expand(batch_size), context, context_mask, memory), delta, action)
        # Diagnostic state is never read by forward or checkpoint recomputation.
        self.last_inference_routing = {"visited_stages": visits, "stage_steps": {
            name: stage_schedule.count(i) for i, name in enumerate(self.stage_names)}, "vggt_forwards": len(visits)}
        action = action.detach().cpu().float()
        if not return_reconstruction:
            return action
        memory = None
        for i, name in enumerate(self.stage_names):
            if name not in reconstructions:
                result = capture(i, True)
                reconstructions[name] = public(i, result)
                del result
        return {"action": action, "reconstruction": {name: reconstructions[name] for name in self.stage_names},
                "routing": self.last_inference_routing}

    def sharing_config(self):
        return {**{name: getattr(self.vggt_backbone, name) for name in
                   ("share_patch_embedding", "share_camera_register_tokens", "share_geometry_heads")},
                "share_bridge": self.share_bridge}

    def checkpoint_metadata(self):
        bridge = self.mot.vggt_bridge
        return {"format": self.CHECKPOINT_FORMAT, "version": self.CHECKPOINT_VERSION,
                "num_stages": self.num_stages, "stage_names": list(self.stage_names),
                "routing_partition": "equal_probability", "boundaries": list(self.time_route_boundaries),
                "boundary_rule": "equal_goes_to_higher_noise", "train_shift": self.train_action_scheduler.shift,
                "num_train_timesteps": self.train_action_scheduler.num_train_timesteps,
                "stage_use_language_condition": list(self.vggt_backbone.stage_use_language_condition),
                "stage_reconstruction_targets": list(self.stage_reconstruction_targets),
                "sharing": self.sharing_config(), "layer_pairs": [list(p) for p in self.route_layer_pairs],
                "route_action_weighting": self.route_action_weighting_config(),
                "reconstruction_loss_config": dict(self.reconstruction_loss_config),
                "lambda_vggt_recon": self.lambda_vggt_recon, "train_vggt": self.train_vggt,
                "reconstruction_loss": self.reconstruction_loss_contract(),
                "vggt_camera_keys": list(self.vggt_backbone.input_camera_keys),
                "action_dim": self.action_expert.action_dim, "text_dim": self.text_dim, "proprio_dim": self.proprio_dim,
                "bridge_config": {key: getattr(bridge, key) for key in ("projection_mode", "projection_sharing",
                    "fastwam_width", "vggt_width", "fastwam_num_heads", "vggt_num_heads")}}

    def validate_training_state_metadata(self, metadata, world_size):
        if metadata.get("model_architecture") != self.checkpoint_metadata():
            raise ValueError("MultiStageRecon full-state architecture mismatch.")
        if metadata.get("world_size") != world_size:
            raise ValueError("MultiStageRecon full-state world size mismatch.")

    def save_checkpoint(self, path, optimizer=None, step=None):
        if optimizer is not None:
            raise ValueError("Weight checkpoints do not contain optimizer state.")
        torch.save({"format": self.CHECKPOINT_FORMAT, "version": self.CHECKPOINT_VERSION,
                    "metadata": self.checkpoint_metadata(), "step": step,
                    **{name: None if module is None else module.state_dict() for name, module in self._checkpoint_modules().items()}}, path)

    def _checkpoint_modules(self):
        return {"action_expert": self.action_expert, "vggt_bridge": self.mot.vggt_bridge,
                "vggt_backbone": self.vggt_backbone, "proprio_encoder": self.proprio_encoder}

    def load_checkpoint(self, path, optimizer=None):
        if optimizer is not None:
            raise ValueError("Weight checkpoints do not contain optimizer state.")
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != self.CHECKPOINT_FORMAT or payload.get("version") != self.CHECKPOINT_VERSION:
            raise ValueError("Expected an independent MultiStageRecon checkpoint.")
        if payload.get("metadata") != self.checkpoint_metadata():
            raise ValueError("MultiStageRecon checkpoint metadata mismatch.")
        # Preflight all modules before changing any weights.
        for name, module in self._checkpoint_modules().items():
            state = payload.get(name)
            if module is None:
                if state is not None:
                    raise ValueError(f"Checkpoint {name} mismatch.")
            else:
                expected = module.state_dict()
                if not isinstance(state, dict) or set(state) != set(expected):
                    raise ValueError(f"Checkpoint {name} keys mismatch.")
                if any(not isinstance(state[k], torch.Tensor) or state[k].shape != v.shape for k, v in expected.items()):
                    raise ValueError(f"Checkpoint {name} shape mismatch.")
        for name, module in self._checkpoint_modules().items():
            if module is not None:
                module.load_state_dict(payload[name], strict=True)
        return payload

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
