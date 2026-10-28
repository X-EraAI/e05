from __future__ import annotations

"""ActionDiT with a language-conditioned geometry VGGT memory."""

import logging
import json
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F

from .action_vggt_fastwam import ActionVGGTFastWAM
from .vggt_reconstruction_loss import compute_vggt_reconstruction_loss, canonical_reconstruction_config

logger = logging.getLogger(__name__)


class E05(ActionVGGTFastWAM):
    """Single-stream ActionVGGT model for language-conditioned geometry.

    The action branch still receives the normal ActionDiT context (including
    the optional proprio token).  The VGGT branch receives only the original
    text context and its padding mask, so proprio is never accidentally
    treated as a language token.
    """

    MODEL_NAME = "e05"
    # Serialized identity is retained so the published 109k bytes stay intact.
    CHECKPOINT_FORMAT = "action_vggt_lang_recon_fastwam"
    CHECKPOINT_VERSION = 1
    evaluation_mode = "action_only"
    supports_language_context_mask = True
    requires_vggt_future_images = False

    def __init__(self, *args, vggt_camera_keys=(), reconstruction_target="future", **kwargs):
        if reconstruction_target not in {"current", "future"}:
            raise ValueError("reconstruction_target must be 'current' or 'future'.")
        self.reconstruction_target = reconstruction_target
        super().__init__(*args, **kwargs)
        if not self.train_vggt:
            # Full-state checkpoints must also retain a transferred, frozen
            # backbone. The parent keeps frozen VGGT outside Module._modules.
            backbone = self.vggt_backbone
            del self.vggt_backbone
            self.vggt_backbone = backbone
            self.vggt_backbone.requires_grad_(False)
        self.vggt_camera_keys = tuple(vggt_camera_keys)
        if self.reconstruction_loss_config is None:
            raise ValueError(
                "E05 requires reconstruction_loss_config."
            )
        if not hasattr(self.vggt_backbone, "language_projector"):
            raise TypeError(
                "E05 requires a language-conditioned VGGT backbone."
            )

    def _future_recon_loss(self, prediction, sample, valid_mask=None):
        """Compatibility entrypoint for existing future-only diagnostics."""
        return self._target_recon_loss(prediction, sample, valid_mask, target="future")

    def _target_recon_loss(self, prediction, sample, valid_mask=None, *, target=None):
        """Supervise the shared geometry heads at the configured target time.

        Backbone output keys retain their historical ``future_*`` names; they
        identify shared tensors, not a second prediction branch.
        """
        target = self.reconstruction_target if target is None else target
        prefix = f"vggt_recon_{target}"
        # Empty reductions are finite zero even if excluded rows contain NaN.
        # Keep every head in the graph on every rank, including all-pad ranks
        # and the confidence branch when confidence loss is disabled. Otherwise
        # ZeRO may launch different gradient collectives across ranks.
        graph_zero = sum(
            prediction[key].reshape(-1)[:0].float().sum()
            for key in ("future_depth", "future_depth_conf", "future_pose")
        )
        if valid_mask is not None:
            keep = valid_mask.to(device=self.device, dtype=torch.bool)
            if keep.shape != prediction["future_depth"].shape[:1]:
                raise ValueError("Reconstruction valid_mask must have shape [B].")
            if not bool(keep.any()):
                zero = graph_zero
                return zero, {
                    f"loss_vggt_{target}_camera": zero.detach(),
                    f"loss_vggt_{target}_depth": zero.detach(),
                    f"loss_vggt_{target}_point": zero.detach(),
                    f"vggt_{target}_valid_depth_ratio": zero.detach(),
                    f"vggt_{target}_scene_scale": zero.detach(),
                }
            prediction = {
                key: prediction[key][keep]
                for key in ("future_depth", "future_depth_conf", "future_pose")
            }
            sample = {
                key: value[keep.to(value.device)]
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == keep.shape[0]
                else value
                for key, value in sample.items()
            }

        loss, metrics = compute_vggt_reconstruction_loss(
            depth=prediction["future_depth"],
            depth_conf=prediction["future_depth_conf"],
            pose_enc=prediction["future_pose"],
            gt_depth=sample[f"{prefix}_depth"].to(self.device),
            gt_valid=sample[f"{prefix}_depth_valid"].to(self.device),
            gt_intrinsics=sample[f"{prefix}_intrinsics"].to(self.device),
            gt_extrinsics=sample[f"{prefix}_extrinsics"].to(self.device),
            **self._reconstruction_loss_kwargs(sample),
        )
        renamed = {}
        for key, value in metrics.items():
            if key.startswith("loss_vggt_"):
                key = key.replace("loss_vggt_", f"loss_vggt_{target}_", 1)
            elif key.startswith("vggt_"):
                key = key.replace("vggt_", f"vggt_{target}_", 1)
            renamed[key] = value
        return loss + graph_zero, renamed

    def _training_reconstructions(self, images, **kwargs):
        return [self.vggt_backbone.forward_with_reconstruction(images, **kwargs)]

    def training_loss(self, sample, vggt_kv_by_layer=None):
        """Train action flow and configured geometry from current RGB + text."""
        if vggt_kv_by_layer is not None:
            raise ValueError(
                "Language-conditioned reconstruction requires live VGGT capture."
            )
        required = {
            "action",
            "context",
            "context_mask",
            "language_context_mask",
            "vggt_images",
            *(f"vggt_recon_{self.reconstruction_target}_{key}"
              for key in ("depth", "depth_valid", "intrinsics", "extrinsics")),
        }
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(
                f"E05 sample is missing {missing}."
            )

        action = sample["action"].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        if action.ndim != 3 or action.shape[2] != self.action_expert.action_dim:
            raise ValueError(
                "`action` must be [B,T,action_dim], got "
                f"{tuple(action.shape)} with action_dim={self.action_expert.action_dim}."
            )
        batch_size, action_horizon, _ = action.shape
        context, context_mask = self._prepare_context(
            sample["context"],
            sample["context_mask"],
            sample.get("proprio"),
            batch_size=batch_size,
            training=True,
        )
        language_context = sample["context"].to(self.device)
        language_context_mask = sample["language_context_mask"].to(
            self.device, dtype=torch.bool
        )

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            if action_is_pad.shape != (batch_size, action_horizon):
                raise ValueError("`action_is_pad` must have shape [B,T].")
            action_is_pad = action_is_pad.to(self.device, dtype=torch.bool)

        reconstructions = self._training_reconstructions(
            sample["vggt_images"].to(self.device, self.torch_dtype),
            context=language_context,
            language_context_mask=language_context_mask,
        )
        noise = torch.randn_like(action)
        timestep = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise, timestep)
        target_action = self.train_action_scheduler.training_target(action, noise, timestep)
        valid_reconstruction = torch.ones(batch_size, device=self.device, dtype=torch.bool)
        if self.reconstruction_target == "future":
            valid_reconstruction = ~sample.get(
                "vggt_recon_future_is_pad",
                torch.zeros(batch_size, dtype=torch.bool),
            ).to(self.device, dtype=torch.bool)
        losses, round_metrics = [], []
        for reconstruction in reconstructions:
            # Each call starts ActionDiT from the SAME noisy action, never from
            # another pass's hidden state or prediction.
            pred_action = self._predict_action(
                noisy_action, timestep, context, context_mask,
                reconstruction["kv_by_layer"],
            )
            token_loss = F.mse_loss(
                pred_action.float(), target_action.float(), reduction="none"
            ).mean(dim=2)
            if action_is_pad is None:
                per_sample = token_loss.mean(dim=1)
            else:
                valid_action = (~action_is_pad).to(token_loss.dtype)
                per_sample = (token_loss * valid_action).sum(dim=1) / valid_action.sum(
                    dim=1
                ).clamp_min(1.0)
            weight = self.train_action_scheduler.training_weight(timestep).to(per_sample)
            loss_action = (per_sample * weight).mean()
            loss_recon_raw, recon_metrics = self._target_recon_loss(
                reconstruction, sample, valid_mask=valid_reconstruction,
            )
            loss_recon = self.lambda_vggt_recon * loss_recon_raw
            losses.append(loss_action + loss_recon)
            round_metrics.append({
                "loss_action": loss_action.detach(),
                "loss_vggt_recon": loss_recon.detach(),
                **recon_metrics,
                f"vggt_{self.reconstruction_target}_valid_ratio": valid_reconstruction.float().mean().detach(),
            })
        metrics = {
            key: sum(item[key] for item in round_metrics) / len(round_metrics)
            for key in round_metrics[0]
        }
        if getattr(self, "report_loop_metrics", False):
            for index, item in enumerate(round_metrics, 1):
                metrics.update({f"loop{index}/{key}": value for key, value in item.items()})
        return sum(losses) / len(losses), metrics

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
        """Run action inference using language-conditioned current observations."""
        self.eval()
        if action_horizon <= 0:
            raise ValueError("`action_horizon` must be positive.")
        # Preserve the caller's text padding before _prepare_context appends a
        # proprio token. Deployment always supplies the original language mask.
        if language_context_mask is None:
            language_context_mask = context_mask
        batch_size = 1 if context.ndim == 2 else int(context.shape[0])
        context, context_mask = self._prepare_context(
            context,
            context_mask,
            proprio,
            batch_size=batch_size,
            training=False,
        )
        if language_context_mask.ndim == 1:
            language_context_mask = language_context_mask.unsqueeze(0)
        language_context_mask = language_context_mask.to(self.device, dtype=torch.bool)
        language_length = int(language_context_mask.shape[1])
        if language_length > int(context.shape[1]):
            raise ValueError("`language_context_mask` is longer than `context`.")
        language_context = context[:, :language_length]

        images = vggt_images.to(self.device, self.torch_dtype)
        if return_reconstruction:
            reconstruction = self.vggt_backbone.forward_with_reconstruction(
                images,
                context=language_context,
                language_context_mask=language_context_mask,
            )
        else:
            reconstruction = None
            memory = self.vggt_backbone(
                images,
                context=language_context,
                language_context_mask=language_context_mask,
            )
        memory = reconstruction["kv_by_layer"] if reconstruction is not None else memory

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
        ).to(self.device, self.torch_dtype)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        for timestep, delta in zip(timesteps, deltas):
            action = self.infer_action_scheduler.step(
                self._predict_action(
                    action,
                    timestep.expand(batch_size),
                    context,
                    context_mask,
                    memory,
                ),
                delta,
                action,
            )
        action = action.detach().cpu().float()
        if reconstruction is None:
            return action
        return {
            "action": action,
            "reconstruction": {
                self.reconstruction_target: {
                    **self._reconstruction_output_metadata(),
                    "depth": reconstruction["future_depth"].detach().cpu().float(),
                    "confidence": reconstruction["future_depth_conf"].detach().cpu().float(),
                    "pose": reconstruction["future_pose"].detach().cpu().float(),
                }
            },
        }

    def _checkpoint_metadata(self):
        bridge = self.mot.vggt_bridge
        return {
            # These fields make it impossible to silently load a future-only
            # language checkpoint into a differently configured model.
            "future_reconstruction": self.reconstruction_target == "future",
            "reconstruction_target": self.reconstruction_target,
            "requires_vggt_future_images": False,
            "supports_language_context_mask": True,
            "text_dim": int(self.text_dim),
            "vggt_camera_keys": self.vggt_camera_keys,
            "layer_pairs": tuple(
                (bridge.vggt_layer_for(i), i)
                for i in range(len(self.action_expert.blocks))
                if bridge.vggt_layer_for(i) is not None
            ),
            "train_vggt": bool(self.train_vggt),
            "lambda_vggt_recon": float(self.lambda_vggt_recon),
            "reconstruction_loss_config": dict(self.reconstruction_loss_config),
            **({"reconstruction_loss": self.reconstruction_loss_contract()}
               if self.reconstruction_loss_contract() is not None else {}),
            "bridge_config": {
                "projection_mode": bridge.projection_mode,
                "projection_sharing": bridge.projection_sharing,
                "fastwam_width": int(bridge.fastwam_width),
                "vggt_width": int(bridge.vggt_width),
                "fastwam_num_heads": int(bridge.fastwam_num_heads),
                "vggt_num_heads": int(bridge.vggt_num_heads),
            },
        }

    def checkpoint_metadata(self):
        """JSON-compatible contract for full training-state restoration."""
        return json.loads(json.dumps({
            **self._checkpoint_metadata(),
            "format": self.CHECKPOINT_FORMAT,
            "training_state_version": 1,
            "action_dim": self.action_expert.action_dim,
            "proprio_dim": self.proprio_dim,
            "action_train_shift": self.train_action_scheduler.shift,
            "action_num_train_timesteps": self.train_action_scheduler.num_train_timesteps,
        }))

    def validate_training_state_metadata(self, metadata, world_size):
        architecture = dict(metadata.get("model_architecture") or {})
        architecture.setdefault("reconstruction_target", "future")
        if architecture != self.checkpoint_metadata():
            raise ValueError(
                "Language reconstruction full-state resume metadata mismatch. "
                "Use the original configuration; legacy state directories without "
                "this contract require a .pt weights checkpoint instead."
            )
        if metadata.get("world_size") != world_size:
            raise ValueError("Language reconstruction full-state resume requires identical world size.")

    def save_checkpoint(self, path, optimizer=None, step=None):
        if optimizer is not None:
            raise ValueError("Weight checkpoints do not contain optimizer state.")
        payload = {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            "target": self.reconstruction_target,
            "text_dim": int(self.text_dim),
            "action_expert": self.action_expert.state_dict(),
            "vggt_bridge": self.mot.vggt_bridge.state_dict(),
            "vggt_backbone": self.vggt_backbone.state_dict(),
            "metadata": self._checkpoint_metadata(),
            "proprio_encoder": (
                None
                if self.proprio_encoder is None
                else self.proprio_encoder.state_dict()
            ),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        torch.save(payload, path)

    def _checkpoint_transfer_formats(self):
        return {self.CHECKPOINT_FORMAT}

    def _checkpoint_transfer_fields(self):
        return {
            "vggt_camera_keys", "train_vggt", "lambda_vggt_recon",
            "reconstruction_loss_config", "reconstruction_loss",
            "reconstruction_target", "future_reconstruction",
        }

    def _read_checkpoint(self, path, *, for_transfer=False):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("Language reconstruction checkpoint must be a mapping.")
        expected = {
            "format",
            "version",
            "target",
            "text_dim",
            "action_expert",
            "vggt_bridge",
            "vggt_backbone",
            "metadata",
            "proprio_encoder",
            "step",
            "torch_dtype",
        }
        missing = sorted(expected - set(payload))
        unexpected = sorted(set(payload) - expected)
        if missing or unexpected:
            raise ValueError(
                "E05 checkpoint key mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if (
            payload["format"] not in (self._checkpoint_transfer_formats() if for_transfer else {self.CHECKPOINT_FORMAT})
            or int(payload["version"]) != self.CHECKPOINT_VERSION
            or payload["target"] not in {"current", "future"}
            or int(payload["text_dim"]) != int(self.text_dim)
        ):
            raise ValueError("Unsupported or incompatible language reconstruction checkpoint.")
        if not isinstance(payload["metadata"], Mapping):
            raise ValueError("Checkpoint metadata must be a mapping.")
        payload["metadata"] = dict(payload["metadata"])
        payload["metadata"].setdefault("reconstruction_target", "future")
        source_target = payload["metadata"]["reconstruction_target"]
        if (source_target != payload["target"]
                or payload["metadata"].get("future_reconstruction") != (source_target == "future")):
            raise ValueError("Checkpoint reconstruction target metadata mismatch.")
        if not for_transfer and source_target != self.reconstruction_target:
            raise ValueError("Checkpoint reconstruction target mismatch; use explicit pretrained transfer.")
        return payload

    def load_pretrained_checkpoint(
        self, path, *, reset_action_projections=False, reset_proprio_encoder=False,
        load_pretrained_action_expert=True, load_pretrained_bridge=True,
    ):
        """Transfer full-model weights, explicitly resetting embodiment-specific layers.

        Validate every component before mutating the model. Training settings and
        camera names and reconstruction target may change; the attention bridge and language contract
        must match. This is weights-only initialization, never training-state resume.

        Disabled components retain their current weights. The runtime calls this
        on a freshly constructed model, so they retain constructor initialization,
        including non-Linear parameters such as ActionDiT block modulation/norms.
        Disabling action loading also keeps its fresh encoder and output head;
        reset_action_projections then has no additional effect.
        """
        payload = self._read_checkpoint(path, for_transfer=True)
        target_metadata = self._checkpoint_metadata()
        source_metadata = payload["metadata"]
        if not isinstance(source_metadata, Mapping):
            raise ValueError("Checkpoint metadata must be a mapping.")
        transferable_fields = self._checkpoint_transfer_fields()

        def architectural(metadata):
            return {key: value for key, value in metadata.items()
                    if key not in transferable_fields}

        if architectural(source_metadata) != architectural(target_metadata):
            raise ValueError("Pretrained checkpoint architecture metadata mismatch.")

        source_proprio = payload["proprio_encoder"]
        if source_proprio is not None and (
            not isinstance(source_proprio, Mapping)
            or set(source_proprio) != {"weight", "bias"}
            or not isinstance(source_proprio["weight"], torch.Tensor)
            or source_proprio["weight"].ndim != 2
            or not isinstance(source_proprio["bias"], torch.Tensor)
            or source_proprio["bias"].shape != source_proprio["weight"].shape[:1]
        ):
            raise ValueError("Malformed pretrained proprio_encoder weights.")

        components = {
            "action_expert": self.action_expert,
            "vggt_bridge": self.mot.vggt_bridge,
            "vggt_backbone": self.vggt_backbone,
            "proprio_encoder": self.proprio_encoder,
        }
        action_reset_keys = {
            "action_encoder.weight", "action_encoder.bias", "head.weight", "head.bias"
        } if reset_action_projections else set()
        states = {}
        loaded_keys, reset_keys, retained_keys = [], [], []
        retained_components = set()
        if not load_pretrained_action_expert:
            retained_components.add("action_expert")
        if not load_pretrained_bridge:
            retained_components.add("vggt_bridge")
        for name, module in components.items():
            source = payload[name]
            if name in retained_components:
                retained_keys.extend(f"{name}.{key}" for key in module.state_dict())
                continue
            if name == "proprio_encoder" and reset_proprio_encoder:
                if module is not None:
                    reset_keys.extend(f"{name}.{key}" for key in module.state_dict())
                continue
            if module is None:
                if source is not None:
                    raise ValueError(
                        f"Unexpected {name} weights; use reset_proprio_encoder for transfer."
                    )
                continue
            if not isinstance(source, Mapping):
                raise ValueError(f"Checkpoint is missing required {name} weights.")
            target = module.state_dict()
            missing, unexpected = set(target) - set(source), set(source) - set(target)
            if missing or unexpected:
                raise ValueError(
                    f"Pretrained {name} key mismatch: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
                )
            skip = action_reset_keys if name == "action_expert" else set()
            states[name] = {}
            for key, value in source.items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"Pretrained {name}.{key} must be a tensor.")
                if key in skip:
                    if value.ndim != target[key].ndim:
                        raise ValueError(f"Pretrained shape mismatch for {name}.{key}.")
                    reset_keys.append(f"{name}.{key}")
                    continue
                if value.shape != target[key].shape:
                    raise ValueError(
                        f"Pretrained shape mismatch for {name}.{key}: "
                        f"checkpoint={tuple(value.shape)}, target={tuple(target[key].shape)}. "
                        "Only explicitly reset action/proprio projections may change shape."
                    )
                states[name][key] = value
                loaded_keys.append(f"{name}.{key}")

        # No parameter replacement: optimizers created afterwards see these same
        # Parameter objects, and all target device/dtype choices are preserved.
        if reset_action_projections and load_pretrained_action_expert:
            self.action_expert.action_encoder.reset_parameters()
            self.action_expert.head.reset_parameters()
        if reset_proprio_encoder and self.proprio_encoder is not None:
            self.proprio_encoder.reset_parameters()
        for name, state in states.items():
            components[name].load_state_dict(
                state, strict=not (name == "action_expert" and reset_action_projections)
            )

        report = {
            "path": str(path),
            "source_action_dim": int(payload["action_expert"]["action_encoder.weight"].shape[1]),
            "target_action_dim": int(self.action_expert.action_dim),
            "source_proprio_dim": (
                None if source_proprio is None else int(source_proprio["weight"].shape[1])
            ),
            "target_proprio_dim": self.proprio_dim,
            "loaded_keys": sorted(loaded_keys),
            "reset_keys": sorted(reset_keys),
            "retained_keys": sorted(retained_keys),
            "metadata_overrides": {
                key: {"source": source_metadata.get(key), "target": target_metadata.get(key)}
                for key in sorted(transferable_fields)
                if source_metadata.get(key) != target_metadata.get(key)
            },
        }
        self.pretrained_load_report = report
        logger.info(
            "Loaded full-model pretrained checkpoint %s: action %s -> %s, proprio %s -> %s; "
            "%d tensors loaded, reset=%s, retained components=%s (%d tensors); "
            "metadata overrides=%s",
            path, report["source_action_dim"], report["target_action_dim"],
            report["source_proprio_dim"], report["target_proprio_dim"],
            len(loaded_keys), report["reset_keys"], sorted(retained_components),
            len(retained_keys), sorted(report["metadata_overrides"]),
        )
        return report

    def load_checkpoint(self, path, optimizer=None):
        if optimizer is not None:
            raise ValueError("Weight checkpoints do not contain optimizer state.")
        payload = self._read_checkpoint(path)
        expected_metadata = self._checkpoint_metadata()
        # Early v1 weights predate both camera-order and layer-pair metadata.
        # They must be instantiated using their original saved configuration;
        # all recorded fields and the tensor state dictionaries remain strict.
        # A partially missing modern contract is still rejected.
        if not ({"vggt_camera_keys", "layer_pairs"} & payload["metadata"].keys()):
            expected_metadata.pop("vggt_camera_keys")
            expected_metadata.pop("layer_pairs")
            logger.warning(
                "Loading legacy LangRecon weights without camera/layer metadata; "
                "use the checkpoint's original saved model configuration."
            )
        if payload["metadata"] != expected_metadata:
            raise ValueError(
                "E05 checkpoint metadata mismatch."
            )
        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.mot.vggt_bridge.load_state_dict(payload["vggt_bridge"], strict=True)
        self.vggt_backbone.load_state_dict(payload["vggt_backbone"], strict=True)
        if self.proprio_encoder is None:
            if payload["proprio_encoder"] is not None:
                raise ValueError("Checkpoint contains unexpected proprio weights.")
        else:
            if payload["proprio_encoder"] is None:
                raise ValueError("Checkpoint is missing required proprio weights.")
            self.proprio_encoder.load_state_dict(
                payload["proprio_encoder"], strict=True
            )
        return payload


__all__ = ["E05"]
