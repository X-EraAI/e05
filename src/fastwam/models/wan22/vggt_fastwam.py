from __future__ import annotations

import logging
from pathlib import Path

import torch

from .fastwam import FastWAM
from .vggt_mot import FrozenVGGTBackbone, validate_vggt_images

logger = logging.getLogger(__name__)


class VGGTFastWAM(FastWAM):
    """FastWAM with a frozen or trainable VGGT Aggregator backbone."""

    def attach_vggt_backbone(
        self,
        vggt_backbone: FrozenVGGTBackbone,
        *,
        train_vggt: bool = False,
    ):
        if hasattr(self, "vggt_backbone"):
            raise RuntimeError("VGGT backbone is already attached.")
        self.train_vggt = bool(train_vggt)
        self.mot.vggt_bridge.detach_vggt_memory = not self.train_vggt
        if self.train_vggt:
            self.vggt_backbone = vggt_backbone
        else:
            object.__setattr__(self, "vggt_backbone", vggt_backbone)
        self.vggt_backbone.to(device=self.device, dtype=self.torch_dtype)
        self.vggt_backbone.train(self.training and self.train_vggt)
        return self

    @classmethod
    def from_fastwam(
        cls,
        base: FastWAM,
        vggt_backbone: FrozenVGGTBackbone,
        *,
        train_vggt: bool = False,
    ):
        """Legacy ownership transfer; runtime constructs ``cls`` directly."""
        if type(base) is not FastWAM:
            # Tiny test doubles cannot be reconstructed through the production factory.
            model = cls.__new__(cls)
            model.__dict__ = base.__dict__.copy()
            model._modules = base._modules.copy()
            model._parameters = base._parameters.copy()
            model._buffers = base._buffers.copy()
            return model.attach_vggt_backbone(
                vggt_backbone,
                train_vggt=train_vggt,
            )
        raise RuntimeError(
            "Production promotion is disabled; construct VGGTFastWAM directly via "
            "VGGTFastWAM.from_wan22_pretrained()."
        )

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        backbone = getattr(self, "vggt_backbone", None)
        if backbone is not None and not getattr(self, "train_vggt", False):
            backbone._apply(fn, recurse=recurse)
            backbone.eval()
        bridge = getattr(self.mot, "vggt_bridge", None)
        parameter = next(bridge.parameters(), None) if bridge is not None else None
        if parameter is None and backbone is not None:
            parameter = next(backbone.parameters(), None)
        if parameter is None:
            parameter = next(super().parameters(), None)
        if parameter is not None:
            self.device = parameter.device
            if parameter.is_floating_point():
                self.torch_dtype = parameter.dtype
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        backbone = getattr(self, "vggt_backbone", None)
        if backbone is not None and not getattr(self, "train_vggt", False):
            backbone.eval()
        return self

    def _capture_inference_vggt(
        self,
        vggt_images,
        existing=None,
        *,
        batch_size=None,
    ):
        if existing is not None:
            return existing
        if vggt_images is None:
            raise ValueError("VGGTFastWAM inference requires `vggt_images`.")
        validate_vggt_images(vggt_images, batch_size=batch_size)
        images = vggt_images.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        return self.vggt_backbone(images)

    def infer_joint(self, *, vggt_images=None, vggt_kv_by_layer=None, **kwargs):
        input_image = kwargs.get("input_image")
        batch_size = None if input_image is None else int(input_image.shape[0])
        memory = self._capture_inference_vggt(
            vggt_images,
            vggt_kv_by_layer,
            batch_size=batch_size,
        )
        return super().infer_joint(
            vggt_images=vggt_images, vggt_kv_by_layer=memory, **kwargs
        )

    def infer_action(self, *, vggt_images=None, vggt_kv_by_layer=None, **kwargs):
        input_image = kwargs.get("input_image")
        batch_size = None if input_image is None else int(input_image.shape[0])
        memory = self._capture_inference_vggt(
            vggt_images,
            vggt_kv_by_layer,
            batch_size=batch_size,
        )
        return super().infer_action(
            vggt_images=vggt_images, vggt_kv_by_layer=memory, **kwargs
        )

    def infer(self, *, vggt_images=None, vggt_kv_by_layer=None, **kwargs):
        input_image = kwargs.get("input_image")
        batch_size = None if input_image is None else int(input_image.shape[0])
        memory = self._capture_inference_vggt(
            vggt_images,
            vggt_kv_by_layer,
            batch_size=batch_size,
        )
        return super().infer(
            vggt_images=vggt_images, vggt_kv_by_layer=memory, **kwargs
        )

    def _capture_vggt(self, sample):
        if "vggt_images" not in sample:
            raise ValueError("VGGTFastWAM requires `sample['vggt_images']`.")
        images = sample["vggt_images"]
        reference = sample.get("video", sample.get("action"))
        batch_size = None if reference is None else int(reference.shape[0])
        validate_vggt_images(images, batch_size=batch_size)
        images = images.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        return self.vggt_backbone(images)

    def training_loss(self, sample, tiled: bool = False, vggt_kv_by_layer=None):
        if self.train_vggt and vggt_kv_by_layer is not None:
            raise ValueError(
                "Trainable VGGT requires live backbone capture during training; "
                "precomputed `vggt_kv_by_layer` is not supported."
            )
        memory = vggt_kv_by_layer if vggt_kv_by_layer is not None else self._capture_vggt(sample)
        return super().training_loss(sample, tiled=tiled, vggt_kv_by_layer=memory)

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {"mot": self.mot.state_dict(), "vggt_bridge": self.mot.vggt_bridge.state_dict(), "step": step, "torch_dtype": str(self.torch_dtype)}
        # `mot` already contains bridge keys; keep one canonical bridge section only.
        payload["mot"] = {k: v for k, v in payload["mot"].items() if not k.startswith("vggt_bridge.")}
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.train_vggt:
            payload["vggt_aggregator"] = self.vggt_backbone.aggregator.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if "mot" not in payload:
            raise ValueError(f"Checkpoint missing `mot`: {path}")
        result = self.mot.load_state_dict(payload["mot"], strict=False)
        unexpected = [k for k in result.unexpected_keys if not k.startswith("vggt_bridge.")]
        missing = [k for k in result.missing_keys if not k.startswith("vggt_bridge.")]
        if unexpected or missing:
            raise RuntimeError(f"MoT checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        if "vggt_bridge" in payload:
            try:
                self.mot.vggt_bridge.load_state_dict(payload["vggt_bridge"], strict=True)
            except RuntimeError as exc:
                raise RuntimeError(f"vggt_bridge checkpoint mismatch: {exc}") from exc
        else:
            logger.warning("Legacy checkpoint has no vggt_bridge weights; keeping initialized bridge parameters.")
        if "vggt_aggregator" in payload:
            try:
                self.vggt_backbone.aggregator.load_state_dict(
                    payload["vggt_aggregator"],
                    strict=True,
                )
            except RuntimeError as exc:
                raise RuntimeError(f"vggt_aggregator checkpoint mismatch: {exc}") from exc
        elif self.train_vggt:
            logger.warning(
                "Checkpoint has no `vggt_aggregator`; keeping the Aggregator "
                "weights loaded during model construction."
            )
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current parameters.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` but current model disables it; ignoring.")
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload


__all__ = ["FrozenVGGTBackbone", "VGGTFastWAM"]
