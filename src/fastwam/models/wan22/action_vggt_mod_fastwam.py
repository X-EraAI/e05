from __future__ import annotations

from pathlib import Path

import torch

from .action_vggt_fastwam import ActionVGGTFastWAM


class ActionVGGTModFastWAM(ActionVGGTFastWAM):
    """Action flow model conditioned by serial va1/va2 VGGT memories."""

    CHECKPOINT_FORMAT = "action_vggt_mod_fastwam"
    CHECKPOINT_VERSION = 1
    LEGACY_CHECKPOINT_VERSION = 1

    def save_checkpoint(self, path, optimizer=None, step=None):
        if optimizer is not None:
            raise ValueError("Weight checkpoints do not contain optimizer state; use Accelerate state checkpoints.")
        payload = {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            "action_expert": self.action_expert.state_dict(),
            "vggt_bridge": self.mot.vggt_bridge.state_dict(),
            "proprio_encoder": None if self.proprio_encoder is None else self.proprio_encoder.state_dict(),
            "va1": self.vggt_backbone.va1.state_dict(),
            "va2": self.vggt_backbone.va2.state_dict(),
            "va2_input_mode": self.vggt_backbone.va2_input_mode,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        if optimizer is not None:
            raise ValueError("Weight checkpoints do not contain optimizer state; use Accelerate state checkpoints.")
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid ActionVGGTModFastWAM checkpoint: {path}.")
        required = {"format", "version", "action_expert", "vggt_bridge", "proprio_encoder", "va1", "va2", "va2_input_mode", "step", "torch_dtype"}
        missing = sorted(required - set(payload))
        unexpected = sorted(set(payload) - required)
        if missing or unexpected:
            raise ValueError(f"ActionVGGTModFastWAM checkpoint key mismatch: missing={missing}, unexpected={unexpected}.")
        if payload["format"] != self.CHECKPOINT_FORMAT or int(payload["version"]) != self.CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported ActionVGGTModFastWAM checkpoint format/version: {payload.get('format')!r}/{payload.get('version')!r}.")
        if payload["va2_input_mode"] != self.vggt_backbone.va2_input_mode:
            raise ValueError("Checkpoint va2_input_mode does not match the constructed model.")
        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.mot.vggt_bridge.load_state_dict(payload["vggt_bridge"], strict=True)
        self.vggt_backbone.va1.load_state_dict(payload["va1"], strict=True)
        self.vggt_backbone.va2.load_state_dict(payload["va2"], strict=True)
        if self.proprio_encoder is None:
            if payload["proprio_encoder"] is not None:
                raise ValueError("Checkpoint contains proprio weights but this model disables proprio.")
        else:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        return payload


__all__ = ["ActionVGGTModFastWAM"]
