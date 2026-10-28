from __future__ import annotations

from pathlib import Path

import torch

from .action_vggt_depth_fastwam import ActionVGGTDepthFastWAM


class ActionVGGTModDepthFastWAM(ActionVGGTDepthFastWAM):
    """Serial-va1/va2 VGGT backbone with joint 48-layer action/depth experts.

    The depth model's flow/loss/inference implementation is intentionally reused.
    The runtime constructs both experts with 48 blocks and supplies the 48 K/V
    memories emitted by :class:`SerialVGGTBackbone` (va1 layers 0..23 followed
    by va2 layers 24..47).
    """

    CHECKPOINT_FORMAT = "action_vggt_mod_depth_fastwam"
    CHECKPOINT_VERSION = 1

    def save_checkpoint(self, path, optimizer=None, step=None):
        if optimizer is not None:
            raise ValueError(
                "Weight checkpoints do not contain optimizer state; use Accelerate state checkpoints."
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
            "va2_input_mode": self.vggt_backbone.va2_input_mode,
            "action_expert": self.action_expert.state_dict(),
            "depth_expert": self.depth_expert.state_dict(),
            "vggt_bridge": self.mot.vggt_bridge.state_dict(),
            "va1": self.vggt_backbone.va1.state_dict(),
            "va2": self.vggt_backbone.va2.state_dict(),
            "proprio_encoder": None if self.proprio_encoder is None else self.proprio_encoder.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            # Bridge topology is part of the checkpoint contract, not merely
            # an implementation detail of the current runtime config.
            "bridge_layer_pairs": tuple(
                (int(vggt), int(fastwam))
                for fastwam, vggt in sorted(self.mot.vggt_bridge.layer_mapping._by_fastwam.items())
            ),
            "bridge_projection_mode": self.mot.vggt_bridge.projection_mode,
            "bridge_projection_sharing": self.mot.vggt_bridge.projection_sharing,
            "bridge_target_experts": tuple(sorted(self.mot.vggt_bridge.target_experts)),
            "action_num_layers": len(self.action_expert.blocks),
            "depth_num_layers": len(self.depth_expert.blocks),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        if optimizer is not None:
            raise ValueError(
                "Weight checkpoints do not contain optimizer state; use Accelerate state checkpoints."
            )
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        expected = {
            "format", "version", "depth_normalization_version", "depth_future_offsets",
            "depth_patch_size", "patch_grid_height", "patch_grid_width", "expected_num_views",
            "depth_reads_action", "train_vggt", "va2_input_mode", "action_expert", "depth_expert",
            "vggt_bridge", "va1", "va2", "proprio_encoder", "step", "torch_dtype",
            "bridge_layer_pairs", "bridge_projection_mode", "bridge_projection_sharing",
            "bridge_target_experts", "action_num_layers", "depth_num_layers",
        }
        actual = set(payload) if isinstance(payload, dict) else set()
        if actual != expected:
            raise ValueError(
                "ActionVGGTModDepthFastWAM checkpoint key mismatch: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
            )
        if payload["format"] != self.CHECKPOINT_FORMAT or int(payload["version"]) != self.CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported ActionVGGTModDepthFastWAM checkpoint format/version: "
                f"{payload.get('format')!r}/{payload.get('version')!r}."
            )
        metadata = {
            "depth_normalization_version": self.DEPTH_NORMALIZATION_VERSION,
            "depth_future_offsets": self.depth_future_offsets,
            "depth_patch_size": self.depth_expert.depth_patch_size,
            "patch_grid_height": self.depth_expert.patch_grid_height,
            "patch_grid_width": self.depth_expert.patch_grid_width,
            "expected_num_views": self.expected_num_views,
            "depth_reads_action": self.depth_reads_action,
            "train_vggt": self.train_vggt,
            "va2_input_mode": self.vggt_backbone.va2_input_mode,
            "bridge_layer_pairs": tuple(
                (int(vggt), int(fastwam))
                for fastwam, vggt in sorted(self.mot.vggt_bridge.layer_mapping._by_fastwam.items())
            ),
            "bridge_projection_mode": self.mot.vggt_bridge.projection_mode,
            "bridge_projection_sharing": self.mot.vggt_bridge.projection_sharing,
            "bridge_target_experts": tuple(sorted(self.mot.vggt_bridge.target_experts)),
            "action_num_layers": len(self.action_expert.blocks),
            "depth_num_layers": len(self.depth_expert.blocks),
        }
        for key, expected_value in metadata.items():
            value = payload[key]
            if key == "depth_future_offsets":
                value = tuple(int(item) for item in value)
            if value != expected_value:
                raise ValueError(f"Checkpoint `{key}` mismatch: expected {expected_value}, got {value}.")
        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.depth_expert.load_state_dict(payload["depth_expert"], strict=True)
        self.mot.vggt_bridge.load_state_dict(payload["vggt_bridge"], strict=True)
        self.vggt_backbone.va1.load_state_dict(payload["va1"], strict=True)
        self.vggt_backbone.va2.load_state_dict(payload["va2"], strict=True)
        proprio_state = payload["proprio_encoder"]
        if self.proprio_encoder is None:
            if proprio_state is not None:
                raise ValueError("Checkpoint contains unexpected proprio weights.")
        else:
            if not isinstance(proprio_state, dict):
                raise ValueError("Checkpoint is missing required proprio weights.")
            self.proprio_encoder.load_state_dict(proprio_state, strict=True)
        return payload


__all__ = ["ActionVGGTModDepthFastWAM"]
