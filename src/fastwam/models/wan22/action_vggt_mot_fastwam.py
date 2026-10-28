from __future__ import annotations

from pathlib import Path

import torch

from .action_vggt_fastwam import ActionVGGTFastWAM


class ActionVGGTMoTFastWAM(ActionVGGTFastWAM):
    """Action flow model conditioned by coupled or isolated va1/va2 memory."""

    CHECKPOINT_FORMAT = "action_vggt_mot_fastwam"
    CHECKPOINT_VERSION = 3
    ROUTED_CHECKPOINT_VERSION = 2
    LEGACY_CHECKPOINT_VERSION = 1

    def _architecture_metadata(self) -> dict[str, object]:
        bridge = self.mot.vggt_bridge
        layer_pairs = tuple(
            (int(vggt_layer), int(action_layer))
            for action_layer, vggt_layer in sorted(
                bridge.layer_mapping._by_fastwam.items()
            )
        )
        return {
            "action_num_layers": int(len(self.action_expert.blocks)),
            "layer_pairs": layer_pairs,
            "bridge_projection_mode": str(bridge.projection_mode),
            "bridge_projection_sharing": str(bridge.projection_sharing),
            "bridge_target_experts": tuple(sorted(bridge.target_experts)),
            "train_vggt": bool(self.train_vggt),
            "vggt_camera_keys": tuple(self.vggt_backbone.input_camera_keys),
            "va1_camera_keys": tuple(self.vggt_backbone.va1_camera_keys),
            "va2_camera_keys": tuple(self.vggt_backbone.va2_camera_keys),
            "isolate_vggt_streams": bool(
                self.vggt_backbone.isolate_vggt_streams
            ),
        }

    def _uses_legacy_full_view_routing(self) -> bool:
        backbone = self.vggt_backbone
        return (
            tuple(backbone.va1_camera_keys) == tuple(backbone.input_camera_keys)
            and tuple(backbone.va2_camera_keys) == tuple(backbone.input_camera_keys)
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        if optimizer is not None:
            raise ValueError(
                "Weight checkpoints do not contain optimizer state; "
                "use an Accelerate training-state checkpoint."
            )
        payload = {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            **self._architecture_metadata(),
            "action_expert": self.action_expert.state_dict(),
            "vggt_bridge": self.mot.vggt_bridge.state_dict(),
            "proprio_encoder": (
                None
                if self.proprio_encoder is None
                else self.proprio_encoder.state_dict()
            ),
            "va1": self.vggt_backbone.va1.state_dict(),
            "va2": self.vggt_backbone.va2.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        if optimizer is not None:
            raise ValueError(
                "Weight checkpoints do not contain optimizer state; "
                "use an Accelerate training-state checkpoint."
            )
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid ActionVGGTMoTFastWAM checkpoint: {path}.")
        common_required = {
            "format",
            "version",
            "action_num_layers",
            "layer_pairs",
            "bridge_projection_mode",
            "bridge_projection_sharing",
            "bridge_target_experts",
            "train_vggt",
            "action_expert",
            "vggt_bridge",
            "proprio_encoder",
            "va1",
            "va2",
            "step",
            "torch_dtype",
        }
        version = int(payload.get("version", -1))
        required = set(common_required)
        if version in {self.ROUTED_CHECKPOINT_VERSION, self.CHECKPOINT_VERSION}:
            required.update(
                {"vggt_camera_keys", "va1_camera_keys", "va2_camera_keys"}
            )
        if version == self.CHECKPOINT_VERSION:
            required.add("isolate_vggt_streams")
        missing = sorted(required - set(payload))
        unexpected = sorted(set(payload) - required)
        if missing or unexpected:
            raise ValueError(
                "ActionVGGTMoTFastWAM checkpoint key mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if (
            payload["format"] != self.CHECKPOINT_FORMAT
            or version
            not in {
                self.LEGACY_CHECKPOINT_VERSION,
                self.ROUTED_CHECKPOINT_VERSION,
                self.CHECKPOINT_VERSION,
            }
        ):
            raise ValueError(
                "Unsupported ActionVGGTMoTFastWAM checkpoint format/version: "
                f"{payload.get('format')!r}/{payload.get('version')!r}."
            )

        expected_metadata = self._architecture_metadata()
        if version == self.LEGACY_CHECKPOINT_VERSION:
            if not self._uses_legacy_full_view_routing():
                raise ValueError(
                    "ActionVGGTMoTFastWAM v1 checkpoints used legacy full-view "
                    "routing. Configure both va1_camera_keys and va2_camera_keys "
                    "to equal vggt_camera_keys before loading this checkpoint."
                )
            expected_metadata = {
                key: value
                for key, value in expected_metadata.items()
                if key
                not in {
                    "vggt_camera_keys",
                    "va1_camera_keys",
                    "va2_camera_keys",
                    "isolate_vggt_streams",
                }
            }
        elif version == self.ROUTED_CHECKPOINT_VERSION:
            # v2 predates configurable stream isolation.  Its tensors are
            # structurally compatible with either mode, so the constructed
            # model's current configuration intentionally controls visibility.
            expected_metadata = {
                key: value
                for key, value in expected_metadata.items()
                if key != "isolate_vggt_streams"
            }
        actual_metadata = {
            key: payload[key]
            for key in expected_metadata
        }
        # torch.load may restore nested tuples as lists in hand-authored files;
        # normalize the one structured field before strict comparison.
        actual_metadata["layer_pairs"] = tuple(
            tuple(int(value) for value in pair)
            for pair in actual_metadata["layer_pairs"]
        )
        actual_metadata["bridge_target_experts"] = tuple(
            actual_metadata["bridge_target_experts"]
        )
        for key in ("vggt_camera_keys", "va1_camera_keys", "va2_camera_keys"):
            if key in actual_metadata:
                actual_metadata[key] = tuple(actual_metadata[key])
        if actual_metadata != expected_metadata:
            raise ValueError(
                "Checkpoint architecture metadata does not match the constructed model: "
                f"checkpoint={actual_metadata}, model={expected_metadata}."
            )

        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.mot.vggt_bridge.load_state_dict(payload["vggt_bridge"], strict=True)
        self.vggt_backbone.va1.load_state_dict(payload["va1"], strict=True)
        self.vggt_backbone.va2.load_state_dict(payload["va2"], strict=True)
        proprio_state = payload["proprio_encoder"]
        if self.proprio_encoder is None:
            if proprio_state is not None:
                raise ValueError(
                    "Checkpoint contains proprio weights but this model disables proprio."
                )
        else:
            if not isinstance(proprio_state, dict):
                raise ValueError(
                    "Checkpoint is missing proprio weights required by this model."
                )
            self.proprio_encoder.load_state_dict(proprio_state, strict=True)
        return payload


__all__ = ["ActionVGGTMoTFastWAM"]
