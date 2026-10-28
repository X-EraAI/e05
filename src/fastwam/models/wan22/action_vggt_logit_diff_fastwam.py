from __future__ import annotations

from pathlib import Path

import torch

from .action_vggt_mot_fastwam import ActionVGGTMoTFastWAM


class ActionVGGTLogitDiffFastWAM(ActionVGGTMoTFastWAM):
    """Action flow model using contrastive logits from two aligned VGGT streams."""

    CHECKPOINT_FORMAT = "action_vggt_logit_diff_fastwam"
    CHECKPOINT_VERSION = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        backbone = self.vggt_backbone
        if tuple(backbone.va1_camera_keys) != tuple(backbone.input_camera_keys):
            raise ValueError(
                "ActionVGGTLogitDiffFastWAM requires va1_camera_keys to equal "
                "vggt_camera_keys so the two memory streams are token aligned."
            )
        if tuple(backbone.va2_camera_keys) != tuple(backbone.input_camera_keys):
            raise ValueError(
                "ActionVGGTLogitDiffFastWAM requires va2_camera_keys to equal "
                "vggt_camera_keys so the two memory streams are token aligned."
            )
        if not bool(backbone.isolate_vggt_streams):
            raise ValueError(
                "ActionVGGTLogitDiffFastWAM requires isolate_vggt_streams=true."
            )
        bridge = self.mot.vggt_bridge
        if not hasattr(bridge, "contrastive_lambda"):
            raise TypeError("ActionVGGTLogitDiffFastWAM requires a logit-difference bridge.")

    def _capture_vggt(self, vggt_images: torch.Tensor, *, batch_size: int | None = None):
        """图像到两路 memory 的张量流（不复制 memory 数据）。

        [B,V,3,H,W] -> 共享归一化/patch/camera/register token 预处理
        -> 隔离的 va1/va2 Aggregator -> 每个 callback 的拼接 K/V
        [B,Hv,M1+M2,Dv] -> split view -> va1_k/va1_v/va2_k/va2_v
        [B,Hv,M,Dv]。global 层 M=V*S，register 层 M=V*P；
        P 为 camera/register 前缀长度。只要求同层两路相等，不要求跨层 M 相等。
        """
        memory = super()._capture_vggt(vggt_images, batch_size=batch_size)
        expected_layers = set(self.mot.vggt_bridge.layer_mapping._by_fastwam.values())
        if set(memory) != expected_layers:
            raise ValueError(
                f"VGGT callback layers must match layer_pairs: expected={expected_layers}, got={set(memory)}."
            )
        result = {}
        for layer_idx, entry in memory.items():
            source_lens = entry.get("source_seq_lens")
            if not isinstance(source_lens, dict) or set(source_lens) != {"va1", "va2"}:
                raise ValueError(
                    f"VGGT layer {layer_idx} must expose source_seq_lens for va1/va2."
                )
            len1 = int(source_lens["va1"])
            len2 = int(source_lens["va2"])
            if len1 <= 0 or len1 != len2:
                raise ValueError(
                    "ActionVGGTLogitDiffFastWAM requires equal va1/va2 token counts, "
                    f"got layer {layer_idx}: va1={len1}, va2={len2}."
                )
            for name in ("k", "v"):
                if entry[name].ndim != 4 or entry[name].shape[2] != len1 + len2:
                    raise ValueError(
                        f"VGGT layer {layer_idx} {name} must be [B,Hv,{len1 + len2},Dv]."
                    )
            k1, k2 = torch.split(entry["k"], (len1, len2), dim=2)
            v1, v2 = torch.split(entry["v"], (len1, len2), dim=2)
            enriched = dict(entry)
            enriched.update({"va1_k": k1, "va1_v": v1, "va2_k": k2, "va2_v": v2})
            result[int(layer_idx)] = enriched
        return result

    def _architecture_metadata(self) -> dict[str, object]:
        bridge = self.mot.vggt_bridge
        metadata = super()._architecture_metadata()
        metadata.update(
            {
                "contrastive_lambda": float(bridge.contrastive_lambda),
                "contrastive_temperature": float(bridge.contrastive_temperature),
            }
        )
        return metadata

    @staticmethod
    def _normalize_metadata(metadata: dict[str, object]) -> dict[str, object]:
        normalized = dict(metadata)
        normalized["layer_pairs"] = tuple(
            tuple(int(value) for value in pair) for pair in normalized["layer_pairs"]
        )
        normalized["bridge_target_experts"] = tuple(normalized["bridge_target_experts"])
        for key in ("vggt_camera_keys", "va1_camera_keys", "va2_camera_keys"):
            normalized[key] = tuple(normalized[key])
        normalized["contrastive_lambda"] = float(normalized["contrastive_lambda"])
        normalized["contrastive_temperature"] = float(normalized["contrastive_temperature"])
        return normalized

    def save_checkpoint(self, path, optimizer=None, step=None):
        if optimizer is not None:
            raise ValueError(
                "Weight checkpoints do not contain optimizer state; "
                "use an Accelerate/DeepSpeed training-state checkpoint."
            )
        payload = {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            **self._architecture_metadata(),
            "action_expert": self.action_expert.state_dict(),
            "vggt_bridge": self.mot.vggt_bridge.state_dict(),
            "proprio_encoder": (
                None if self.proprio_encoder is None else self.proprio_encoder.state_dict()
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
                "use an Accelerate/DeepSpeed training-state checkpoint."
            )
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid ActionVGGTLogitDiffFastWAM checkpoint: {path}.")
        metadata_keys = set(self._architecture_metadata())
        required = {
            "format", "version", *metadata_keys, "action_expert", "vggt_bridge",
            "proprio_encoder", "va1", "va2", "step", "torch_dtype",
        }
        missing = sorted(required - set(payload))
        unexpected = sorted(set(payload) - required)
        if missing or unexpected:
            raise ValueError(
                "ActionVGGTLogitDiffFastWAM checkpoint key mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if payload["format"] != self.CHECKPOINT_FORMAT or int(payload["version"]) != self.CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported ActionVGGTLogitDiffFastWAM checkpoint format/version: "
                f"{payload['format']!r}/{payload['version']!r}."
            )
        expected = self._architecture_metadata()
        actual = self._normalize_metadata({key: payload[key] for key in metadata_keys})
        expected = self._normalize_metadata(expected)
        if actual != expected:
            raise ValueError(
                "Checkpoint architecture metadata does not match the constructed model: "
                f"checkpoint={actual}, model={expected}."
            )

        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.mot.vggt_bridge.load_state_dict(payload["vggt_bridge"], strict=True)
        self.vggt_backbone.va1.load_state_dict(payload["va1"], strict=True)
        self.vggt_backbone.va2.load_state_dict(payload["va2"], strict=True)
        proprio_state = payload["proprio_encoder"]
        if self.proprio_encoder is None:
            if proprio_state is not None:
                raise ValueError("Checkpoint contains proprio weights but this model disables proprio.")
        else:
            if not isinstance(proprio_state, dict):
                raise ValueError("Checkpoint is missing required proprio weights.")
            self.proprio_encoder.load_state_dict(proprio_state, strict=True)
        return payload


__all__ = ["ActionVGGTLogitDiffFastWAM"]
