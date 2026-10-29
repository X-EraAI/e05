"""Read the original release's serialized identifiers without exposing old APIs.

These constants describe existing checkpoint/config bytes. New checkpoints and
public Python imports use E05 names; no alternate model is supported here.
"""
from collections.abc import Mapping

LEGACY_FORMAT = "action_vggt_lang_recon_fastwam"
LEGACY_PACKAGE = "fastwam"


def normalize_metadata(value):
    """Translate the original E05 contract while preserving all other values."""
    if isinstance(value, Mapping):
        return {
            key.replace(LEGACY_PACKAGE, "e05"): normalize_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(normalize_metadata(item) for item in value)
    if isinstance(value, list):
        return [normalize_metadata(item) for item in value]
    if isinstance(value, str):
        if value == LEGACY_FORMAT:
            return "e05"
        if value in {"vggt_to_fastwam", "fastwam_to_vggt"}:
            return value.replace(LEGACY_PACKAGE, "e05")
    return value


def normalize_checkpoint(payload):
    """Adapt only the original language reconstruction model's checkpoint."""
    if not isinstance(payload, Mapping) or payload.get("format") != LEGACY_FORMAT:
        return payload
    result = dict(payload)
    result["format"] = "e05"
    if isinstance(result.get("metadata"), Mapping):
        result["metadata"] = normalize_metadata(result["metadata"])
    if isinstance(result.get("vggt_bridge"), Mapping):
        result["vggt_bridge"] = normalize_metadata(result["vggt_bridge"])
    return result


def normalize_config(config):
    """Translate saved release configurations at the file-loading boundary."""
    from omegaconf import OmegaConf

    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, str):
            if value in {"fastwam.runtime.create_e05",
                         "fastwam.runtime.create_action_vggt_lang_recon_fastwam"}:
                return "e05.create_e05"
            if value.startswith(LEGACY_PACKAGE + "."):
                return value.replace(LEGACY_PACKAGE, "e05").replace("FastWAMProcessor", "E05Processor")
            return normalize_metadata(value)
        return value

    return OmegaConf.create(convert(OmegaConf.to_container(config, resolve=False)))
