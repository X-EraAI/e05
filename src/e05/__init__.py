"""Public Python interface for the e05 model."""
from fastwam.models.wan22.e05 import E05


def create_e05(**kwargs):
    """Construct e05 for training or loading a complete release checkpoint."""
    from fastwam.runtime import create_e05 as build
    return build(**kwargs)


__all__ = ["E05", "create_e05"]
__version__ = "0.1.0"
