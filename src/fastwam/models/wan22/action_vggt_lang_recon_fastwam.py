"""Legacy import compatibility for checkpoints and older downstream code.

The public model is e05.E05. The original serialized checkpoint format is retained.
"""
from .e05 import E05

ActionVGGTLangReconFastWAM = E05
__all__ = ["ActionVGGTLangReconFastWAM"]
