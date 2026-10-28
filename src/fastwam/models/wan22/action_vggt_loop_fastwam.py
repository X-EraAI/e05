"""Reconstruction-first recurrent VGGT variant for action prediction."""

from .action_vggt_recon_fastwam import ActionVGGTReconFastWAM
from .vggt_mot import LoopVGGTBackbone


class ActionVGGTLoopFastWAM(ActionVGGTReconFastWAM):
    """Use pass-1 VGGT features for reconstruction and pass-N K/V for action."""

    CHECKPOINT_FORMAT = "action_vggt_loop_fastwam"
    CHECKPOINT_VERSION = 2
    evaluation_mode = "action_only"

    def __init__(self, *, num_vggt_forwards: int = 4, **kwargs) -> None:
        backbone = kwargs.get("vggt_backbone")
        if not isinstance(backbone, LoopVGGTBackbone):
            raise TypeError(
                "ActionVGGTLoopFastWAM requires `vggt_backbone` to be a "
                "LoopVGGTBackbone."
            )
        if backbone.num_vggt_forwards != num_vggt_forwards:
            raise ValueError(
                "Model/backbone `num_vggt_forwards` mismatch: "
                f"{num_vggt_forwards} vs {backbone.num_vggt_forwards}."
            )
        super().__init__(**kwargs)
        self.num_vggt_forwards = num_vggt_forwards


__all__ = ["ActionVGGTLoopFastWAM"]
