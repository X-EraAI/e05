"""Named reconstruction variant of :class:`ActionVGGTFastWAM`."""
from .action_vggt_fastwam import ActionVGGTFastWAM


class ActionVGGTReconFastWAM(ActionVGGTFastWAM):
    """Action VGGT model with official-style depth/camera/point supervision.

    The implementation lives in the base class so action-only behavior remains
    unchanged; this class gives Hydra/checkpoints an explicit model identity.
    """

    CHECKPOINT_FORMAT = "action_vggt_recon_fastwam"
    CHECKPOINT_VERSION = 1
    evaluation_mode = "action_only"
