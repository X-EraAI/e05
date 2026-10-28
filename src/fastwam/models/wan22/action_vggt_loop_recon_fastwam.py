"""Language-conditioned VGGT recurrence with complete supervision per pass."""

from .action_vggt_lang_recon_fastwam import ActionVGGTLangReconFastWAM
from .vggt_lang_recon_backbone import VGGTLoopLanguageReconBackbone


class ActionVGGTLoopReconFastWAM(ActionVGGTLangReconFastWAM):
    CHECKPOINT_FORMAT = "action_vggt_loop_recon_fastwam"
    report_loop_metrics = True

    def __init__(self, *args, num_vggt_forwards=2, **kwargs):
        if (isinstance(num_vggt_forwards, bool)
                or not isinstance(num_vggt_forwards, int) or num_vggt_forwards < 1):
            raise ValueError("num_vggt_forwards must be a positive integer.")
        super().__init__(*args, **kwargs)
        if not isinstance(self.vggt_backbone, VGGTLoopLanguageReconBackbone):
            raise TypeError("Loop reconstruction requires VGGTLoopLanguageReconBackbone.")
        if self.vggt_backbone.num_vggt_forwards != num_vggt_forwards:
            raise ValueError("Model and backbone num_vggt_forwards must match.")
        self.num_vggt_forwards = num_vggt_forwards

    def _training_reconstructions(self, images, **kwargs):
        return self.vggt_backbone.forward_all_reconstructions(images, **kwargs)

    def _prepare_context(self, *args, **kwargs):
        context, mask = super()._prepare_context(*args, **kwargs)
        # Attention masks do not prevent NaN padding from contaminating value
        # matmuls. Sanitize before ActionDiT projects its context as VGGT does.
        return context.masked_fill(~mask.unsqueeze(-1), 0), mask

    def _checkpoint_metadata(self):
        return {
            **super()._checkpoint_metadata(),
            "num_vggt_forwards": self.num_vggt_forwards,
            "loop_state": "final_visual_and_language_no_detach",
            "loop_supervision": f"mean_full_action_and_{self.reconstruction_target}_loss",
            "action_hidden_recurrence": False,
        }

    def _checkpoint_transfer_formats(self):
        return {self.CHECKPOINT_FORMAT, ActionVGGTLangReconFastWAM.CHECKPOINT_FORMAT}

    def _checkpoint_transfer_fields(self):
        # Explicit transfer copies weights only; strict resume still compares
        # the entire recurrence and supervision contract.
        return super()._checkpoint_transfer_fields() | {
            "num_vggt_forwards", "loop_state", "loop_supervision",
            "action_hidden_recurrence",
        }


__all__ = ["ActionVGGTLoopReconFastWAM"]
