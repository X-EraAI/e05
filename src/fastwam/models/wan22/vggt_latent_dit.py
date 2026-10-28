from __future__ import annotations

from typing import Any

import torch

from .action_dit import ActionDiT
from .wan_video_dit import sinusoidal_embedding_1d


class VGGTLatentDiT(ActionDiT):
    """DiT expert that predicts adapter-compressed VGGT latents.

    Latents use channel-first ``[B,C,F,H,W]`` storage.  The inherited DiT
    operates on ``[B,F*H*W,C]`` tokens; no current ``z0`` is concatenated.
    """

    def __init__(self, *, latent_frames: int, latent_height: int,
                 latent_width: int, **kwargs: Any) -> None:
        if int(kwargs.get("action_dim", -1)) <= 0:
            raise ValueError("`action_dim` must be positive for VGGT latent DiT.")
        self.latent_frames = int(latent_frames)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        if min(self.latent_frames, self.latent_height, self.latent_width) <= 0:
            raise ValueError("VGGT latent layout dimensions must be positive.")
        sequence_length = self.latent_frames * self.latent_height * self.latent_width
        if int(kwargs.get("max_seq_len", 1024)) < sequence_length:
            raise ValueError(
                f"VGGT latent token length {sequence_length} exceeds max_seq_len."
            )
        super().__init__(**kwargs)
        self.latent_sequence_length = sequence_length
        self.frame_embedding = torch.nn.Embedding(self.latent_frames, self.hidden_dim)
        self.row_embedding = torch.nn.Embedding(self.latent_height, self.hidden_dim)
        self.column_embedding = torch.nn.Embedding(self.latent_width, self.hidden_dim)
        scale = self.hidden_dim ** -0.5
        for emb in (self.frame_embedding, self.row_embedding, self.column_embedding):
            torch.nn.init.normal_(emb.weight, std=scale)

    def _position_embedding(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        frame = self.frame_embedding.weight[:, None, None, :]
        row = self.row_embedding.weight[None, :, None, :]
        col = self.column_embedding.weight[None, None, :, :]
        return (frame + row + col).reshape(self.latent_sequence_length, self.hidden_dim).to(device=device, dtype=dtype)

    def flatten_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """``[B,48,F,24,20] -> [B,F*24*20,48]`` for DiT."""
        expected = (self.action_dim, self.latent_frames, self.latent_height, self.latent_width)
        if latent.ndim != 5 or tuple(latent.shape[1:]) != expected:
            raise ValueError(f"Expected latent [B,{','.join(map(str, expected))}], got {tuple(latent.shape)}.")
        return latent.permute(0, 2, 3, 4, 1).reshape(latent.shape[0], self.latent_sequence_length, self.action_dim).contiguous()

    def unflatten_latent(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[B,F*24*20,48] -> [B,48,F,24,20]`` after DiT."""
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (self.latent_sequence_length, self.action_dim):
            raise ValueError(f"Expected tokens [B,{self.latent_sequence_length},{self.action_dim}], got {tuple(tokens.shape)}.")
        return tokens.view(tokens.shape[0], self.latent_frames, self.latent_height, self.latent_width, self.action_dim).permute(0, 4, 1, 2, 3).contiguous()

    def pre_dit(self, action_tokens, timestep, context, context_mask=None):
        state = super().pre_dit(action_tokens=action_tokens, timestep=timestep, context=context, context_mask=context_mask)
        state["tokens"] = state["tokens"] + self._position_embedding(device=state["tokens"].device, dtype=state["tokens"].dtype).unsqueeze(0)
        state["meta"].update({"latent_frames": self.latent_frames, "latent_height": self.latent_height, "latent_width": self.latent_width})
        return state


__all__ = ["VGGTLatentDiT"]
