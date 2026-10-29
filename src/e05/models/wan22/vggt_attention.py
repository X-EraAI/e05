"""Shared token and attention operations for the E05 language backbone."""
from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn

def _expand_input_tokens(
    aggregator: nn.Module,
    patch_tokens: torch.Tensor,
    *,
    batch_size: int,
    num_frames: int,
) -> torch.Tensor:
    """Build shared ``[B,V,S,D]`` camera/register/patch input tokens."""
    camera = aggregator.camera_token[:, 0:1].expand(
        batch_size, 1, *aggregator.camera_token.shape[2:]
    )
    if num_frames > 1:
        camera = torch.cat(
            [
                camera,
                aggregator.camera_token[:, 1:].expand(
                    batch_size, num_frames - 1, *aggregator.camera_token.shape[2:]
                ),
            ],
            dim=1,
        )
    camera = camera.reshape(batch_size * num_frames, 1, -1)

    register = aggregator.register_token[:, 0:1].expand(
        batch_size, 1, *aggregator.register_token.shape[2:]
    )
    if num_frames > 1:
        register = torch.cat(
            [
                register,
                aggregator.register_token[:, 1:].expand(
                    batch_size, num_frames - 1, *aggregator.register_token.shape[2:]
                ),
            ],
            dim=1,
        )
    register = register.reshape(
        batch_size * num_frames, register.shape[2], register.shape[3]
    )
    tokens = torch.cat([camera, register, patch_tokens], dim=1)
    return tokens.reshape(batch_size, num_frames, tokens.shape[1], tokens.shape[2])


def _attention_qkv(
    block: nn.Module,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one VGGT expert to head-major ``[B,H,S,Dh]`` Q/K/V."""
    attention = block.attn
    qkv = attention.qkv(block.norm1(tokens))
    batch_size, seq_len, _ = qkv.shape
    width = int(attention.qkv.in_features)
    qkv = qkv.reshape(
        batch_size,
        seq_len,
        3,
        int(attention.num_heads),
        width // int(attention.num_heads),
    )
    q, k, v = torch.unbind(qkv, dim=2)
    q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
    if bool(attention.use_qk_norm):
        q = attention.q_norm(q)
        k = attention.k_norm(k)
    return q, k, v


def _post_inter_frame_attention(
    block: nn.Module,
    residual_tokens: torch.Tensor,
    attention_output: torch.Tensor,
) -> torch.Tensor:
    """Apply the owning expert's output projection, residual, and FFN."""
    attention = block.attn
    projected = attention.proj(attention_output)
    projected = attention.proj_drop(projected)
    x = residual_tokens + block.ls1(projected)
    return x + block.ls2(block.mlp(block.norm2(x)))


def _active_inter_frame_tokens(
    tokens: torch.Tensor,
    *,
    attention_type: str,
    patch_token_start: int,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Flatten the token subset used by a VGGT inter-frame block.

    Global layers mix all ``V*S`` tokens.  Register layers mix only the
    ``V*P`` camera/register prefix and pass patch tokens through unchanged.
    """
    batch_size, num_frames, num_tokens, width = tokens.shape
    if attention_type == "global":
        active = tokens.reshape(batch_size, num_frames * num_tokens, width)

        def restore(updated: torch.Tensor) -> torch.Tensor:
            return updated.reshape(batch_size, num_frames, num_tokens, width)

        return active, restore
    if attention_type != "register":
        raise ValueError(f"Unknown inter-frame attention type: {attention_type!r}.")

    prefix = tokens[:, :, :patch_token_start]
    patches = tokens[:, :, patch_token_start:]
    active = prefix.reshape(batch_size, num_frames * patch_token_start, width)

    def restore(updated: torch.Tensor) -> torch.Tensor:
        updated_prefix = updated.reshape(
            batch_size, num_frames, patch_token_start, width
        )
        return torch.cat([updated_prefix, patches], dim=2)

    return active, restore
