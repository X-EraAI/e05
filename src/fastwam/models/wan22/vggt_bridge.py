from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LayerProjectionSet(nn.Module):
    """Projection parameters owned by one independently bridged layer."""

    def __init__(self, projection_mode: str, fastwam_width: int, vggt_width: int) -> None:
        super().__init__()
        if projection_mode == "vggt_to_fastwam":
            self.vggt_k_proj = nn.Linear(vggt_width, fastwam_width, bias=False)
            self.vggt_v_proj = nn.Linear(vggt_width, fastwam_width, bias=False)
        else:
            self.fastwam_q_proj = nn.Linear(fastwam_width, vggt_width, bias=False)
            self.fastwam_k_proj = nn.Linear(fastwam_width, vggt_width, bias=False)
            self.fastwam_v_proj = nn.Linear(fastwam_width, vggt_width, bias=False)
            self.output_proj = nn.Linear(vggt_width, fastwam_width, bias=False)


class LayerMapping:
    """Validated one-to-one VGGT-to-FastWAM layer lookup."""

    def __init__(
        self,
        layer_pairs: Sequence[tuple[int, int]],
        *,
        vggt_num_layers: int = 24,
        fastwam_num_layers: int = 30,
    ) -> None:
        if not layer_pairs:
            raise ValueError("`layer_pairs` must be non-empty.")
        by_fastwam: dict[int, int] = {}
        seen_vggt: set[int] = set()
        for pair in layer_pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each layer pair must contain exactly (vggt_layer, fastwam_layer).")
            vggt_layer, fastwam_layer = pair
            if isinstance(vggt_layer, bool) or not isinstance(vggt_layer, int):
                raise TypeError("VGGT layer indices must be integers.")
            if isinstance(fastwam_layer, bool) or not isinstance(fastwam_layer, int):
                raise TypeError("FastWAM layer indices must be integers.")
            if not 0 <= vggt_layer < vggt_num_layers:
                raise ValueError(f"VGGT layer {vggt_layer} is out of range [0, {vggt_num_layers}).")
            if not 0 <= fastwam_layer < fastwam_num_layers:
                raise ValueError(f"FastWAM layer {fastwam_layer} is out of range [0, {fastwam_num_layers}).")
            if vggt_layer in seen_vggt:
                raise ValueError(f"duplicate VGGT layer {vggt_layer} in `layer_pairs`.")
            if fastwam_layer in by_fastwam:
                raise ValueError(f"duplicate FastWAM layer {fastwam_layer} in `layer_pairs`.")
            seen_vggt.add(vggt_layer)
            by_fastwam[fastwam_layer] = vggt_layer
        self._by_fastwam = by_fastwam

    def vggt_layer_for(self, fastwam_layer: int) -> int | None:
        """Return the mapped VGGT layer in O(1), or None when unmapped."""
        return self._by_fastwam.get(fastwam_layer)


class VGGTSharedAttentionBridge(nn.Module):
    """One-way VGGT K/V memory bridge for video/action MoT attention.

    Inputs use flat FastWAM Q/K/V [B,S,H*Dh]. VGGT callback K/V use
    [B,H,S,Dh]. The MoT mask is boolean [sum(S),sum(S)], where True means
    a query may read a key. VGGT contributes keys/values only, never queries.
    """

    VALID_EXPERTS = frozenset({"video", "action", "four_d", "depth", "vggt"})
    VALID_MODES = frozenset({"vggt_to_fastwam", "fastwam_to_vggt"})
    VALID_PROJECTION_SHARING = frozenset({"shared", "independent"})

    def __init__(
        self,
        *,
        fastwam_width: int = 3072,
        vggt_width: int = 1024,
        fastwam_num_heads: int = 24,
        vggt_num_heads: int = 16,
        vggt_num_layers: int = 24,
        fastwam_num_layers: int = 30,
        layer_pairs: Sequence[tuple[int, int]],
        target_experts: Sequence[str],
        projection_mode: str,
        projection_sharing: str = "shared",
        detach_vggt_memory: bool = True,
    ) -> None:
        super().__init__()
        if projection_mode not in self.VALID_MODES:
            raise ValueError(f"Invalid `projection_mode`: {projection_mode!r}.")
        if projection_sharing not in self.VALID_PROJECTION_SHARING:
            raise ValueError(
                f"Invalid `projection_sharing`: {projection_sharing!r}; "
                "expected 'shared' or 'independent'."
            )
        targets = tuple(target_experts)
        if not targets or len(set(targets)) != len(targets) or not set(targets) <= self.VALID_EXPERTS:
            raise ValueError(
                "`target_experts` must be a unique non-empty subset of "
                "{'video', 'action', 'four_d', 'depth', 'vggt'}."
            )
        if fastwam_width <= 0 or vggt_width <= 0:
            raise ValueError("Attention widths must be positive.")
        if fastwam_num_heads <= 0 or vggt_num_heads <= 0:
            raise ValueError("Attention heads must be positive.")
        if fastwam_width % fastwam_num_heads or vggt_width % vggt_num_heads:
            raise ValueError("Attention width must be divisible by its number of heads.")

        self.layer_mapping = LayerMapping(
            layer_pairs,
            vggt_num_layers=int(vggt_num_layers),
            fastwam_num_layers=int(fastwam_num_layers),
        )
        self.target_experts = frozenset(targets)
        self.projection_mode = projection_mode
        self.projection_sharing = projection_sharing
        self.detach_vggt_memory = bool(detach_vggt_memory)
        self.fastwam_width = fastwam_width
        self.vggt_width = vggt_width
        self.fastwam_num_heads = fastwam_num_heads
        self.vggt_num_heads = vggt_num_heads

        if projection_sharing == "independent":
            self.layer_projections = nn.ModuleDict({
                str(fastwam_layer): _LayerProjectionSet(projection_mode, fastwam_width, vggt_width)
                for _, fastwam_layer in layer_pairs
            })
        elif projection_mode == "vggt_to_fastwam":
            # Keep these historical top-level names unchanged so checkpoints
            # trained before projection sharing was configurable still load.
            self.vggt_k_proj = nn.Linear(vggt_width, fastwam_width, bias=False)
            self.vggt_v_proj = nn.Linear(vggt_width, fastwam_width, bias=False)
        else:
            self.fastwam_q_proj = nn.Linear(fastwam_width, vggt_width, bias=False)
            self.fastwam_k_proj = nn.Linear(fastwam_width, vggt_width, bias=False)
            self.fastwam_v_proj = nn.Linear(fastwam_width, vggt_width, bias=False)
            self.output_proj = nn.Linear(vggt_width, fastwam_width, bias=False)

    def vggt_layer_for(self, fastwam_layer: int) -> int | None:
        return self.layer_mapping.vggt_layer_for(fastwam_layer)

    def _projections_for(self, fastwam_layer: int) -> nn.Module:
        if self.projection_sharing == "shared":
            return self
        return self.layer_projections[str(fastwam_layer)]

    def _flatten_vggt_memory(
        self,
        vggt_k: torch.Tensor,
        vggt_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.detach_vggt_memory:
            vggt_k = vggt_k.detach()
            vggt_v = vggt_v.detach()
        return (
            vggt_k.transpose(1, 2).contiguous().flatten(2),
            vggt_v.transpose(1, 2).contiguous().flatten(2),
        )

    @staticmethod
    def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, heads: int) -> torch.Tensor:
        """SDPA over flat [B,S,H*Dh], with allow-mask [Sq,Sk]."""
        batch, q_len, width = q.shape
        head_dim = width // heads
        qh = q.view(batch, q_len, heads, head_dim).transpose(1, 2)
        kh = k.view(batch, k.shape[1], heads, head_dim).transpose(1, 2)
        vh = v.view(batch, v.shape[1], heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask[None, None])
        return out.transpose(1, 2).contiguous().view(batch, q_len, width)

    def _validate(
        self,
        fastwam_qkv: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        vggt_k: torch.Tensor,
        vggt_v: torch.Tensor,
        attention_mask: torch.Tensor,
        expert_order: Sequence[str],
        *,
        allow_partial_targets: bool = False,
    ) -> list[int]:
        if tuple(expert_order) != tuple(fastwam_qkv.keys()):
            raise ValueError("`expert_order` must exactly match FastWAM Q/K/V insertion order.")
        expert_set = set(expert_order)
        if (
            not expert_order
            or len(expert_set) != len(expert_order)
            or not expert_set <= self.VALID_EXPERTS
        ):
            raise ValueError(
                "FastWAM Q/K/V experts must be a unique non-empty subset of "
                "{'video', 'action', 'four_d', 'depth', 'vggt'}."
            )
        if not allow_partial_targets and not self.target_experts <= expert_set:
            raise ValueError(
                f"Bridge target experts {sorted(self.target_experts)} are not all present "
                f"in FastWAM Q/K/V experts {list(expert_order)}."
            )
        if vggt_k.ndim != 4 or vggt_v.ndim != 4:
            raise ValueError("VGGT K/V must have shape [B,H,S,Dh].")
        if vggt_k.shape != vggt_v.shape:
            raise ValueError("VGGT K/V must have the same shape.")
        if vggt_k.shape[1] != self.vggt_num_heads or vggt_k.shape[3] * vggt_k.shape[1] != self.vggt_width:
            raise ValueError("VGGT K/V head shape does not match configured width/heads.")
        first = fastwam_qkv[expert_order[0]][0]
        parameters = tuple(self.parameters())
        if parameters and any(p.dtype != first.dtype or p.device != first.device for p in parameters):
            raise ValueError(
                "VGGT bridge parameters must match FastWAM Q/K/V dtype/device; "
                "move the containing module with `.to(device=..., dtype=...)`."
            )
        if vggt_k.shape[0] != first.shape[0]:
            raise ValueError("VGGT and FastWAM batch sizes must match.")
        if vggt_k.dtype != first.dtype or vggt_v.dtype != first.dtype:
            raise ValueError("VGGT and FastWAM dtype must match.")
        if vggt_k.device != first.device or vggt_v.device != first.device:
            raise ValueError("VGGT and FastWAM device must match.")
        seq_lens: list[int] = []
        for name in expert_order:
            triple = fastwam_qkv[name]
            if len(triple) != 3 or any(t.ndim != 3 for t in triple):
                raise ValueError(f"FastWAM {name} Q/K/V must be [B,S,H*Dh].")
            if not (triple[0].shape == triple[1].shape == triple[2].shape):
                raise ValueError(f"FastWAM {name} Q/K/V shapes must match.")
            if triple[0].shape[0] != first.shape[0] or triple[0].shape[-1] != self.fastwam_width:
                raise ValueError(f"FastWAM {name} batch/width does not match configuration.")
            if any(t.dtype != first.dtype or t.device != first.device for t in triple):
                raise ValueError(f"FastWAM {name} Q/K/V dtype/device must match other experts.")
            seq_lens.append(triple[0].shape[1])
        total = sum(seq_lens)
        if attention_mask.dtype != torch.bool or attention_mask.ndim != 2 or tuple(attention_mask.shape) != (total, total):
            raise ValueError(f"`attention_mask` must be bool [{total},{total}].")
        if attention_mask.device != first.device:
            raise ValueError("`attention_mask` device must match Q/K/V.")
        return seq_lens

    def forward_video_prefill(
        self,
        fastwam_layer: int,
        video_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        vggt_k: torch.Tensor,
        vggt_v: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        vggt_layer_idx: int,
    ) -> torch.Tensor:
        """Bridge VGGT memory into video-only prefill attention.

        This path has no action input: video queries retain the video self-mask,
        while VGGT keys/values are appended as always-readable one-way memory.
        """
        if "video" not in self.target_experts:
            raise ValueError("Video-prefill bridge requires `video` in `target_experts`.")
        mapped_vggt_layer = self.vggt_layer_for(fastwam_layer)
        if mapped_vggt_layer is None:
            raise ValueError(f"FastWAM layer {fastwam_layer} is not mapped to VGGT memory.")
        if vggt_layer_idx != mapped_vggt_layer:
            raise ValueError(
                f"FastWAM layer {fastwam_layer} expects mapped VGGT layer "
                f"{mapped_vggt_layer}, got {vggt_layer_idx}."
            )
        if not isinstance(video_qkv, (tuple, list)) or len(video_qkv) != 3:
            raise ValueError("FastWAM video Q/K/V must be a triple of [B,S,H*Dh] tensors.")
        q, k, v = video_qkv
        if any(t.ndim != 3 for t in video_qkv) or not (q.shape == k.shape == v.shape):
            raise ValueError("FastWAM video Q/K/V must have matching [B,S,H*Dh] shapes.")
        if q.shape[-1] != self.fastwam_width:
            raise ValueError("FastWAM video Q/K/V width does not match configuration.")
        if vggt_k.ndim != 4 or vggt_v.ndim != 4 or vggt_k.shape != vggt_v.shape:
            raise ValueError("VGGT K/V must have the same [B,H,S,Dh] shape.")
        if vggt_k.shape[1] != self.vggt_num_heads or vggt_k.shape[1] * vggt_k.shape[3] != self.vggt_width:
            raise ValueError("VGGT K/V head shape does not match configured width/heads.")
        if vggt_k.shape[0] != q.shape[0]:
            raise ValueError("VGGT and FastWAM batch sizes must match.")
        if any(t.dtype != q.dtype or t.device != q.device for t in (*video_qkv, vggt_k, vggt_v)):
            raise ValueError("VGGT and FastWAM video Q/K/V dtype/device must match.")
        parameters = tuple(self.parameters())
        if parameters and any(p.dtype != q.dtype or p.device != q.device for p in parameters):
            raise ValueError(
                "VGGT bridge parameters must match FastWAM Q/K/V dtype/device; "
                "move the containing module with `.to(device=..., dtype=...)`."
            )
        seq_len = q.shape[1]
        if (
            attention_mask.dtype != torch.bool
            or attention_mask.ndim != 2
            or tuple(attention_mask.shape) != (seq_len, seq_len)
            or attention_mask.device != q.device
        ):
            raise ValueError(f"`attention_mask` must be bool [{seq_len},{seq_len}] on the Q/K/V device.")

        memory_k, memory_v = self._flatten_vggt_memory(vggt_k, vggt_v)
        memory_mask = torch.ones((seq_len, memory_k.shape[1]), dtype=torch.bool, device=q.device)
        mask = torch.cat([attention_mask, memory_mask], dim=1)
        projections = self._projections_for(fastwam_layer)
        if self.projection_mode == "vggt_to_fastwam":
            return self._sdpa(
                q,
                torch.cat([k, projections.vggt_k_proj(memory_k)], dim=1),
                torch.cat([v, projections.vggt_v_proj(memory_v)], dim=1),
                mask,
                self.fastwam_num_heads,
            )
        small = self._sdpa(
            projections.fastwam_q_proj(q),
            torch.cat([projections.fastwam_k_proj(k), memory_k], dim=1),
            torch.cat([projections.fastwam_v_proj(v), memory_v], dim=1),
            mask,
            self.vggt_num_heads,
        )
        return projections.output_proj(small)

    def forward(
        self,
        fastwam_layer: int,
        fastwam_qkv: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        vggt_k: torch.Tensor,
        vggt_v: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        expert_order: Sequence[str] = ("video", "action"),
        vggt_layer_idx: int,
        isolated_expert_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute per-expert outputs while preserving MoT query-row mask semantics.

        When ``isolated_expert_attention`` is enabled, every expert reads only
        its own FastWAM K/V diagonal block plus the shared VGGT memory. This is
        an opt-in fast path for callers whose off-diagonal mask blocks are all
        false; the default joint behavior remains backward compatible.
        """
        mapped_vggt_layer = self.vggt_layer_for(fastwam_layer)
        if mapped_vggt_layer is None:
            raise ValueError(f"FastWAM layer {fastwam_layer} is not mapped to VGGT memory.")
        if vggt_layer_idx != mapped_vggt_layer:
            raise ValueError(
                f"FastWAM layer {fastwam_layer} expects mapped VGGT layer "
                f"{mapped_vggt_layer}, got {vggt_layer_idx}."
            )
        seq_lens = self._validate(
            fastwam_qkv,
            vggt_k,
            vggt_v,
            attention_mask,
            expert_order,
            allow_partial_targets=isolated_expert_attention,
        )
        memory_k, memory_v = self._flatten_vggt_memory(vggt_k, vggt_v)
        all_k = None
        all_v = None
        if not isolated_expert_attention:
            all_k = torch.cat([fastwam_qkv[name][1] for name in expert_order], dim=1)
            all_v = torch.cat([fastwam_qkv[name][2] for name in expert_order], dim=1)
        outputs: dict[str, torch.Tensor] = {}
        row_start = 0
        memory_len = memory_k.shape[1]
        projections = self._projections_for(fastwam_layer)

        # These projections depend only on layer memory, not on the querying
        # expert. Hoisting them avoids building duplicate autograd graphs for
        # every bridge target at the same layer.
        projected_memory_k = None
        projected_memory_v = None
        projected_all_k = None
        projected_all_v = None
        if self.projection_mode == "vggt_to_fastwam":
            projected_memory_k = projections.vggt_k_proj(memory_k)
            projected_memory_v = projections.vggt_v_proj(memory_v)
        elif not isolated_expert_attention:
            projected_all_k = projections.fastwam_k_proj(all_k)
            projected_all_v = projections.fastwam_v_proj(all_v)

        for name, seq_len in zip(expert_order, seq_lens):
            q = fastwam_qkv[name][0]
            row_end = row_start + seq_len
            if isolated_expert_attention:
                fastwam_k = fastwam_qkv[name][1]
                fastwam_v = fastwam_qkv[name][2]
                row_mask = attention_mask[row_start:row_end, row_start:row_end]
            else:
                fastwam_k = all_k
                fastwam_v = all_v
                row_mask = attention_mask[row_start:row_end]
            if name not in self.target_experts:
                outputs[name] = self._sdpa(
                    q, fastwam_k, fastwam_v, row_mask, self.fastwam_num_heads
                )
            elif self.projection_mode == "vggt_to_fastwam":
                k = torch.cat([fastwam_k, projected_memory_k], dim=1)
                v = torch.cat([fastwam_v, projected_memory_v], dim=1)
                memory_mask = torch.ones((seq_len, memory_len), dtype=torch.bool, device=q.device)
                outputs[name] = self._sdpa(q, k, v, torch.cat([row_mask, memory_mask], dim=1), self.fastwam_num_heads)
            else:
                # Target attention runs separately in VGGT width. All allowed FastWAM
                # expert K/V are projected too, preserving MoT inter-read semantics
                # without ever concatenating 3072-wide and 1024-wide tensors.
                q_small = projections.fastwam_q_proj(q)
                if isolated_expert_attention:
                    projected_k = projections.fastwam_k_proj(fastwam_k)
                    projected_v = projections.fastwam_v_proj(fastwam_v)
                else:
                    projected_k = projected_all_k
                    projected_v = projected_all_v
                k_small = torch.cat([projected_k, memory_k], dim=1)
                v_small = torch.cat([projected_v, memory_v], dim=1)
                memory_mask = torch.ones((seq_len, memory_len), dtype=torch.bool, device=q.device)
                small = self._sdpa(q_small, k_small, v_small, torch.cat([row_mask, memory_mask], dim=1), self.vggt_num_heads)
                outputs[name] = projections.output_proj(small)
            row_start = row_end
        return outputs
