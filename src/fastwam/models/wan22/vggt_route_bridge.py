"""Batch-routed bridges; ActionDiT remains a single shared transformer."""
from __future__ import annotations

import copy

import torch
from torch import nn


class VGGTRouteBridge(nn.Module):
    """Select independent high/low bridges with an explicit per-sample mask.

    Action Q/K/V [B,Sa,Da] and visual K/V [B,Hv,Sv,Dh] are partitioned
    along B, attended independently, then restored to [B,Sa,Da]. Routing
    lives in the memory dictionary, so checkpoint recomputation cannot read
    a stale mutable route. The token attention mask [Sa,Sa] is unchanged.
    """

    required_memory_keys = frozenset({"layer_idx", "k", "v", "high_noise_mask"})

    def __init__(self, bridge):
        super().__init__()
        self.high = bridge
        self.low = copy.deepcopy(bridge)
        for name in (
            "target_experts", "projection_mode", "projection_sharing",
            "fastwam_width", "vggt_width", "fastwam_num_heads", "vggt_num_heads",
        ):
            setattr(self, name, getattr(bridge, name))

    @property
    def detach_vggt_memory(self):
        return self.high.detach_vggt_memory

    @detach_vggt_memory.setter
    def detach_vggt_memory(self, value):
        self.high.detach_vggt_memory = value
        self.low.detach_vggt_memory = value

    def vggt_layer_for(self, layer):
        return self.high.vggt_layer_for(layer)

    def forward_memory(self, layer, qkv, memory, attention_mask, **kwargs):
        mask = memory["high_noise_mask"]
        if mask.dtype != torch.bool or mask.shape != (memory["k"].shape[0],):
            raise ValueError("Bridge high_noise_mask must be boolean [B].")
        indices, results = [], []
        anchor = memory["k"].reshape(-1)[:0].sum()
        for selected, bridge in ((mask, self.high), (~mask, self.low)):
            index = selected.nonzero(as_tuple=False).flatten()
            if index.numel():
                sub_qkv = {name: tuple(x.index_select(0, index) for x in values)
                           for name, values in qkv.items()}
                result = bridge(
                    layer, sub_qkv, memory["k"].index_select(0, index),
                    memory["v"].index_select(0, index), attention_mask, **kwargs,
                )
                indices.append(index)
                results.append(result)
            elif self.training and torch.is_grad_enabled():
                # Execute only this layer's inactive bridge on one dummy row.
                # A zero output connection keeps ZeRO gradient hooks balanced
                # across ranks, without updating the inactive route.
                dummy = bridge(layer, {n: tuple(x[:1] for x in v) for n, v in qkv.items()},
                               memory["k"][:1], memory["v"][:1], attention_mask, **kwargs)
                anchor = anchor + sum(x.reshape(-1)[:0].sum() for x in dummy.values())
        inverse = torch.argsort(torch.cat(indices))
        return {name: torch.cat([result[name] for result in results], dim=0)
                .index_select(0, inverse) + anchor for name in results[0]}
