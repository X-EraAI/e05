"""Independent visual-stream projections followed by one joint attention."""
from __future__ import annotations

import copy

import torch
from torch import nn


class VGGTStreamBridge(nn.Module):
    """Project va1/va2 independently without splitting ActionDiT attention.

    Visual memory [B,Hv,S1+S2,Dh] is split along S, flattened to [B,Si,Dv],
    then projected by its owner's K/V maps to [B,Si,Da]. A single SDPA reads
    [action K/V, projected va1, projected va2]. Source identity is explicit
    in memory, including when only one stream is selected; no mutable routing
    state is used, so activation-checkpoint recomputation is deterministic.
    """

    required_memory_keys = frozenset({
        "layer_idx", "k", "v", "source_seq_lens", "action_vggt_source",
    })

    def __init__(self, bridge):
        super().__init__()
        if bridge.projection_mode != "vggt_to_fastwam":
            raise ValueError("share_bridge=false requires projection_mode=vggt_to_fastwam.")
        if bridge.target_experts != frozenset({"action"}):
            raise ValueError("VGGTStreamBridge supports the action-only model.")
        self.va1 = bridge
        self.va2 = copy.deepcopy(bridge)
        for name in ("target_experts", "projection_mode", "projection_sharing",
                     "fastwam_width", "vggt_width", "fastwam_num_heads", "vggt_num_heads"):
            setattr(self, name, getattr(bridge, name))

    @property
    def layer_mapping(self):
        return self.va1.layer_mapping

    @property
    def detach_vggt_memory(self):
        return self.va1.detach_vggt_memory

    @detach_vggt_memory.setter
    def detach_vggt_memory(self, value):
        self.va1.detach_vggt_memory = self.va2.detach_vggt_memory = value

    def vggt_layer_for(self, layer):
        return self.va1.vggt_layer_for(layer)

    def forward_memory(self, layer, qkv, memory, attention_mask, *, expert_order,
                       vggt_layer_idx, isolated_expert_attention=False):
        if tuple(expert_order) != ("action",):
            raise ValueError("VGGTStreamBridge requires action-only Q/K/V.")
        if self.vggt_layer_for(layer) != vggt_layer_idx:
            raise ValueError("VGGT stream bridge layer mapping mismatch.")
        source = memory["action_vggt_source"]
        if source not in ("both", "va1", "va2"):
            raise ValueError(f"Invalid action_vggt_source: {source}")
        names = ("va1", "va2") if source == "both" else (source,)
        lengths = [int(memory["source_seq_lens"][name]) for name in names]
        if any(n <= 0 for n in lengths) or sum(lengths) != memory["k"].shape[2]:
            raise ValueError("Visual source lengths do not match concatenated K/V.")
        self.va1._validate(qkv, memory["k"], memory["v"], attention_mask, expert_order)
        k, v = self.va1._flatten_vggt_memory(memory["k"], memory["v"])
        projected_k, projected_v = [], []
        for name, ki, vi in zip(names, k.split(lengths, dim=1), v.split(lengths, dim=1)):
            projections = getattr(self, name)._projections_for(layer)
            projected_k.append(projections.vggt_k_proj(ki))
            projected_v.append(projections.vggt_v_proj(vi))
        q, action_k, action_v = qkv["action"]
        keys = torch.cat([action_k, *projected_k], dim=1)
        values = torch.cat([action_v, *projected_v], dim=1)
        allowed = torch.cat([attention_mask, torch.ones(
            (q.shape[1], sum(lengths)), dtype=torch.bool, device=q.device)], dim=1)
        output = self.va1._sdpa(q, keys, values, allowed, self.fastwam_num_heads)
        # Single-source runs still keep inactive bridge parameters connected
        # with zero gradients for ZeRO. No duplicate attention is performed.
        if self.training and torch.is_grad_enabled():
            for name in {"va1", "va2"} - set(names):
                for parameter in getattr(self, name)._projections_for(layer).parameters():
                    output = output + parameter.reshape(-1)[:0].sum()
        return {"action": output}
