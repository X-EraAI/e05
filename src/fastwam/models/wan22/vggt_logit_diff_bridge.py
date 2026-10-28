from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vggt_bridge import LayerMapping


class _LogitDiffProjectionSet(nn.Module):
    """Projection parameters for one logit-difference bridge layer."""

    def __init__(self, projection_mode: str, fastwam_width: int, vggt_width: int) -> None:
        super().__init__()
        if projection_mode == "vggt_to_fastwam":
            # VGGT2 supplies a key only.  Its value is deliberately unused.
            self.vggt1_k_proj = nn.Linear(vggt_width, fastwam_width, bias=False)
            self.vggt2_k_proj = nn.Linear(vggt_width, fastwam_width, bias=False)
            self.vggt1_v_proj = nn.Linear(vggt_width, fastwam_width, bias=False)
        else:
            self.fastwam_q_proj = nn.Linear(fastwam_width, vggt_width, bias=False)
            self.fastwam_k_proj = nn.Linear(fastwam_width, vggt_width, bias=False)
            self.fastwam_v_proj = nn.Linear(fastwam_width, vggt_width, bias=False)
            self.output_proj = nn.Linear(vggt_width, fastwam_width, bias=False)


class VGGTLogitDiffAttentionBridge(nn.Module):
    """Joint action attention with contrastive VGGT memory logits.

    The bridge receives action/video Q/K/V in flat ``[B,S,D]`` format and a
    memory dictionary containing two aligned VGGT streams:

    ``va1_k/va1_v/va2_k``: ``[B,Hv,M,Dv]``.

    Tensor flow in ``vggt_to_fastwam`` mode is:

    ``action Q/K/V [B,Sa,Df]`` -> ``S_action [B,Ha,Sa,Sa]``
    ``VGGT K1/K2 [B,M,Dv]`` -> ``S1/S2 [B,Ha,Sa,M]``
    ``S_mem = (S1 - lambda*S2) / temperature``
    ``concat(S_action,S_mem) -> softmax -> concat(V_action,V1)``.

    VGGT2 therefore contributes only a negative logit signal; its value is
    never concatenated into the weighted-sum values.  Logits and softmax are
    evaluated in fp32 for stable mixed-precision training.
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
        target_experts: Sequence[str] = ("action",),
        projection_mode: str,
        projection_sharing: str = "shared",
        contrastive_lambda: float = 0.1,
        contrastive_temperature: float = 1.0,
        detach_vggt_memory: bool = True,
    ) -> None:
        super().__init__()
        if projection_mode not in self.VALID_MODES:
            raise ValueError(f"Invalid `projection_mode`: {projection_mode!r}.")
        if projection_sharing not in self.VALID_PROJECTION_SHARING:
            raise ValueError(f"Invalid `projection_sharing`: {projection_sharing!r}.")
        if not math.isfinite(float(contrastive_lambda)) or float(contrastive_lambda) < 0:
            raise ValueError("`contrastive_lambda` must be finite and non-negative.")
        if not math.isfinite(float(contrastive_temperature)) or float(contrastive_temperature) <= 0:
            raise ValueError("`contrastive_temperature` must be finite and positive.")
        targets = tuple(target_experts)
        if not targets or len(set(targets)) != len(targets) or not set(targets) <= self.VALID_EXPERTS:
            raise ValueError("`target_experts` must be a unique non-empty expert subset.")
        if fastwam_width <= 0 or vggt_width <= 0:
            raise ValueError("Attention widths must be positive.")
        if fastwam_num_heads <= 0 or vggt_num_heads <= 0:
            raise ValueError("Attention head counts must be positive.")
        if fastwam_width % fastwam_num_heads or vggt_width % vggt_num_heads:
            raise ValueError("Attention widths must be divisible by head counts.")

        self.layer_mapping = LayerMapping(
            layer_pairs,
            vggt_num_layers=int(vggt_num_layers),
            fastwam_num_layers=int(fastwam_num_layers),
        )
        self.target_experts = frozenset(targets)
        self.projection_mode = str(projection_mode)
        self.projection_sharing = str(projection_sharing)
        self.contrastive_lambda = float(contrastive_lambda)
        self.contrastive_temperature = float(contrastive_temperature)
        self.detach_vggt_memory = bool(detach_vggt_memory)
        self.fastwam_width = int(fastwam_width)
        self.vggt_width = int(vggt_width)
        self.fastwam_num_heads = int(fastwam_num_heads)
        self.vggt_num_heads = int(vggt_num_heads)

        if projection_sharing == "independent":
            self.layer_projections = nn.ModuleDict({
                str(fastwam_layer): _LogitDiffProjectionSet(
                    self.projection_mode, self.fastwam_width, self.vggt_width
                )
                for _, fastwam_layer in layer_pairs
            })
        else:
            self._shared_projections = _LogitDiffProjectionSet(
                self.projection_mode, self.fastwam_width, self.vggt_width
            )

    def vggt_layer_for(self, fastwam_layer: int) -> int | None:
        return self.layer_mapping.vggt_layer_for(fastwam_layer)

    def _projections_for(self, fastwam_layer: int) -> nn.Module:
        if self.projection_sharing == "independent":
            return self.layer_projections[str(fastwam_layer)]
        return self._shared_projections

    @staticmethod
    def _flat_vggt(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError("VGGT memory tensors must have shape [B,H,M,Dh].")
        return tensor.transpose(1, 2).contiguous().flatten(2)

    @staticmethod
    def _heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
        batch, seq_len, width = tensor.shape
        if width % heads:
            raise ValueError(f"Tensor width {width} is not divisible by {heads} heads.")
        return tensor.view(batch, seq_len, heads, width // heads).transpose(1, 2)

    @staticmethod
    def _scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # [B,H,Sa,Dh] @ [B,H,Dh,M] -> [B,H,Sa,M]。
        # 仅 .float() 不够：外层 AMP 会把 matmul 再降到 bf16，必须显式禁用。
        with torch.autocast(device_type=q.device.type, enabled=False):
            return torch.matmul(q.float(), k.float().transpose(-2, -1)) / (q.shape[-1] ** 0.5)

    @staticmethod
    def _weighted_values(weights: torch.Tensor, values: torch.Tensor, heads: int) -> torch.Tensor:
        # fp32 softmax 权重乘 [V_action,V1]，合并 head 后才恢复模型 dtype。
        with torch.autocast(device_type=values.device.type, enabled=False):
            output = torch.matmul(weights.float(), values.float())
        output = output.transpose(1, 2).contiguous().flatten(2)
        return output.to(dtype=values.dtype)

    def _validate_memory(
        self,
        memory: Mapping[str, object],
        *,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        required = {"layer_idx", "va1_k", "va1_v", "va2_k"}
        missing = sorted(required - set(memory))
        if missing:
            raise ValueError(f"VGGT logit-difference memory is missing keys: {missing}.")
        va1_k = memory["va1_k"]
        va1_v = memory["va1_v"]
        va2_k = memory["va2_k"]
        if not all(isinstance(tensor, torch.Tensor) for tensor in (va1_k, va1_v, va2_k)):
            raise TypeError("va1_k, va1_v, and va2_k must be tensors.")
        if va1_k.ndim != 4 or va1_v.ndim != 4 or va2_k.ndim != 4:
            raise ValueError("VGGT K/V tensors must have shape [B,H,M,Dh].")
        if va1_k.shape != va1_v.shape or va1_k.shape != va2_k.shape:
            raise ValueError(
                "Aligned VGGT streams must have identical K/V shapes, got "
                f"va1_k={tuple(va1_k.shape)}, va1_v={tuple(va1_v.shape)}, "
                f"va2_k={tuple(va2_k.shape)}."
            )
        if va1_k.shape[1] != self.vggt_num_heads or va1_k.shape[3] * va1_k.shape[1] != self.vggt_width:
            raise ValueError("VGGT head shape does not match configured width/heads.")
        if va1_k.shape[0] != reference.shape[0]:
            raise ValueError("VGGT and ActionDiT batch sizes must match.")
        if va1_k.shape[2] <= 0:
            raise ValueError("VGGT memory token count must be positive.")
        if any(tensor.dtype != reference.dtype or tensor.device != reference.device for tensor in (va1_k, va1_v, va2_k)):
            raise ValueError("VGGT and ActionDiT tensors must share dtype and device.")
        return va1_k, va1_v, va2_k

    def _validate_qkv(
        self,
        fastwam_qkv: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor,
        expert_order: Sequence[str],
    ) -> list[int]:
        if tuple(expert_order) != tuple(fastwam_qkv.keys()):
            raise ValueError("`expert_order` must match FastWAM Q/K/V insertion order.")
        if not expert_order or not set(expert_order) <= self.VALID_EXPERTS:
            raise ValueError("Invalid FastWAM expert order.")
        first = fastwam_qkv[expert_order[0]][0]
        if first.ndim != 3:
            raise ValueError("FastWAM Q/K/V must have shape [B,S,D].")
        seq_lens: list[int] = []
        for name in expert_order:
            q, k, v = fastwam_qkv[name]
            if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
                raise ValueError(f"FastWAM {name} Q/K/V must have matching [B,S,D] shapes.")
            if q.shape[0] != first.shape[0] or q.shape[-1] != self.fastwam_width:
                raise ValueError(f"FastWAM {name} batch/width does not match configuration.")
            if any(tensor.dtype != first.dtype or tensor.device != first.device for tensor in (q, k, v)):
                raise ValueError("FastWAM Q/K/V tensors must share dtype and device.")
            seq_lens.append(int(q.shape[1]))
        total = sum(seq_lens)
        if attention_mask.dtype != torch.bool or tuple(attention_mask.shape) != (total, total):
            raise ValueError(f"`attention_mask` must be bool [{total},{total}].")
        if attention_mask.device != first.device:
            raise ValueError("`attention_mask` device must match FastWAM tensors.")
        return seq_lens

    def _attention_for_target(
        self,
        *,
        fastwam_layer: int,
        q: torch.Tensor,
        fastwam_k: torch.Tensor,
        fastwam_v: torch.Tensor,
        row_mask: torch.Tensor,
        va1_k: torch.Tensor,
        va1_v: torch.Tensor,
        va2_k: torch.Tensor,
    ) -> torch.Tensor:
        """action Q/K/V -> 同一 head 空间 -> 联合 logits -> 加权和 [B,Sa,Df]。

        vggt_to_fastwam 将 K1/K2/V1 投影到 Df；fastwam_to_vggt 将
        action Q/K/V 投影到 Hv*Dv，attention 后通过 output_proj 回到 Df。
        输出随后由 MoT 交还 ActionDiT 的 output projection/residual/FFN。
        """
        projections = self._projections_for(fastwam_layer)
        if self.projection_mode == "vggt_to_fastwam":
            q_width = self.fastwam_num_heads
            memory_k1 = projections.vggt1_k_proj(self._flat_vggt(va1_k))
            memory_k2 = projections.vggt2_k_proj(self._flat_vggt(va2_k))
            memory_v1 = projections.vggt1_v_proj(self._flat_vggt(va1_v))
            qh = self._heads(q, q_width)
            kh = self._heads(fastwam_k, q_width)
            vh = self._heads(fastwam_v, q_width)
            km1 = self._heads(memory_k1, q_width)
            km2 = self._heads(memory_k2, q_width)
            vm1 = self._heads(memory_v1, q_width)
            output_projection = None
        else:
            q_small = projections.fastwam_q_proj(q)
            k_small = projections.fastwam_k_proj(fastwam_k)
            v_small = projections.fastwam_v_proj(fastwam_v)
            q_width = self.vggt_num_heads
            qh = self._heads(q_small, q_width)
            kh = self._heads(k_small, q_width)
            vh = self._heads(v_small, q_width)
            # VGGT memory is already head-major [B,H,M,Dh].
            km1 = va1_k
            km2 = va2_k
            vm1 = va1_v
            output_projection = projections.output_proj

        # 三个 logits 都为 [B,H,Sa,S]；memory 的 S=M，action 的 S=Sa。
        score_action = self._scores(qh, kh)
        score_1 = self._scores(qh, km1)
        score_2 = self._scores(qh, km2)
        score_memory = (score_1 - self.contrastive_lambda * score_2) / self.contrastive_temperature

        memory_mask = torch.ones(
            (q.shape[1], score_memory.shape[-1]), dtype=torch.bool, device=q.device
        )
        joint_mask = torch.cat([row_mask, memory_mask], dim=1)[None, None]
        joint_scores = torch.cat([score_action, score_memory], dim=-1)
        joint_scores = joint_scores.masked_fill(~joint_mask, torch.finfo(torch.float32).min)
        weights = F.softmax(joint_scores, dim=-1, dtype=torch.float32)
        values = torch.cat([vh, vm1], dim=2)
        output = self._weighted_values(weights, values, q_width)
        if output_projection is not None:
            output = output_projection(output)
        return output.to(dtype=q.dtype)

    def forward_memory(
        self,
        fastwam_layer: int,
        fastwam_qkv: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        memory: Mapping[str, object],
        attention_mask: torch.Tensor,
        *,
        expert_order: Sequence[str] = ("action",),
        vggt_layer_idx: int,
        isolated_expert_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Consume aligned va1/va2 memory while keeping the legacy MoT API intact."""
        mapped = self.vggt_layer_for(fastwam_layer)
        if mapped is None or int(vggt_layer_idx) != int(mapped):
            raise ValueError(
                f"FastWAM layer {fastwam_layer} expects VGGT layer {mapped}, got {vggt_layer_idx}."
            )
        if memory.get("layer_idx") != vggt_layer_idx:
            raise ValueError("Memory callback layer_idx does not match the mapped VGGT layer.")
        seq_lens = self._validate_qkv(fastwam_qkv, attention_mask, expert_order)
        first = fastwam_qkv[expert_order[0]][0]
        va1_k, va1_v, va2_k = self._validate_memory(memory, reference=first)
        parameters = tuple(self.parameters())
        if parameters and any(p.dtype != first.dtype or p.device != first.device for p in parameters):
            raise ValueError("Logit-difference bridge parameters must match FastWAM tensors.")
        if self.detach_vggt_memory:
            va1_k, va1_v, va2_k = va1_k.detach(), va1_v.detach(), va2_k.detach()

        outputs: dict[str, torch.Tensor] = {}
        row_start = 0
        all_k = None
        all_v = None
        if not isolated_expert_attention:
            all_k = torch.cat([fastwam_qkv[name][1] for name in expert_order], dim=1)
            all_v = torch.cat([fastwam_qkv[name][2] for name in expert_order], dim=1)
        for name, seq_len in zip(expert_order, seq_lens):
            row_end = row_start + seq_len
            q = fastwam_qkv[name][0]
            if isolated_expert_attention:
                fastwam_k = fastwam_qkv[name][1]
                fastwam_v = fastwam_qkv[name][2]
                row_mask = attention_mask[row_start:row_end, row_start:row_end]
            else:
                fastwam_k = all_k
                fastwam_v = all_v
                row_mask = attention_mask[row_start:row_end]
            if name in self.target_experts:
                outputs[name] = self._attention_for_target(
                    fastwam_layer=fastwam_layer,
                    q=q,
                    fastwam_k=fastwam_k,
                    fastwam_v=fastwam_v,
                    row_mask=row_mask,
                    va1_k=va1_k,
                    va1_v=va1_v,
                    va2_k=va2_k,
                )
            else:
                # Non-target experts retain the original FastWAM attention.
                outputs[name] = self._plain_attention(q, fastwam_k, fastwam_v, row_mask)
            row_start = row_end
        return outputs

    def _plain_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        qh = self._heads(q, self.fastwam_num_heads)
        kh = self._heads(k, self.fastwam_num_heads)
        vh = self._heads(v, self.fastwam_num_heads)
        scores = self._scores(qh, kh).masked_fill(
            ~mask[None, None], torch.finfo(torch.float32).min
        )
        weights = F.softmax(scores, dim=-1)
        return self._weighted_values(weights, vh, self.fastwam_num_heads).to(dtype=q.dtype)


__all__ = ["VGGTLogitDiffAttentionBridge"]
