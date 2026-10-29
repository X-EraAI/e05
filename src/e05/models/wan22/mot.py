from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .attention import flash_attention, modulate, rope_apply
from e05.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        vggt_bridge: Optional[nn.Module] = None,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        unknown_experts = set(mixtures) - {"action"}
        if unknown_experts:
            raise ValueError(f"Unsupported experts in `mixtures`: {sorted(unknown_experts)}.")
        if "action" not in mixtures:
            raise ValueError("`mixtures` must include the 'action' expert.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.vggt_bridge = vggt_bridge
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )

        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )



    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        vggt_kv_by_layer: Optional[Dict[int, dict[str, torch.Tensor]]] = None,
        context_by_layer_all: Optional[
            Dict[str, Dict[int, Optional[dict]]]
        ] = None,
        stop_action_grad_to_video: bool = False,
        isolated_expert_attention: bool = False,
        active_experts: Optional[Sequence[str]] = None,
    ):
        if active_experts is None:
            expert_order = tuple(self.expert_order)
        else:
            if not isolated_expert_attention:
                raise ValueError(
                    "`active_experts` is only supported with "
                    "`isolated_expert_attention=True`."
                )
            if isinstance(active_experts, str):
                raise TypeError("`active_experts` must be a sequence of expert names.")
            expert_order = tuple(active_experts)
            if not expert_order or len(set(expert_order)) != len(expert_order):
                raise ValueError("`active_experts` must be non-empty and contain no duplicates.")
            unknown = set(expert_order) - set(self.expert_order)
            if unknown:
                raise ValueError(f"Unknown active experts: {sorted(unknown)}.")
            registered_subsequence = tuple(
                name for name in self.expert_order if name in expert_order
            )
            if expert_order != registered_subsequence:
                raise ValueError(
                    "`active_experts` must preserve registered expert order: "
                    f"expected {registered_subsequence}, got {expert_order}."
                )
            if stop_action_grad_to_video:
                raise ValueError(
                    "`active_experts` cannot be combined with "
                    "`stop_action_grad_to_video=True`."
                )

        missing = [k for k in expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")
        if stop_action_grad_to_video:
            if self.vggt_bridge is not None:
                raise ValueError(
                    "`stop_action_grad_to_video=True` is only supported for base E05 "
                    "without a VGGT bridge."
                )
            missing_stop_grad_experts = {"video", "action"} - set(self.expert_order)
            if missing_stop_grad_experts:
                raise ValueError(
                    "`stop_action_grad_to_video=True` requires both video and action experts; "
                    f"missing {sorted(missing_stop_grad_experts)}."
                )
        if context_by_layer_all is not None:
            unknown_context_experts = set(context_by_layer_all) - set(expert_order)
            if unknown_context_experts:
                raise ValueError(
                    "`context_by_layer_all` contains unknown experts: "
                    f"{sorted(unknown_context_experts)}."
                )
            for name, contexts_by_layer in context_by_layer_all.items():
                if not isinstance(contexts_by_layer, dict):
                    raise TypeError(
                        f"`context_by_layer_all[{name!r}]` must be a dict keyed by layer."
                    )
                invalid_layers = [
                    layer_idx
                    for layer_idx in contexts_by_layer
                    if (
                        isinstance(layer_idx, bool)
                        or not isinstance(layer_idx, int)
                        or not 0 <= layer_idx < self.num_layers
                    )
                ]
                if invalid_layers:
                    raise ValueError(
                        f"`context_by_layer_all[{name!r}]` contains invalid layers "
                        f"{invalid_layers}; expected integers in [0,{self.num_layers})."
                    )

        tokens_all = {name: embeds_all[name] for name in expert_order}

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 3. Run the original mixed path, unless this E05 layer maps
            # to an available VGGT callback memory entry. Cache keys are VGGT
            # layer indices, not E05 indices, preventing silent misalignment.
            q_cat = None
            if not stop_action_grad_to_video and not isolated_expert_attention:
                q_cat = torch.cat(q_chunks, dim=1)
            k_cat = None
            v_cat = None
            if not isolated_expert_attention:
                k_cat = torch.cat(k_chunks, dim=1)
                v_cat = torch.cat(v_chunks, dim=1)

            total_seq = sum(seq_lens)
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            bridge_outputs = None
            if self.vggt_bridge is not None and vggt_kv_by_layer is not None:
                vggt_layer_idx = self.vggt_bridge.vggt_layer_for(layer_idx)
                memory = vggt_kv_by_layer.get(vggt_layer_idx) if vggt_layer_idx is not None else None
                if memory is not None:
                    uses_memory_dict = hasattr(self.vggt_bridge, "forward_memory")
                    required_memory = (
                        getattr(self.vggt_bridge, "required_memory_keys",
                                {"layer_idx", "va1_k", "va1_v", "va2_k"})
                        if uses_memory_dict else {"layer_idx", "k", "v"}
                    )
                    if not isinstance(memory, dict) or not required_memory <= memory.keys():
                        raise ValueError(
                            f"VGGT memory for layer {vggt_layer_idx} must contain {sorted(required_memory)}."
                        )
                    qkv = {
                        name: (
                            q_chunks[idx],
                            k_chunks[idx].to(dtype=q_chunks[idx].dtype),
                            v_chunks[idx].to(dtype=q_chunks[idx].dtype),
                        )
                        for idx, name in enumerate(expert_order)
                    }
                    bridge_memory = dict(memory)
                    attention_dtype = qkv[expert_order[0]][0].dtype
                    for key, value in tuple(bridge_memory.items()):
                        if isinstance(value, torch.Tensor) and (
                            key in {"k", "v"} or key.endswith(("_k", "_v"))
                        ):
                            bridge_memory[key] = value.to(dtype=attention_dtype)
                    if uses_memory_dict:
                        # 保留两路 memory 边界。非重入 checkpoint 支持嵌套字典，
                        # 反向重算 logits/softmax，同时向 Q/K/V 与两路 VGGT 传梯度。
                        # 层号直接作为参数传入，避免循环闭包在重算时引用最后一层。
                        bridge_kwargs = dict(
                            expert_order=expert_order,
                            vggt_layer_idx=memory["layer_idx"],
                            isolated_expert_attention=isolated_expert_attention,
                        )
                        bridge_args = (layer_idx, qkv, bridge_memory, attention_mask)
                        if self.mot_checkpoint_mixed_attn and self.training:
                            bridge_outputs = torch.utils.checkpoint.checkpoint(
                                self.vggt_bridge.forward_memory,
                                *bridge_args,
                                use_reentrant=False,
                                **bridge_kwargs,
                            )
                        else:
                            bridge_outputs = self.vggt_bridge.forward_memory(
                                *bridge_args, **bridge_kwargs,
                            )
                    else:
                        bridge_outputs = self.vggt_bridge(
                            layer_idx,
                            qkv,
                            bridge_memory["k"],
                            bridge_memory["v"],
                            attention_mask,
                            expert_order=expert_order,
                            vggt_layer_idx=memory["layer_idx"],
                            isolated_expert_attention=isolated_expert_attention,
                        )

            mixed = None
            mixed_by_expert = None
            if isolated_expert_attention and bridge_outputs is None:
                mixed_by_expert = {}
                block_start = 0
                for expert_idx, (name, seq_len) in enumerate(
                    zip(expert_order, seq_lens)
                ):
                    block_end = block_start + seq_len
                    mixed_by_expert[name] = self._mixed_attention(
                        q_cat=q_chunks[expert_idx],
                        k_cat=k_chunks[expert_idx],
                        v_cat=v_chunks[expert_idx],
                        attention_mask=attention_mask[
                            block_start:block_end, block_start:block_end
                        ],
                    )
                    block_start = block_end
            elif stop_action_grad_to_video:
                mixed_by_expert = {}
                row_start = 0
                for query_idx, (query_name, query_seq_len) in enumerate(
                    zip(expert_order, seq_lens)
                ):
                    row_end = row_start + query_seq_len
                    row_k = k_cat
                    row_v = v_cat
                    if query_name == "action":
                        row_k = torch.cat(
                            [
                                source_k.detach() if source_name == "video" else source_k
                                for source_name, source_k in zip(expert_order, k_chunks)
                            ],
                            dim=1,
                        )
                        row_v = torch.cat(
                            [
                                source_v.detach() if source_name == "video" else source_v
                                for source_name, source_v in zip(expert_order, v_chunks)
                            ],
                            dim=1,
                        )
                    mixed_by_expert[query_name] = self._mixed_attention(
                        q_cat=q_chunks[query_idx],
                        k_cat=row_k,
                        v_cat=row_v,
                        attention_mask=attention_mask[row_start:row_end],
                    )
                    row_start = row_end
            elif bridge_outputs is None:
                mixed = self._mixed_attention(
                    q_cat=q_cat,
                    k_cat=k_cat,
                    v_cat=v_cat,
                    attention_mask=attention_mask,
                )

            start = 0
            for name, seq_len in zip(expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                if mixed_by_expert is not None:
                    mixed_slice = mixed_by_expert[name]
                elif bridge_outputs is not None:
                    mixed_slice = bridge_outputs[name]
                else:
                    mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)
                # A model may override cross-attention context at selected
                # transformer depths while all other layers retain the normal
                # language/state payload. This is intentionally independent of
                # the VGGT K/V shared-attention bridge above.
                if context_by_layer_all is not None:
                    context_payload = context_by_layer_all.get(name, {}).get(
                        layer_idx, context_payload
                    )

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all
