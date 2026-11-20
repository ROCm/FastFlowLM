# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Quantize the Qwen3.5-Omni *thinker* LLM (HF bf16 safetensors) to the Q4NX format.

This reads the HF-native ``qwen35-omni-bf16`` checkpoint directly (no GGUF step) and
produces ``model.q4nx`` following ``configs/qwen3.5_omni.json``.

Only the thinker language model is handled here (``thinker.model.*``). The audio
tower, vision tower, talker and code2wav stacks are ignored.

Two structural facts drive the code:

1.  The thinker text stack alternates linear-attention and full-attention layers
    (see ``layer_types`` in ``config.json``). Full-attention layers carry the usual
    q/k/v/o projections; linear-attention layers carry the gated-delta-net SSM with a
    *fused* ``in_proj_qkvz`` and ``in_proj_ba``.

2.  The Q4NX ``name_map`` expects the SSM projections already *split* (qkv_proj,
    self_attn.gate_proj (= z), ssm_alpha_proj (= a), ssm_beta_proj (= b), ...). We
    reproduce the HF ``fix_query_key_value_ordering`` split, adapted to weight
    matrices ``[out_features, in_features]`` (the reference snippet operates on
    activations ``[b, sq, proj]`` and splits the last dim; for a weight the projected
    dimension is dim-0).

The heavy Q4NX bit-packing (``_pack_q4nx`` / ``_pack_q8nx``) is reused verbatim from
``q4nx.model_converter.__Q4NX_Converter`` via duck-typing -- we only need the packer to
see ``row_block_size`` / ``col_block_size`` / ``parallel_size`` / ``keep_block_in_2D``,
so we borrow the unbound methods rather than subclassing (the base class is an ABC tied
to the GGUF registry).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, Tuple

# Make the FLM_Q4NX_Converter root importable so ``q4nx`` resolves regardless of
# the current working directory this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gguf import quantize, GGMLQuantizationType

# Reuse the exact Q4NX packing routines used by the GGUF path.
from q4nx.model_converter import __Q4NX_Converter as _Q4NXBase
from q4nx.gguf_tensor import GGUFTensor


class Qwen35OmniThinkerQuantizer:
    """Quantize the Qwen3.5-Omni thinker to Q4NX from HF bf16 safetensors."""

    HF_PREFIX = "thinker.model"

    def __init__(self, model_dir: str, config_path: str, output_dir: str):
        self.model_dir = model_dir
        self.config_path = config_path
        self.output_dir = output_dir

        self._load_config()
        self._load_model_metadata()

        self.q4nx_tensors: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ config

    def _load_config(self):
        with open(self.config_path) as f:
            self.q4nx_config = json.load(f)

        qc = self.q4nx_config["q4nx_config"]
        # These four attributes are the only state the borrowed packers read.
        self.row_block_size = qc["row_block_size"]
        self.col_block_size = qc["col_block_size"]
        self.parallel_size = qc["parallel_size"]
        self.keep_block_in_2D = qc["keep_block_in_2D"]

        self.name_map = self.q4nx_config["name_map"]

    def _tensor_type(self, config_key: str) -> GGMLQuantizationType:
        """Resolve the target Q4NX type for a config entry."""
        t = self.name_map[config_key].get(
            "default_tensor_type", self.q4nx_config["default_tensor_type"]
        )
        return {
            "Q4_0": GGMLQuantizationType.Q4_0,
            "Q4_1": GGMLQuantizationType.Q4_1,
            "Q8_0": GGMLQuantizationType.Q8_0,
            "BF16": GGMLQuantizationType.BF16,
            "F16": GGMLQuantizationType.F16,
            "F32": GGMLQuantizationType.F32,
        }[t]

    def _q4nx_name(self, config_key: str, bid: int | None = None) -> str:
        name = self.name_map[config_key]["q4nx_name"]
        return name.format(bid=bid) if bid is not None else name

    # ---------------------------------------------------------- model metadata

    def _load_model_metadata(self):
        """Read the safetensors index + config.json (dims and layer_types)."""
        index_path = os.path.join(self.model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            self.weight_map: Dict[str, str] = json.load(f)["weight_map"]

        with open(os.path.join(self.model_dir, "config.json")) as f:
            hf_cfg = json.load(f)
        text_cfg = hf_cfg["thinker_config"]["text_config"]

        self.num_layers = text_cfg["num_hidden_layers"]
        self.hidden_size = text_cfg["hidden_size"]
        self.head_dim = text_cfg["head_dim"]
        self.num_attention_heads = text_cfg["num_attention_heads"]
        self.num_key_value_heads = text_cfg["num_key_value_heads"]
        self.tie_word_embeddings = text_cfg["tie_word_embeddings"]
        self.layer_types = text_cfg["layer_types"]

        # Linear-attention (gated delta net) dims.
        self.linear_num_key_heads = text_cfg["linear_num_key_heads"]
        self.linear_num_value_heads = text_cfg["linear_num_value_heads"]
        self.linear_key_head_dim = text_cfg["linear_key_head_dim"]
        self.linear_value_head_dim = text_cfg["linear_value_head_dim"]

        # Cache open shard handles.
        self._shards: Dict[str, object] = {}

    def _shard(self, filename: str):
        if filename not in self._shards:
            self._shards[filename] = safe_open(
                os.path.join(self.model_dir, filename), framework="pt", device="cpu"
            )
        return self._shards[filename]

    def _get(self, hf_name: str) -> torch.Tensor:
        """Load a tensor by its full HF name from the correct shard."""
        filename = self.weight_map[hf_name]
        return self._shard(filename).get_tensor(hf_name)

    def _get_layer(self, bid: int, suffix: str) -> torch.Tensor:
        return self._get(f"{self.HF_PREFIX}.layers.{bid}.{suffix}")

    # ------------------------------------------------------- qkvz / ba splitter

    def _split_qkvz(
        self, mixed_qkvz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split fused ``in_proj_qkvz`` weight into (query, key, value, z).

        Weight shape is ``[proj_out, in_features]``; the reference snippet splits
        activations on the last dim, so here we split dim-0 (the projected dim).
        """
        nk = self.linear_num_key_heads
        nv = self.linear_num_value_heads
        hk = self.linear_key_head_dim
        hv = self.linear_value_head_dim
        in_features = mixed_qkvz.shape[-1]

        # [nk, (2*hk + 2*hv*nv/nk), in]
        w = mixed_qkvz.view(nk, 2 * hk + 2 * hv * nv // nk, in_features)
        split = [hk, hk, (nv // nk) * hv, (nv // nk) * hv]
        query, key, value, z = torch.split(w, split, dim=1)

        query = query.reshape(-1, in_features).contiguous()   # [nk*hk, in]
        key = key.reshape(-1, in_features).contiguous()       # [nk*hk, in]
        value = value.reshape(-1, in_features).contiguous()   # [nv*hv, in]
        z = z.reshape(-1, in_features).contiguous()           # [nv*hv, in]
        return query, key, value, z

    def _split_ba(
        self, mixed_ba: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split fused ``in_proj_ba`` weight into (b, a), each ``[nv, in]``."""
        nk = self.linear_num_key_heads
        nv = self.linear_num_value_heads
        in_features = mixed_ba.shape[-1]

        w = mixed_ba.view(nk, 2 * nv // nk, in_features)
        b, a = torch.split(w, [nv // nk, nv // nk], dim=1)
        b = b.reshape(nv, in_features).contiguous()
        a = a.reshape(nv, in_features).contiguous()
        return b, a

    # ---------------------------------------------------------------- quantize

    def _quantize(
        self, weight: torch.Tensor, tensor_type: GGMLQuantizationType
    ) -> torch.Tensor:
        """Quantize + Q4NX-pack a 2D weight, or cast unquantized tensors."""
        if tensor_type == GGMLQuantizationType.BF16:
            return weight.contiguous().to(torch.bfloat16)
        if tensor_type == GGMLQuantizationType.F16:
            return weight.contiguous().to(torch.float16)
        if tensor_type == GGMLQuantizationType.F32:
            return weight.contiguous().to(torch.float32)

        assert weight.dim() == 2, f"quantized weight must be 2D, got {weight.shape}"
        cols = weight.shape[1]
        w_np = weight.to(torch.float32).contiguous().numpy()
        q = quantize(w_np, tensor_type)

        if tensor_type == GGMLQuantizationType.Q4_1:
            d, m, qw = GGUFTensor.unpack_q4_1(q, cols)
        elif tensor_type == GGMLQuantizationType.Q4_0:
            d, m, qw = GGUFTensor.unpack_q4_0(q, cols)
        elif tensor_type == GGMLQuantizationType.Q8_0:
            d, m, qw = GGUFTensor.unpack_q8_0(q, cols)
            m = None
        else:
            raise ValueError(f"Unsupported tensor type: {tensor_type}")

        # Row-block padding: the Q4NX packers require a multiple of row_block_size
        # rows. Small SSM projections (e.g. alpha/beta with 16 rows) are duplicated
        # up to a full block, matching the reference GGUF converter's behaviour
        # (the kernel only reads the live rows).
        if d.shape[0] < self.row_block_size:
            reps = self.row_block_size // d.shape[0]
            d = d.repeat(reps, 1).contiguous()
            qw = qw.repeat(reps, 1).contiguous()
            if m is not None:
                m = m.repeat(reps, 1).contiguous()

        if tensor_type == GGMLQuantizationType.Q8_0:
            return _Q4NXBase._pack_q8nx(self, d, None, qw)
        return _Q4NXBase._pack_q4nx(self, d, m, qw)

    def _emit(self, config_key: str, weight: torch.Tensor, bid: int | None = None):
        name = self._q4nx_name(config_key, bid)
        ttype = self._tensor_type(config_key)
        self.q4nx_tensors[name] = self._quantize(weight, ttype)
        print(f"  {name:60s} {ttype.name}")

    def _emit_bf16(self, name: str, weight: torch.Tensor):
        """Emit a raw row-major bf16 tensor under an explicit name (no packing)."""
        self.q4nx_tensors[name] = weight.contiguous().to(torch.bfloat16)
        print(f"  {name:60s} BF16 (raw)")

    def _emit_norm(self, config_key: str, weight: torch.Tensor, bid: int | None = None):
        """Emit an RMSNorm weight, adding 1.0 (HF stores it zero-centered)."""
        self._emit(config_key, (weight.to(torch.float32) + 1.0).contiguous(), bid)

    # ------------------------------------------------------------------ layers

    def _reorder_q_gate(self, q_proj: torch.Tensor) -> torch.Tensor:
        """Regroup fused q/gate rows from per-head interleaved to q-block then gate-block.

        HF gated attention emits ``q_proj`` as ``[num_heads * 2 * head_dim, in]`` laid out
        per head as ``(head, {query,gate}, head_dim)``. The NPU runs q_proj and g_proj as
        two matmuls over contiguous row slices, so it needs all query rows first then all
        gate rows: ``(g p h) c -> (p g h) c`` with ``p=2`` (query/gate), ``h=head_dim``.
        """
        g = self.num_attention_heads
        h = self.head_dim
        c = q_proj.shape[-1]
        return (
            q_proj.view(g, 2, h, c).permute(1, 0, 2, 3).reshape(2 * g * h, c).contiguous()
        )

    def _convert_full_attention_layer(self, bid: int):
        print(f"[layer {bid}] full_attention")
        q_proj = self._reorder_q_gate(self._get_layer(bid, "self_attn.q_proj.weight"))
        self._emit("q_proj", q_proj, bid)
        self._emit_norm("q_norm", self._get_layer(bid, "self_attn.q_norm.weight"), bid)
        self._emit("k_proj", self._get_layer(bid, "self_attn.k_proj.weight"), bid)
        self._emit_norm("k_norm", self._get_layer(bid, "self_attn.k_norm.weight"), bid)
        self._emit("v_proj", self._get_layer(bid, "self_attn.v_proj.weight"), bid)
        self._emit("o_proj", self._get_layer(bid, "self_attn.o_proj.weight"), bid)
        self._convert_common(bid)

    def _convert_linear_attention_layer(self, bid: int):
        print(f"[layer {bid}] linear_attention")

        # Split the fused SSM input projections (HF fix_query_key_value_ordering).
        qkvz = self._get_layer(bid, "linear_attn.in_proj_qkvz.weight")
        ba = self._get_layer(bid, "linear_attn.in_proj_ba.weight")
        query, key, value, z = self._split_qkvz(qkvz)
        b, a = self._split_ba(ba)

        # qkv fused for Q4NX (Q8_0); z is the output gate (Q4_1).
        qkv = torch.cat([query, key, value], dim=0).contiguous()
        self._emit("linear_qkv_proj", qkv, bid)
        self._emit("attn_gate_proj", z, bid)
        self._emit("linear_beta_proj", b, bid)
        self._emit("linear_alpha_proj", a, bid)

        # Plain row-major bf16 copies of alpha/beta (no packing, no reorder).
        self._emit_bf16(f"model.layers.{bid}.linear_attn.ssm_alpha_proj.bf16.weight", a)
        self._emit_bf16(f"model.layers.{bid}.linear_attn.ssm_beta_proj.bf16.weight", b)

        # SSM scalars / conv / norm / out.
        # HF stores A_log; Q4NX ssm_a expects a = -exp(A_log) (kept in f32).
        a_log = self._get_layer(bid, "linear_attn.A_log").to(torch.float32)
        self._emit("linear_a", (-torch.exp(a_log)).contiguous(), bid)
        self._emit("linear_dt", self._get_layer(bid, "linear_attn.dt_bias"), bid)

        # conv1d: HF [channels, 1, kernel] -> squeeze -> transpose to [kernel, channels].
        conv = self._get_layer(bid, "linear_attn.conv1d.weight").squeeze(1)
        self._emit("linear_conv", conv.t().contiguous(), bid)

        self._emit("linear_norm", self._get_layer(bid, "linear_attn.norm.weight"), bid)
        self._emit(
            "linear_out_proj", self._get_layer(bid, "linear_attn.out_proj.weight"), bid
        )
        self._convert_common(bid)

    def _convert_common(self, bid: int):
        """Layer norms + dense MLP shared by both layer kinds."""
        self._emit_norm(
            "pre_attn_norm", self._get_layer(bid, "input_layernorm.weight"), bid
        )
        self._emit_norm(
            "post_attn_norm",
            self._get_layer(bid, "post_attention_layernorm.weight"),
            bid,
        )
        self._emit("gate_proj", self._get_layer(bid, "mlp.gate_proj.weight"), bid)
        self._emit("up_proj", self._get_layer(bid, "mlp.up_proj.weight"), bid)
        self._emit("down_proj", self._get_layer(bid, "mlp.down_proj.weight"), bid)

    # ----------------------------------------------------------------- convert

    def convert(self):
        # Embedding (bf16) and tied lm_head (quantized copy of the embedding).
        embed = self._get(f"{self.HF_PREFIX}.embed_tokens.weight")
        self._emit("embedding", embed, None)
        if self.tie_word_embeddings:
            print("[INFO] tied embeddings -> lm_head is a quantized embedding copy")
            self._emit("lm_head", embed, None)
        else:
            self._emit("lm_head", self._get("lm_head.weight"), None)

        for bid in range(self.num_layers):
            if self.layer_types[bid] == "full_attention":
                self._convert_full_attention_layer(bid)
            else:
                self._convert_linear_attention_layer(bid)

        self._emit_norm("model_final_norm", self._get(f"{self.HF_PREFIX}.norm.weight"), None)

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "model.q4nx")
        print(f"[INFO] Saving {len(self.q4nx_tensors)} tensors to {out_path}")
        save_file(self.q4nx_tensors, out_path)
        print("[INFO] Done.")


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    quantizer = Qwen35OmniThinkerQuantizer(
        model_dir=HERE,
        config_path=os.path.join(HERE, "..", "configs", "qwen3.5_omni.json"),
        output_dir=os.path.join(HERE, "..", "q4nx_files", "qwen3.5-omni-thinker"),
    )
    quantizer.convert()
