# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Qwen3.5-27B (and fine-tunes such as vllm-sr/d3-turbo) -> q4nx.

`Qwen35.convert` hard-codes a value/key head ratio of 2 (`q = 2`, `chunk(2)`) which
holds for Qwen3.5-4B/9B (32 or 32 v heads over 16 qk groups) but not for the 27B:
48 value heads over 16 key groups is a ratio of 3, and the `attn_qkv` / `ssm_conv1d`
rows no longer split down the middle (4096 q+k rows, 6144 v rows).  Here the ratio
and the split point are read out of the GGUF (same approach as qwen38.py).

Differences from qwen38.py, on purpose:
  * an untied checkpoint without an `output.weight` is treated as tied, like
    qwen35.py: the Q8_0 token embedding is written as `lm_head.weight` and there
    is no bf16 `model.embed_tokens.weight` (d3_npu reads `lm_head.weight`).
  * no MTP block, vision tower is the blocked Qwen3.5 layout of qwen35.py
    (`-t vision` is delegated to `Qwen35.convert`).

The engine's alpha/beta descriptor is {D, ceil(Hv / 32) * 32}: 48 value heads are
zero-padded to 64 rows (the padded rows decode to zero; the runtime only reads the
first Hv).  The bf16 prefill copies (`ssm_{alpha,beta}_proj.bf16.weight`, {D, Hv}) are
not written here: they come exactly from the checkpoint's in_proj_a/in_proj_b, whose
HF head order already is the grouped (g q) order of the reordered tensors.
"""

import re

from einops import rearrange
from gguf import GGUFReader, GGMLQuantizationType, dequantize
import torch
import torch.nn.functional as F

from ..constants import ModelArch
from .qwen35 import Qwen35


class Qwen35_27B(Qwen35, model_arch=ModelArch.QWEN35_27B):
    vision_source_policy = "warn"   # vision conversion not checked on a real BF16 mmproj yet
    ARCH = "qwen35"
    GGML_GROUP_SIZE = 32

    def __init__(self, gguf_reader: GGUFReader):
        print("[INFO] Using Qwen35_27B converter")
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        # the mmproj GGUF is a `clip` file without any qwen35.* field
        super().initialize()
        self.is_mmproj = "clip.has_vision_encoder" in self.gguf_reader.fields
        if self.is_mmproj:
            print("[INFO] mmproj GGUF: vision tower only")
        else:
            self._read_geometry()

    # ------------------------------------------------------------------ #
    def _field(self, suffix):
        field = self.gguf_reader.fields.get(f"{self.ARCH}.{suffix}")
        if field is None:
            raise KeyError(f"GGUF is missing {self.ARCH}.{suffix}")
        return field.contents()

    def _read_geometry(self):
        self.attn_head_dim = self._field("attention.value_length")
        self.full_attention_interval = self._field("full_attention_interval")
        self.l_head_dim = self._field("ssm.state_size")
        self.l_qk_heads = self._field("ssm.group_count")
        self.l_v_heads = self._field("ssm.inner_size") // self.l_head_dim
        assert self.l_v_heads % self.l_qk_heads == 0, (
            f"{self.l_v_heads} v heads do not divide into {self.l_qk_heads} qk groups")
        self.l_expand = self.l_v_heads // self.l_qk_heads
        # rows of attn_qkv / ssm_conv1d that belong to q and k (v is the rest)
        self.l_qk_rows = 2 * self.l_qk_heads * self.l_head_dim
        print(f"[INFO] delta-net: {self.l_qk_heads} qk groups x {self.l_expand} = "
              f"{self.l_v_heads} v heads of {self.l_head_dim}, qk rows {self.l_qk_rows}")

    # ------------------------------------------------------------------ #
    # reorders.  d / m / qw share the row axis, so a row permutation applies to each
    # unchanged; column permutations differ (d, m: one entry per 32 columns).
    # With expand == 1 every reorder below is the identity.
    # ------------------------------------------------------------------ #
    def _v_rows(self, parts):
        """(expand groups head_dim) rows -> (groups expand head_dim)."""
        return [rearrange(t, "(q g p) c -> (g q p) c", q=self.l_expand, p=self.l_head_dim).contiguous()
                for t in parts]

    def _reorder_qkv(self, parts):
        out = []
        for t in parts:
            qk, v = t[:self.l_qk_rows], t[self.l_qk_rows:]
            out.append(torch.cat([qk, self._v_rows([v])[0]], dim=0).contiguous())
        return out

    def _reorder_out_proj_cols(self, d, m, qw):
        p_scale = self.l_head_dim // self.GGML_GROUP_SIZE
        d = rearrange(d, "r (q g p) -> r (g q p)", q=self.l_expand, p=p_scale).contiguous()
        m = rearrange(m, "r (q g p) -> r (g q p)", q=self.l_expand, p=p_scale).contiguous()
        qw = rearrange(qw, "r (q g p) -> r (g q p)", q=self.l_expand, p=self.l_head_dim).contiguous()
        return [d, m, qw]

    def _reorder_per_head_rows(self, parts):
        return [rearrange(t, "(q g) c -> (g q) c", q=self.l_expand).contiguous() for t in parts]

    def _reorder_per_head_vector(self, t):
        return rearrange(t, "(q g) -> (g q)", q=self.l_expand).contiguous()

    def _pad_rows_to_multiple(self, parts, multiple=32):
        """Zero-pad rows to the engine descriptor's ceil(Hv / 32) * 32 (48 -> 64)."""
        rows = parts[0].shape[0]
        pad = (-rows) % multiple
        if pad == 0:
            return parts
        print(f"[INFO] Padding {rows} rows -> {rows + pad}")
        return [F.pad(t, (0, 0, 0, pad), "constant", 0).contiguous() for t in parts]

    # ------------------------------------------------------------------ #
    def convert(self, q4nx_path: str, weights_type: str = "language"):
        if weights_type == "vision":
            if not self.is_mmproj:
                raise ValueError("-t vision needs the mmproj GGUF")
            return super().convert(q4nx_path, weights_type)
        if weights_type != "language":
            raise ValueError(f"Unsupported weights_type: {weights_type} for Qwen3.5-27B")
        if self.is_mmproj:
            raise ValueError("language conversion was given an mmproj GGUF")

        self.q4nx_tensors = {}
        self._drop_mtp_tensors()
        tied_embedding = not self._has_lm_head()
        if tied_embedding:
            print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
            unpacked = self.gguf_tensors["token_embd.weight"].unpack(GGMLQuantizationType.Q8_0)
            target_dtype = self.gguf_tensors["token_embd.weight"].get_used_quantization_type(GGMLQuantizationType.Q8_0)
            self.q4nx_tensors["lm_head.weight"] = self._pack(*unpacked, tensor_type=target_dtype)

        for gguf_tensor in self.gguf_tensors.values():
            name = gguf_tensor.name
            if name not in self.forward_name_map:
                print(f"[WARN] No name mapping for {name}, skipping")
                continue
            new_name = self.forward_name_map[name]
            requested = self.tensor_q4nx_type_map[name]
            target_dtype = gguf_tensor.get_used_quantization_type(requested)
            print(f"Processing tensor: {name} with type {gguf_tensor.tensor_type.name} -> {new_name} with dtype {target_dtype.name}")

            if "token_embd.weight" in name:
                if tied_embedding:
                    print("[INFO] Tied embedding, skipping bf16 copy of token_embd.weight")
                    continue
                w = torch.from_numpy(dequantize(gguf_tensor.data, gguf_tensor.tensor_type)).contiguous().to(torch.bfloat16)
                self.q4nx_tensors[new_name] = w
                continue

            match = re.match(r"^blk\.(\d+)\.", name)
            layer_id = int(match.group(1)) if match else 0
            is_full = self._is_full_attention_layer(layer_id)
            unpacked = list(gguf_tensor.unpack(requested))

            if is_full:
                if "q_proj" in new_name:
                    print("[INFO] Seperate q, gate for q_proj")
                    unpacked = [rearrange(t, "(g p h) c -> (p g h) c", p=2, h=self.attn_head_dim).contiguous()
                                for t in unpacked]
            else:
                if "self_attn.gate_proj" in new_name:
                    print("[INFO] Reorder Gate")
                    unpacked = self._v_rows(unpacked)
                elif "qkv_proj" in new_name:
                    print("[INFO] Reorder v rows of qkv_proj")
                    unpacked = self._reorder_qkv(unpacked)
                elif "ssm_out_proj" in new_name:
                    print(f"[INFO] Reorder for {new_name}")
                    unpacked = self._reorder_out_proj_cols(*unpacked)
                elif ("ssm_alpha_proj" in new_name or "ssm_beta_proj" in new_name) and "bf16" not in new_name:
                    print(f"[INFO] Reorder for {new_name}")
                    unpacked = self._reorder_per_head_rows(unpacked)
                    unpacked = self._pad_rows_to_multiple(unpacked, 32)
                elif "ssm_conv1d" in new_name:
                    print("[INFO] transpose conv1d")
                    unpacked = [self._reorder_qkv([unpacked[0]])[0].T.contiguous()]
                elif name.endswith("ssm_a") or "ssm_dt" in name:
                    self.q4nx_tensors[new_name] = self._reorder_per_head_vector(
                        unpacked[0].to(torch.float32)).contiguous()
                    continue

            self.q4nx_tensors[new_name] = self._pack(*unpacked, tensor_type=target_dtype)

        self._extract_tokenizer_json(q4nx_path)
        self._export_q4nx_tensors(q4nx_path)
