# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Converter for the DFlash speculative-decoding draft (llama.cpp GGUF, general.architecture = "dflash").

Matmul weights come out as Q4_1 q4nx blocks (the NPU format the NPU drafter will also read) except the tensors listed in
configs/dflash.json -> dflash_config.bf16_tensors (substrings of the q4nx tensor name, e.g. "model.fc.weight", "mlp.down_proj"), which stay BF16 in the 32-row block layout the
bf16_proj.h cores read (same layout as Gemma4's inp_gate / proj tensors). Norms are BF16 vectors.
The draft has no embedding and no lm_head: it shares the target's.
"""
import numpy as np
import torch
from einops import rearrange
from gguf import GGMLQuantizationType, GGUFReader, quantize

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch


class DFlash(__Q4NX_Converter, model_arch=ModelArch.DFLASH):
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()
        self.bf16_tensors = set(self.q4nx_config.get("dflash_config", {}).get("bf16_tensors", []))

    def _meta(self, key, default=None):
        f = self.gguf_reader.fields.get(key)
        return f.contents() if f is not None else default

    @staticmethod
    def _block_layout_bf16(w: torch.Tensor, row_block_size: int = 32) -> torch.Tensor:
        """[N, K] -> [N/32, K, 32] (each 32-row block column-major), bf16."""
        n, k = w.shape
        assert n % row_block_size == 0
        w = rearrange(w.contiguous(), '(b r) k -> b r k', r=row_block_size)
        return rearrange(w, 'b r k -> b k r').contiguous().to(torch.bfloat16)

    def _q4_1_pack(self, tensor) -> torch.Tensor:
        # BF16 source: quantize explicitly (unpack() would hand the BF16 matrix back unpacked).
        w = tensor.unpack(GGMLQuantizationType.BF16)[0].to(torch.float32).numpy()
        data = quantize(w, GGMLQuantizationType.Q4_1).copy()
        d, m, qw = tensor.unpack_q4_1(data, tensor.shape[0])
        return self._pack_q4nx(d, m, qw)

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        if weights_type != 'language':
            raise ValueError("The DFlash draft has no vision/audio weights")
        self.q4nx_tensors = {}
        missing = [n for n in self.gguf_tensors if n not in self.forward_name_map]
        assert not missing, f"unmapped GGUF tensors: {missing}"
        for name, t in self.gguf_tensors.items():
            qname = self.forward_name_map[name]
            want = self.tensor_q4nx_type_map[name]
            if len(t.shape) == 1:                                   # norm vectors
                self.q4nx_tensors[qname] = t.unpack(GGMLQuantizationType.F32)[0].to(torch.bfloat16)
            elif want == GGMLQuantizationType.BF16 or any(p in qname for p in self.bf16_tensors):
                w = t.unpack(GGMLQuantizationType.BF16)[0].to(torch.bfloat16)
                self.q4nx_tensors[qname] = self._block_layout_bf16(w)
            else:
                assert want == GGMLQuantizationType.Q4_1, f"{name}: only Q4_1 / BF16 are supported"
                self.q4nx_tensors[qname] = self._q4_1_pack(t)
        i32 = lambda v: torch.tensor(v, dtype=torch.int32)
        f32 = lambda v: torch.tensor(v, dtype=torch.float32)
        tl = [int(x) - 1 for x in self._meta("dflash.target_layers")]   # GGUF stores layer-input indices (+1)
        self.q4nx_tensors["dflash.block_size"] = i32([self._meta("dflash.block_size")])
        self.q4nx_tensors["dflash.target_layer_ids"] = i32(tl)         # HF convention: layer outputs 5,11,17,23,35
        self.q4nx_tensors["dflash.embedding_scale"] = f32([self._meta("dflash.embedding_scale")])
        self.q4nx_tensors["dflash.rope_theta"] = f32([self._meta("dflash.rope.freq_base")])
        self.q4nx_tensors["dflash.rms_eps"] = f32([self._meta("dflash.attention.layer_norm_rms_epsilon")])
        self.q4nx_tensors["dflash.sliding_window"] = i32([self._meta("dflash.attention.sliding_window")])
        self._export_q4nx_tensors(q4nx_path, filename="dflash_weight.q4nx")
