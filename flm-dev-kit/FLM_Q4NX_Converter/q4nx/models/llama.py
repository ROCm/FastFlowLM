# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, quantize
from safetensors.torch import save_file
import torch
from gguf import dequantize
from einops import rearrange

class Llama(__Q4NX_Converter, model_arch=ModelArch.LLAMA):
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}

        if not self._has_lm_head():
            print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
            unpacked = self.gguf_tensors["token_embd.weight"].unpack(self.default_tensor_type)
            self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)

        for key, gguf_tensor in self.gguf_tensors.items():
            if "token_embd.weight" in gguf_tensor.name: # this should be bf16
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                continue

            unpacked = gguf_tensor.unpack(self.default_tensor_type)

            if "q_proj" in self.forward_name_map[gguf_tensor.name] or "k_proj" in self.forward_name_map[gguf_tensor.name]: # for llama q_proj, the order is special
                # 0, 1, 2, .... 127
                # 0, 64, 1, ..., 127
                DH = self.gguf_reader.fields["llama.rope.dimension_count"].contents()
                pp = DH // 2
                d, m, qw = unpacked
                d = rearrange(d, '(g p q) c -> (g q p) c', p = pp, q = 2).contiguous()
                m = rearrange(m, '(g p q) c -> (g q p) c', p = pp, q = 2).contiguous()
                qw = rearrange(qw, '(g p q) c -> (g q p) c', p = pp, q = 2).contiguous()
                unpacked = (d, m, qw)

            self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_q4nx(*unpacked)

        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
