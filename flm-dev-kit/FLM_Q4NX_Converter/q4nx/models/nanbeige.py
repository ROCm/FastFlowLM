# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, quantize
from safetensors.torch import save_file
import torch
from gguf import dequantize
from einops import rearrange

class Nanbeige(__Q4NX_Converter, model_arch=ModelArch.NANBEIGE):
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        print("[INFO] Converting nanbeige model to Q4NX format...")
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

            if "ffn_down.weight" in gguf_tensor.name: # special padding to multiple of 512
                d, m, q = unpacked
                din = q.shape[1]
                din_pad = (din + 511) // 512 * 512
                print(f"Padding ffn_down from {din} to {din_pad}")
                din_dm = din_pad // 32
                d_pad = torch.zeros((d.shape[0], din_dm), dtype=d.dtype)
                m_pad = torch.zeros((m.shape[0], din_dm), dtype=m.dtype)
                q_pad = torch.zeros((q.shape[0], din_pad), dtype=q.dtype)
                d_pad[:, :d.shape[1]] = d
                m_pad[:, :m.shape[1]] = m
                q_pad[:, :q.shape[1]] = q
                unpacked = (d_pad, m_pad, q_pad)
        
            if "ffn_up.weight" in gguf_tensor.name or "ffn_gate.weight" in gguf_tensor.name: # special padding to multiple of 512
                d, m, q = unpacked
                dout = q.shape[0]
                dout_pad = (dout + 511) // 512 * 512
                print(f"Padding ffn_up/gate from {dout} to {dout_pad}")
                d_pad = torch.zeros((dout_pad, d.shape[1]), dtype=d.dtype)
                m_pad = torch.zeros((dout_pad, m.shape[1]), dtype=m.dtype)
                q_pad = torch.zeros((dout_pad, q.shape[1]), dtype=q.dtype)
                d_pad[:d.shape[0], :] = d
                m_pad[:m.shape[0], :] = m
                q_pad[:q.shape[0], :] = q
                unpacked = (d_pad, m_pad, q_pad)
                    
            self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_q4nx(*unpacked)

        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
