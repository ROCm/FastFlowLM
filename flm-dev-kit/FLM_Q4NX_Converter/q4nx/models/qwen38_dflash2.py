# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""z-lab/Qwen3.8-27B-DFlash2 draft (HF safetensors) -> dflash_weight.q4nx.

The draft is a 5-layer Qwen3-style block-diffusion model with two-tap dynamic convolutions and a candidate selector
(see detail/qwen3_8mtp_npu/speculate/dflash2_draft.hpp). It shares the target's token embedding and lm_head, so the
package carries neither.

Q4_K is the format the engine's dequant_mm.xclbin streams, so the big matrices are quantized to it: llama.cpp's
quantize_row_q4_K_ref (via ggml's C API, ctypes) produces GGML Q4_K blocks and the existing unpack_q4_k / _pack_q4k
path turns them into the FLM q4_k chunk layout, as for the base model. Everything else stays BF16 in its HF layout:
  * norms and q/k head norms (vectors)
  * attention_conv / mlp_conv base_kernel [2, 2, 5120] and kernel_projection [1280, 5120] (the engine runs these
    on the host; 1280 is not a multiple of the 512-wide array tile)
  * candidate_selector hidden_projection [256, 5120] and the two codebooks [248320, 256] (a few rows are gathered
    per step, the table is never streamed)
Tensor names get a "model." prefix, like the base model's.
"""
import ctypes
import os

import numpy as np
import torch
from safetensors import safe_open

from ..model_converter import __Q4NX_Converter
from ..gguf_tensor import GGUFTensor
from ..constants import ModelArch

GGML_TYPE_Q4_K = 12
Q4_K_BLOCK, Q4_K_BYTES = 256, 144


def _load_ggml(path):
    cands = [path, os.environ.get("GGML_BASE_LIB"),
             "libggml-base.so"]
    last = None
    for c in cands:
        if not c:
            continue
        try:
            lib = ctypes.CDLL(c)
            lib.ggml_quantize_chunk.restype = ctypes.c_size_t
            lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                                ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
            return lib
        except OSError as e:
            last = e
    raise RuntimeError(f"cannot load libggml-base.so (set GGML_BASE_LIB): {last}")


class Qwen38DFlash2(__Q4NX_Converter, model_arch=ModelArch.QWEN38_DFLASH2):
    """Not driven by a GGUF: build with `Qwen38DFlash2(hf_dir)` and call `convert(out_dir)`."""

    def __init__(self, hf_dir, ggml_lib=None):
        self.hf_dir = hf_dir
        self.gguf_tensors = {}
        self.q4nx_tensors = {}
        self._load_config()
        self.ggml = _load_ggml(ggml_lib)

    def _create_name_maps(self):
        self.forward_name_map = {}
        self.backward_name_map = {}
        self.tensor_q4nx_type_map = {}

    @staticmethod
    def q4nx_name(hf_name):
        return "model." + hf_name

    @staticmethod
    def is_matmul(hf_name):
        return hf_name == "fc.weight" or hf_name.endswith(
            ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
             "gate_proj.weight", "up_proj.weight", "down_proj.weight"))

    def quantize_q4_k(self, w):
        """[N, K] bf16/fp32 -> GGML Q4_K bytes (N * K / 256 blocks of 144 B)."""
        n, k = w.shape
        assert k % Q4_K_BLOCK == 0
        src = np.ascontiguousarray(w.to(torch.float32).numpy())
        dst = np.empty(n * k // Q4_K_BLOCK * Q4_K_BYTES, dtype=np.uint8)
        got = self.ggml.ggml_quantize_chunk(GGML_TYPE_Q4_K, src.ctypes.data, dst.ctypes.data, 0, n, k, None)
        assert got == dst.size, (got, dst.size)
        return dst

    def pack_matmul(self, w):
        n, k = w.shape
        raw = self.quantize_q4_k(w)
        t, u, q = GGUFTensor.unpack_q4_k(raw, k)
        return self._pack_q4k(t, u, q)

    def convert(self, q4nx_path, weights_type="language", only=None):
        f = safe_open(os.path.join(self.hf_dir, "model.safetensors"), "pt")
        for name in sorted(f.keys()):
            if only and not only(name):
                continue
            w = f.get_tensor(name)
            qname = self.q4nx_name(name)
            if self.is_matmul(name):
                print(f"[Q4_K] {name} {tuple(w.shape)}", flush=True)
                self.q4nx_tensors[qname] = self.pack_matmul(w)
            else:
                print(f"[BF16] {name} {tuple(w.shape)}", flush=True)
                self.q4nx_tensors[qname] = w.to(torch.bfloat16).contiguous()
        self._export_q4nx_tensors(q4nx_path, filename="dflash_weight.q4nx")
