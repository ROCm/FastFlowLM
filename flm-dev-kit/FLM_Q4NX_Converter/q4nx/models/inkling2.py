# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from ..gguf_tensor import GGUFTensor

from .qwen38_dflash2 import _load_ggml, GGML_TYPE_Q4_K, Q4_K_BLOCK, Q4_K_BYTES

from gguf import GGUFReader, GGMLQuantizationType, dequantize
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


class Inkling2Loop(__Q4NX_Converter, model_arch=ModelArch.INKLING2_LOOP):
    """Converter for the `inkling2_loop` GGUF architecture (thinkingmachines Inkling-2-Mini).

    The GGUF comes from llama.cpp's convert_hf_to_gguf.py, which already
    dequantizes the NVFP4 / MXFP8 experts of the HF checkpoint, then from
    llama-quantize with the recipe in configs/inkling2_loop.json (see the
    per-tensor types there).  This converter does not re-quantize: every
    quantized tensor must arrive in the type the config asks for, except
    the expert down (below), which arrives BF16 and is quantized here.

    Layout notes:

    * 24 weight layers; layers 6..17 run three times per token with the same
      weights (the loop), so there is one set of weights per layer here.
    * Attention: q [3072, 2304], k/v [1024, 2304], o [2304, 3072], an output
      gate [3072, 2304] and the relative-position query rel [384, 2304]
      (24 heads x 16).  No RoPE.  The tiny projections (sinks [24, 2304],
      dynamic conv taps [32, 2304] on the global layers) are BF16 for the host.
    * MoE: 128 experts.  gate_up is [128, 2304, 2304] with the 1152 gate rows
      first, then the 1152 up rows (llama.cpp de-interleaves HF's w13).
      down is [128, 2304, 1152]; 1152 is not a multiple of 256, so GGUF has
      no Q4_K form of it.  The recipe keeps it BF16 and the converter zero-pads
      each row to 1280 (the engine's FF_PAD) and quantizes it with ggml's own
      Q4_K quantizer (ggml_quantize_chunk, the call llama-quantize --pure
      makes; libggml-base.so, set GGML_BASE_LIB).  Groups of 32 never straddle
      the pad, so the pad groups are exact zeros and the real ones are quantized
      as llama.cpp would.  INKLING2_IMATRIX=<file> (a llama.cpp GGUF importance
      matrix, e.g. the one used for the recipe's other tensors) weights the
      quantization per expert as llama-quantize --imatrix does; the 128 pad
      columns weigh 0.  Without it the quantization is unweighted.
    * Router, expert bias, convolution taps, block-residual queries, norms
      and the loop controllers stay F32: transformers runs them in fp32, and
      the router's sigmoid top-8 picks are decided by near-ties.

    Storage per tensor kind:

        token_embd                     -> BF16 (lossless, already BF16)
        projections and experts        -> per config (Q4_K / Q4_1 / Q8_0 blocks)
        lm_head [201024, 2304]         -> Q8_0 (q8nx blocks)
        sinks, dynamic conv projections -> BF16
        everything else                -> F32 as in the GGUF
    """

    _QUANTIZED = (GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q4_1, GGMLQuantizationType.Q8_0)

    def __init__(self, gguf_reader: GGUFReader):
        print("[INFO] Using Inkling2Loop converter")
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()

    def _require_type(self, gguf_tensor: GGUFTensor, expected: GGMLQuantizationType):
        # The NPU kernels read one block format per tensor kind: a GGUF from a
        # different recipe must fail here, not turn into a silent re-quantization.
        if gguf_tensor.tensor_type != expected:
            raise ValueError(f"{gguf_tensor.name} is {gguf_tensor.tensor_type.name}, "
                             f"configs/inkling2_loop.json expects {expected.name}")

    def _pack_quantized(self, gguf_tensor: GGUFTensor, target: GGMLQuantizationType) -> torch.Tensor:
        self._require_type(gguf_tensor, target)
        # A 3D expert tensor [n_expert, rows, cols] unpacks as (n_expert * rows, cols); with
        # rows a multiple of row_block_size the packed experts are simply stacked.
        rows = int(gguf_tensor.shape[1])
        assert rows % self.row_block_size == 0, \
            f"{gguf_tensor.name}: {rows} rows do not fill whole {self.row_block_size}-row blocks"
        d, m, q = gguf_tensor.unpack(target)
        return self._pack(d, m, q, tensor_type=target)

    def _expert_imatrix(self, name: str, cols: int, padded: int):
        """INKLING2_IMATRIX: [n_expert, padded] float32 importance of this expert-down tensor (in_sum2 / counts per expert,
        as llama-quantize reads it; the pad columns 0), or None if no imatrix is given."""
        path = os.environ.get("INKLING2_IMATRIX")
        if not path:
            return None
        if not hasattr(self, "imatrix"):
            print(f"[INFO] importance matrix {path}")
            self.imatrix = {t.name: t for t in GGUFReader(path).tensors}
        try:
            sums, counts = self.imatrix[name + ".in_sum2"], self.imatrix[name + ".counts"]
        except KeyError:
            raise ValueError(f"INKLING2_IMATRIX {path} has no entry for {name}") from None
        s = np.asarray(sums.data, dtype=np.float32)
        c = np.asarray(counts.data, dtype=np.float32).reshape(-1, 1)
        if s.ndim != 2 or s.shape[1] != cols or c.shape[0] != s.shape[0] or (c <= 0).any():
            raise ValueError(f"{name}: imatrix shapes {s.shape} / {c.shape} do not fit {cols} columns "
                             f"(every expert needs a positive count)")
        imp = np.zeros((s.shape[0], padded), np.float32)
        imp[:, :cols] = s / c
        return np.ascontiguousarray(imp)

    def _pack_padded_q4k(self, gguf_tensor: GGUFTensor) -> torch.Tensor:
        if gguf_tensor.tensor_type != GGMLQuantizationType.BF16:
            raise ValueError(f"{gguf_tensor.name} is {gguf_tensor.tensor_type.name}; the recipe must keep "
                             f"ffn_down_exps BF16 (it is padded and quantized to Q4_K here)")
        if not hasattr(self, "ggml"):
            self.ggml = _load_ggml(None)
        cols = int(gguf_tensor.shape[0])
        padded = -(-cols // Q4_K_BLOCK) * Q4_K_BLOCK
        w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type).reshape(-1, cols)
        src = np.zeros((w.shape[0], padded), np.float32)
        src[:, :cols] = w
        raw = np.empty(src.size // Q4_K_BLOCK * Q4_K_BYTES, dtype=np.uint8)
        imp = self._expert_imatrix(gguf_tensor.name, cols, padded)
        if imp is None:
            got = self.ggml.ggml_quantize_chunk(GGML_TYPE_Q4_K, src.ctypes.data, raw.ctypes.data, 0,
                                                src.shape[0], padded, None)
        else:
            # one call per expert: ggml applies one importance vector to all rows of a call
            ne = imp.shape[0]
            rows = src.shape[0] // ne
            with ThreadPoolExecutor(os.cpu_count() or 1) as ex:
                parts = list(ex.map(lambda e: self.ggml.ggml_quantize_chunk(
                    GGML_TYPE_Q4_K, src.ctypes.data, raw.ctypes.data, e * rows * padded, rows, padded,
                    imp[e].ctypes.data), range(ne)))
            got = sum(parts)
        assert got == raw.size, (got, raw.size)
        t, u, q = GGUFTensor.unpack_q4_k(raw, padded)
        return self._pack(t, u, q, tensor_type=GGMLQuantizationType.Q4_K)

    #: convert.py --verify-hf compares the written vision file with the HF checkpoint using this script
    vision_hf_verifier = "verify_inkling2_vision_q4nx.py"

    def _convert_vision(self, q4nx_path: str):
        """Image tower (mmproj GGUF from llama.cpp's convert_hf_to_gguf.py --mmproj) -> vision_weight.q4nx.

        Every tensor is stored BF16 (the checkpoint's own dtype; llama.cpp widens the small ones to F32,
        which is exact).  The matmul weights are laid out for the vision MM kernel (64-row x 512-column
        tiles, see vision_mm_weight_rearrange), zero-padded up to whole 512 x 512 tiles: the patch
        embedding [1024, 1200] gets K 1536, proj_out [2304, 4096] gets N 2560.  q, k and v are stacked
        into one [3072, 1024] qkv weight (and bias), the order the attention kernel splits them in.
        The patch embedding comes back in the checkpoint's [py, px, c] patch order (llama.cpp stores it
        as a conv kernel [c, py, px]).
        """
        if not self.vision_MM_K or not self.vision_MM_N:
            raise ValueError("configs/inkling2_loop.json has no vision_config")
        # vision_source_policy "error": the tower ships as bf16, so a source in any other dtype is another export
        self.vision_source_policy = "error"
        self._check_vision_source_is_bf16(strict_f32=True)

        t = {}
        for gguf_tensor in self.gguf_tensors.values():
            name = self.forward_name_map.get(gguf_tensor.name)
            if name is None:
                raise RuntimeError(f"vision GGUF tensor {gguf_tensor.name} has no mapping")
            w = torch.from_numpy(dequantize(gguf_tensor.data, gguf_tensor.tensor_type)).to(torch.bfloat16).contiguous()
            if name.endswith("patch_embedding.weight"):
                assert w.shape == (1024, 3, 20, 20), w.shape
                w = w.permute(0, 2, 3, 1).reshape(1024, 1200).contiguous()
            t[name] = w

        B = "model.visual.backbone."
        n_layers = sum(1 for n in t if n.endswith(".self_attn.q_proj.weight"))
        for i in range(n_layers):
            a = f"{B}layers.{i}.self_attn."
            for kind in ("weight", "bias"):
                t[f"{a}qkv_proj.{kind}"] = torch.cat([t.pop(f"{a}{x}_proj.{kind}") for x in "qkv"], dim=0).contiguous()
        mm = [n for n in t if n.endswith(("qkv_proj.weight", "out_proj.weight", "c_fc.weight", "c_proj.weight"))]
        mm += [B + "embeddings.patch_embedding.weight", "model.visual.proj_in.weight", "model.visual.proj_out.weight"]
        for n in mm:
            t[n] = self._multi_modal_mm_weight_rearrange(t[n], self.vision_MM_K, self.vision_MM_N)

        print(f"[INFO] vision: {n_layers} layers, {len(t)} tensors")
        self.q4nx_tensors = t
        self._export_q4nx_tensors(q4nx_path, filename="vision_weight.q4nx")

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        if weights_type == 'vision':
            return self._convert_vision(q4nx_path)
        if weights_type != 'language':
            raise ValueError(f"Unsupported weights_type: {weights_type} for inkling2_loop")
        if not self._has_lm_head():
            raise ValueError("inkling2_loop GGUF has no output.weight; Inkling-2 does not tie embeddings")

        self.q4nx_tensors = {}
        skipped = []
        for gguf_tensor in self.gguf_tensors.values():
            if gguf_tensor.name not in self.forward_name_map:
                skipped.append(gguf_tensor.name)
                continue

            q4nx_name = self.forward_name_map[gguf_tensor.name]
            target = self.tensor_q4nx_type_map[gguf_tensor.name]
            print(f"Processing tensor: {gguf_tensor.name} with type "
                  f"{gguf_tensor.tensor_type.name} -> {q4nx_name} with dtype {target.name}")

            if gguf_tensor.name.endswith("ffn_down_exps.weight"):
                packed = self._pack_padded_q4k(gguf_tensor)
            elif target in self._QUANTIZED:
                packed = self._pack_quantized(gguf_tensor, target)
            else:
                w = torch.from_numpy(dequantize(gguf_tensor.data, gguf_tensor.tensor_type))
                dtype = torch.float32 if target == GGMLQuantizationType.F32 else torch.bfloat16
                packed = w.to(dtype).contiguous()

            self.q4nx_tensors[q4nx_name] = packed

        # Every GGUF tensor is used by the forward pass: an unmapped one is a config bug.
        if skipped:
            raise RuntimeError(f"{len(skipped)} GGUF tensors have no mapping, e.g. {skipped[:5]}")

        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
