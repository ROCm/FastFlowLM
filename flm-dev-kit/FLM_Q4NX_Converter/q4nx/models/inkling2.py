# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from ..gguf_tensor import GGUFTensor

from gguf import GGUFReader, GGMLQuantizationType, dequantize
import torch


class Inkling2Loop(__Q4NX_Converter, model_arch=ModelArch.INKLING2_LOOP):
    """Converter for the `inkling2_loop` GGUF architecture (thinkingmachines Inkling-2-Mini).

    The GGUF comes from llama.cpp's convert_hf_to_gguf.py, which already
    dequantizes the NVFP4 / MXFP8 experts of the HF checkpoint, then from
    llama-quantize with the recipe in configs/inkling2_loop.json (see the
    per-tensor types there).  This converter does not re-quantize: every
    quantized tensor must arrive in the type the config asks for.

    Layout notes:

    * 24 weight layers; layers 6..17 run three times per token with the same
      weights (the loop), so there is one set of weights per layer here.
    * Attention: q [3072, 2304], k/v [1024, 2304], o [2304, 3072], an output
      gate [3072, 2304] and the relative-position query rel [384, 2304]
      (24 heads x 16).  No RoPE.  The tiny projections (sinks [24, 2304],
      dynamic conv taps [32, 2304] on the global layers) are BF16 for the host.
    * MoE: 128 experts.  gate_up is [128, 2304, 2304] with the 1152 gate rows
      first, then the 1152 up rows (llama.cpp de-interleaves HF's w13).
      down is [128, 2304, 1152]; 1152 is not a multiple of 256, so it cannot
      be Q4_K and ships as Q4_1 (the q4nx block pads it to whole col blocks).
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

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        if weights_type != 'language':
            raise ValueError(f"Unsupported weights_type: {weights_type} for inkling2_loop (text only)")
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

            if target in self._QUANTIZED:
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
