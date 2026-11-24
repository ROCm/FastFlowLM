# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from ..gguf_tensor import GGUFTensor
from ..utils import round_up_to_multiple

from gguf import GGUFReader, GGMLQuantizationType, dequantize, quantize
import numpy as np
import torch
import torch.nn.functional as F


class HunyuanDense(__Q4NX_Converter, model_arch=ModelArch.HUNYUAN_DENSE):
    """Converter for the `hunyuan-dense` GGUF architecture (e.g. Hy-MT2-1.8B).

    Layout is a plain GQA decoder with per-head q/k RMS norms, so the tensor
    surgery other families need (fused QKV splits, gate/up interleaving) does
    not apply here -- every projection goes through the packer untouched.

    In particular there is no q/k channel reorder of the kind `llama.py` does:
    `LLM_ARCH_HUNYUAN_DENSE` is a `LLAMA_ROPE_TYPE_NEOX` arch in llama.cpp
    (`llama_model_rope_type()` in `src/llama-model.cpp`) and its HF converter
    (`HunYuanModel.modify_tensors` in `conversion/hunyuan.py`) never calls
    `permute`, so the rows arrive in HF `rotate_half` order already -- which is
    what the kernels want.  See "RoPE channel order" in the README.

    Two things are specific to this family as shipped:

    * The checkpoint ties the word embeddings: there is no `output.weight`.
      `token_embd.weight` therefore does double duty -- it is dequantized from
      its GGUF type straight to bf16 for `model.embed_tokens.weight`, and separately
      packed for `lm_head.weight` (re-quantized only if the config asks for another type).  Both tensors are written, so the DLL
      never has to take its dequantize-the-lm_head fallback.
    * The vocabulary (120818) is not a multiple of `row_block_size`, so the
      lm_head rows are zero padded up to a whole number of blocks (the packers
      only lay out whole blocks).  The readers derive the block grid from the
      buffer size and skip the padded rows.  The bf16 embedding is *not*
      padded: it is a plain [vocab, dim] tensor.
    """

    #: GGUF types the packers read straight out of the file.
    _NATIVE_TYPES = (
        GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.Q4_1,
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.MXFP4,
    )

    _UNQUANTIZED_TYPES = (
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        GGMLQuantizationType.BF16,
    )

    def __init__(self, gguf_reader: GGUFReader):
        print("[INFO] Using HunyuanDense converter")
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()

    def _unpack_as(self, gguf_tensor: GGUFTensor, target: GGMLQuantizationType):
        """Unpack `gguf_tensor` into the planes of `target`, re-quantizing if needed.

        `GGUFTensor.unpack` keeps a tensor in whatever quantization the file
        already uses whenever that type is one the packers understand, which is
        the right default when the config only states a *fallback* type.  Here
        the config states the type the kernel will actually read, and the whole
        model has to land on one of them: this GGUF ships `ffn_down` as Q4_1
        while every other projection is Q4_0, and a q41_block_t sitting in a
        __Q4_0__ layer is a silently misread weight.  So a native type that
        disagrees with the target is round tripped through fp32.
        """
        if target in self._UNQUANTIZED_TYPES:
            # Norms: unpack already dequantizes, and quantize() has no encoder
            # for an unquantized target.
            return gguf_tensor.unpack(target)

        if gguf_tensor.tensor_type == target:
            return gguf_tensor.unpack(target)

        if gguf_tensor.tensor_type not in self._NATIVE_TYPES:
            # unpack()'s own dequantize -> quantize path already lands on target.
            return gguf_tensor.unpack(target)

        print(f"[INFO] Re-quantizing {gguf_tensor.name} from "
              f"{gguf_tensor.tensor_type.name} to {target.name}")
        w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
        w = np.ascontiguousarray(w.astype(np.float32))
        qdata = quantize(w, target).copy()

        columns = int(gguf_tensor.shape[0])
        if target == GGMLQuantizationType.Q4_0:
            return GGUFTensor.unpack_q4_0(qdata, columns)
        if target == GGMLQuantizationType.Q4_1:
            return GGUFTensor.unpack_q4_1(qdata, columns)
        if target == GGMLQuantizationType.Q8_0:
            return GGUFTensor.unpack_q8_0(qdata, columns)
        raise ValueError(f"Cannot re-quantize into {target.name}")

    def _pack_vocab(self, gguf_tensor: GGUFTensor,
                    target: GGMLQuantizationType) -> torch.Tensor:
        """Pack a vocab-sized tensor, zero padding the rows to a whole block.

        `_pack_q4nx` pads *columns* on its own but assumes the row count already
        divides `row_block_size`, and `_pack_q8nx` assumes the same, so a vocab
        that is not a multiple of 32 has to be padded here.  Zero scales over
        zero quants decode to zero rows, and the readers stop at `vocab_size`
        anyway.

        Q8_0 needs one extra step: `_pack_q8nx` only implements the 2D-blocked
        reshape.  That emits the same byte sequence as the flat one -- only the
        declared safetensors shape differs -- so the flag is flipped for the
        call and the result flattened back to (blocks, block_bytes) to match
        every other tensor in the file.
        """
        d, m, qw = self._unpack_as(gguf_tensor, target)

        rows = qw.shape[0]
        rows_padded = round_up_to_multiple(rows, self.row_block_size)
        if rows_padded != rows:
            pad = rows_padded - rows
            print(f"[INFO] Padding {gguf_tensor.name} rows {rows} -> {rows_padded} "
                  f"to fill a {self.row_block_size}-row block")
            d = F.pad(d, (0, 0, 0, pad), "constant", 0)
            if m is not None:
                m = F.pad(m, (0, 0, 0, pad), "constant", 0)
            qw = F.pad(qw, (0, 0, 0, pad), "constant", 0)

        if target != GGMLQuantizationType.Q8_0:
            return self._pack(d, m, qw, tensor_type=target)

        keep_block_in_2D = self.keep_block_in_2D
        self.keep_block_in_2D = True
        try:
            packed = self._pack(d, m, qw, tensor_type=target)
        finally:
            self.keep_block_in_2D = keep_block_in_2D

        return packed if keep_block_in_2D else packed.flatten(0, 1).contiguous()

    def _dequantize_bf16(self, gguf_tensor: GGUFTensor) -> torch.Tensor:
        """Dequantize a tensor straight to bf16, no blocking (the embedding)."""
        w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
        return torch.from_numpy(w).contiguous().to(torch.bfloat16)

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        if weights_type != 'language':
            raise ValueError(f"Unsupported weights_type: {weights_type} for hunyuan-dense")

        self.q4nx_tensors = {}

        if not self._has_lm_head():
            # Tied embeddings: the same quantized token_embd feeds both outputs, but
            # at different precisions -- bf16 for the lookup, quantized for the
            # output projection -- so it is packed twice rather than aliased.
            lm_head_type = self.tensor_q4nx_type_map["output.weight"]
            print(f"[INFO] Model does not have a lm_head, packing lm_head.weight "
                  f"from token_embd.weight as {lm_head_type.name}")
            self.q4nx_tensors["lm_head.weight"] = \
                self._pack_vocab(self.gguf_tensors["token_embd.weight"], lm_head_type)

        for key, gguf_tensor in self.gguf_tensors.items():
            if gguf_tensor.name not in self.forward_name_map:
                print(f"[WARN] No name mapping for {gguf_tensor.name}, skipping")
                continue

            q4nx_name = self.forward_name_map[gguf_tensor.name]
            target = self.tensor_q4nx_type_map[gguf_tensor.name]
            print(f"Processing tensor: {gguf_tensor.name} with type "
                  f"{gguf_tensor.tensor_type.name} -> {q4nx_name} with dtype {target.name}")

            if gguf_tensor.name == "token_embd.weight":
                if target in self._UNQUANTIZED_TYPES:
                    self.q4nx_tensors[q4nx_name] = self._dequantize_bf16(gguf_tensor)
                else:
                    self.q4nx_tensors[q4nx_name] = self._pack_vocab(gguf_tensor, target)
                continue

            if gguf_tensor.name == "output.weight":
                self.q4nx_tensors[q4nx_name] = self._pack_vocab(gguf_tensor, target)
                continue

            unpacked = self._unpack_as(gguf_tensor, target)
            self.q4nx_tensors[q4nx_name] = self._pack(*unpacked, tensor_type=target)

        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
