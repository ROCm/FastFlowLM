# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, quantize
from safetensors.torch import save_file
import torch
from gguf.constants import GGMLQuantizationType


class LFM3(__Q4NX_Converter, model_arch=ModelArch.LFM3):
    """LiquidAI LFM3 (conv + sliding-window + full attention, sandwich norm, per-head attention gate).

    Source GGUF can be the vendor's quantization-aware Q4_0 file (its Q4_0 blocks are packed as they are) or
    the BF16 file (weights are quantized to Q4_0 here, round to nearest).
    """

    # a Q4 block covers 32 rows; tensors with fewer rows (the 16-row attention gate) stay BF16
    MIN_ROWS_FOR_Q4 = 32

    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()

    def _rows(self, gguf_tensor) -> int:
        # GGUF shapes are (columns, rows)
        return gguf_tensor.shape[1] if len(gguf_tensor.shape) > 1 else 1

    def _unpack_as(self, gguf_tensor, target_type: GGMLQuantizationType):
        """(scales, mins, quants) of the tensor in `target_type`. Quantizes first if the source is BF16/F32/F16."""
        if gguf_tensor.tensor_type in (GGMLQuantizationType.BF16, GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            w = gguf_tensor.dequantize().to(torch.float32).numpy()
            quantized = quantize(w, target_type).copy()
            unpack = gguf_tensor.unpack_q4_0 if target_type == GGMLQuantizationType.Q4_0 else gguf_tensor.unpack_q4_1
            return unpack(quantized, gguf_tensor.shape[0])
        return gguf_tensor.unpack(target_type)

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}

        # token_embd is tied to lm_head. The embedding goes out as BF16 (from --embed-bf16 if given, else dequantized
        # from the GGUF); lm_head is that same table quantized to Q4_1, like LFM2.
        token_embd = self.gguf_tensors["token_embd.weight"]
        vocab_size, hidden_size = token_embd.shape[1], token_embd.shape[0]
        embedding = self._load_bf16_embedding(vocab_size, hidden_size)
        if embedding is None:
            embedding = torch.from_numpy(dequantize(token_embd.data, token_embd.tensor_type)).contiguous().to(torch.bfloat16)
        self.q4nx_tensors[self.forward_name_map["token_embd.weight"]] = embedding

        if not self._has_lm_head():
            print("[INFO] Model has no lm_head (tied): quantizing the embedding table to Q4_1 as lm_head")
            quantized = quantize(embedding.to(torch.float32).numpy(), GGMLQuantizationType.Q4_1).copy()
            unpacked = token_embd.unpack_q4_1(quantized, hidden_size)
            self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)

        for key, gguf_tensor in self.gguf_tensors.items():
            if key == "token_embd.weight":
                continue
            if key == "rope_freqs.weight":
                continue  # the engine builds its own rope tables from config.json
            q4nx_name = self.forward_name_map[key]

            is_matrix = len(gguf_tensor.shape) == 2 and gguf_tensor.shape[1] > 1 and "conv.weight" not in key
            if is_matrix and self._rows(gguf_tensor) < self.MIN_ROWS_FOR_Q4:
                w = gguf_tensor.dequantize().to(torch.bfloat16)
                print(f"[INFO] {key}: {self._rows(gguf_tensor)} rows < {self.MIN_ROWS_FOR_Q4}, stored as BF16")
                self.q4nx_tensors[q4nx_name] = w
                continue

            if is_matrix:
                unpacked = self._unpack_as(gguf_tensor, self.default_tensor_type)
            else:
                unpacked = gguf_tensor.unpack(self.default_tensor_type)  # norms, conv taps: BF16 pass-through
            self.q4nx_tensors[q4nx_name] = self._pack_q4nx(*unpacked)

        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
