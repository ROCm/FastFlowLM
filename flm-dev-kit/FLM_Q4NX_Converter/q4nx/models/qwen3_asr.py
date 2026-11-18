# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, GGMLQuantizationType
import torch


class Qwen3ASR(__Q4NX_Converter, model_arch=ModelArch.QWEN3_ASR):
    """Confucius4-R2T2 / Qwen3-ASR: a Qwen3-1.7B text decoder plus a 24-layer audio tower.

    -t language : the text GGUF (blk.* names). Tied embedding: `model.embed_tokens.weight` is BF16 (pass --embed-bf16
                  <model.safetensors> to take it from the original BF16 checkpoint) and `lm_head.weight` is built from
                  token_embd in the config's default type (Q8_0). Every other tensor keeps the type it has in the GGUF.
    -t audio    : a GGUF holding thinker.audio_tower.* in HF names (see configs/qwen3_asr.json). BF16, with the audio MM
                  tile re-order on the matrices the NPU matmul reads, exactly like Qwen3.5-Omni's tower.
    -t npu_q41  : the F16 (or BF16) text GGUF -> npu_q41.q4nx, the Q4_1 copies the mixed NPU decode reads (q, k, o, gate, up of
                  every layer and the tied lm_head; v and down stay Q8_0 from model.q4nx). See q4nx/q41_npu.py.
    """

    def __init__(self, gguf_reader: GGUFReader):
        print("[INFO] Using Qwen3ASR converter")
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    # audio matrices that go through the NPU matmul (same list as qwen35_omni.py)
    _AUDIO_MATRIX_SUFFIXES = ("fc1.weight", "fc2.weight", "k_proj.weight", "out_proj.weight", "q_proj.weight",
                              "v_proj.weight", "conv_out.weight", "proj1.weight", "proj2.weight")

    def convert(self, q4nx_path: str, weights_type: str = "language"):
        self.q4nx_tensors = {}
        if weights_type == "language":
            self._convert_language()
            self._extract_tokenizer_json(q4nx_path)
            self._export_q4nx_tensors(q4nx_path)
        elif weights_type == "audio":
            self._convert_audio()
            self._export_q4nx_tensors(q4nx_path, filename="audio_weight.q4nx")
        elif weights_type == "npu_q41":
            self._convert_npu_q41()
            self._export_q4nx_tensors(q4nx_path, filename="npu_q41.q4nx")
        else:
            raise ValueError(f"Unsupported weights_type {weights_type} for Qwen3-ASR (use language, audio or npu_q41)")

    # the projections the mixed NPU decode reads as Q4_1 (generic_decoding_layer/models/qwen3-asr-1.7b-mix.h)
    _NPU_Q41_SUFFIXES = ("attn_q.weight", "attn_k.weight", "attn_output.weight", "ffn_gate.weight", "ffn_up.weight")

    def _convert_npu_q41(self):
        from ..q41_npu import quantize_q41, pack_q41
        for key, t in self.gguf_tensors.items():
            is_embd = t.name == "token_embd.weight"
            if not (is_embd or t.name.endswith(self._NPU_Q41_SUFFIXES)):
                continue
            if t.tensor_type not in (GGMLQuantizationType.F16, GGMLQuantizationType.BF16, GGMLQuantizationType.F32):
                raise ValueError(f"-t npu_q41 needs an unquantized GGUF (F16 / BF16), {t.name} is {t.tensor_type.name}")
            w = torch.from_numpy(dequantize(t.data, t.tensor_type)).float()       # [rows, cols]
            d, m, q = quantize_q41(w)
            name = "lm_head.weight" if is_embd else self.forward_name_map[t.name]
            self.q4nx_tensors[name] = pack_q41(d, m, q)
            print(f"[INFO] {name}: Q4_1 for the NPU, {tuple(w.shape)}")

    def _convert_language(self):
        embd = self.gguf_tensors["token_embd.weight"]
        if self._has_lm_head():
            raise ValueError("Qwen3-ASR is expected to have a tied embedding (no output.weight)")
        # lm_head = the embedding table in the decoder's weight type
        print("[INFO] Tied embedding: lm_head.weight from token_embd.weight in", self.default_tensor_type.name)
        unpacked = embd.unpack(self.default_tensor_type)
        target = embd.get_used_quantization_type(self.default_tensor_type)
        self.q4nx_tensors["lm_head.weight"] = self._pack(*unpacked, tensor_type=target)

        for key, t in self.gguf_tensors.items():
            if t.name == "token_embd.weight":
                w = self._load_bf16_embedding(embd.shape[1], embd.shape[0])   # GGUF dims are (cols, rows)
                if w is None:
                    print("[WARN] Embedding taken from the GGUF, not from BF16 safetensors (use --embed-bf16)")
                    w = torch.from_numpy(dequantize(t.data, t.tensor_type)).contiguous().to(torch.bfloat16)
                self.q4nx_tensors[self.forward_name_map[t.name]] = w
                continue
            if t.name not in self.forward_name_map:
                print(f"[WARN] No name mapping for {t.name}, skipping")
                continue
            target = t.get_used_quantization_type(self.tensor_q4nx_type_map[t.name])
            unpacked = t.unpack(self.tensor_q4nx_type_map[t.name])
            self.q4nx_tensors[self.forward_name_map[t.name]] = self._pack(*unpacked, tensor_type=target)

    def _convert_audio(self):
        for key, t in self.gguf_tensors.items():
            if t.name not in self.forward_name_map:
                print(f"[WARN] No name mapping for {t.name}, skipping")
                continue
            unpacked = t.unpack(GGMLQuantizationType.BF16)
            assert len(unpacked) == 1 and isinstance(unpacked[0], torch.Tensor), "audio tensors are expected unquantized"
            w = unpacked[0]
            if w.dtype != torch.bfloat16:
                w = w.to(torch.bfloat16)
            new_name = self.forward_name_map[t.name]
            if new_name.endswith(self._AUDIO_MATRIX_SUFFIXES):
                w = self.audio_mm_weight_rearrange(w).contiguous()
            self.q4nx_tensors[new_name] = w
