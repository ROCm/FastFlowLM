# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from ..gguf_tensor import GGUFTensor

from gguf import GGUFReader, GGMLQuantizationType, dequantize
import torch


class HyMemory30BA3B(__Q4NX_Converter, model_arch=ModelArch.HY_MEMORY_30B_A3B):
    """Converter for the `hy_v3` GGUF architecture (tencent Hy-Memory-30B-A3B).

    The shape is close to Qwen3.6-35B-A3B's MoE half, and the q4nx names are
    the Qwen3.6 ones so the engine's weight table can be reused, but the
    tensor surgery Qwen3.5/3.6 need does not apply here:

    * Every layer is plain GQA attention (32 q / 4 kv heads, head_dim 128)
      with per-head q/k RMS norms -- no linear attention, no q/gate split, so
      no row reorders at all.
    * No RoPE row permute either: llama.cpp keeps the HF `rotate_half` row
      order for this arch.  Checked against the BF16 checkpoint: q/k/v/o
      dequantized from the GGUF reach cosine ~0.998 with the HF rows as they
      are, which a permuted copy could not.
    * Norm weights are plain multipliers (no "+1" offset), bit-identical to HF.
    * Layer 0 is a dense FFN (6912); layers 1..47 are MoE with 128 experts of
      width 768 (expert_feed_forward_length, *not* HF's moe_intermediate_size
      1536) plus one 768-wide shared expert.  There is no shared-expert gate.
    * The router is sigmoid with a per-expert bias (`exp_probs_b`) that only
      picks the top-8.  The bias is kept FP32: it is 128 floats per layer, and
      the near-ties it decides would flip under bf16 rounding.

    Storage per tensor kind:

        token_embd                -> BF16 (the GGUF copy is already BF16, lossless)
        norms                     -> BF16 (GGUF F32 holds bf16 values, lossless)
        router  [128, 2048]       -> BF16, transposed to [2048, 128] like Qwen3.6
        expert bias [128]         -> F32
        every projection (Q4_K)   -> q4k blocks via _pack_q4k
        lm_head (Q8_0)            -> q8nx blocks via _pack_q8nx
    """

    def __init__(self, gguf_reader: GGUFReader):
        print("[INFO] Using HyMemory30BA3B converter")
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()

    def _require_type(self, gguf_tensor: GGUFTensor, expected: GGMLQuantizationType):
        """Fail loudly instead of silently re-quantizing.

        The NPU kernels read one block format per tensor kind, so a GGUF that
        ships, say, a Q6_K ffn_down (common in *_K_M mixes) must not quietly
        turn into a Q4_1 block inside a Q4_K layer.
        """
        if gguf_tensor.tensor_type != expected:
            raise ValueError(f"{gguf_tensor.name} is {gguf_tensor.tensor_type.name}, "
                             f"this converter expects {expected.name}")

    def _pack_q4k_tensor(self, gguf_tensor: GGUFTensor) -> torch.Tensor:
        self._require_type(gguf_tensor, GGMLQuantizationType.Q4_K)

        # A 3D expert tensor [n_expert, rows, cols] unpacks as (n_expert * rows, cols).
        # As long as rows divides row_block_size no 32-row block straddles two
        # experts, so the packed result is just the experts stacked one after
        # another: expert e owns block rows [e * rows / 32, (e + 1) * rows / 32).
        rows_per_matrix = int(gguf_tensor.shape[1])
        assert rows_per_matrix % self.row_block_size == 0, \
            f"{gguf_tensor.name}: {rows_per_matrix} rows do not fill whole {self.row_block_size}-row blocks"

        t, u, q = gguf_tensor.unpack(GGMLQuantizationType.Q4_K)
        return self._pack(t, u, q, tensor_type=GGMLQuantizationType.Q4_K)

    def _pack_lm_head(self, gguf_tensor: GGUFTensor) -> torch.Tensor:
        self._require_type(gguf_tensor, GGMLQuantizationType.Q8_0)
        vocab = int(gguf_tensor.shape[1])
        assert vocab % self.row_block_size == 0, \
            f"vocab {vocab} is not a multiple of {self.row_block_size}; would need row padding"
        d, _, qw = gguf_tensor.unpack(GGMLQuantizationType.Q8_0)
        return self._pack(d, None, qw, tensor_type=GGMLQuantizationType.Q8_0)

    def _to_bf16(self, gguf_tensor: GGUFTensor) -> torch.Tensor:
        # dequantize() handles F32 and BF16 alike; both are exact in float32.
        w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
        return torch.from_numpy(w).to(torch.bfloat16).contiguous()

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        if weights_type != 'language':
            raise ValueError(f"Unsupported weights_type: {weights_type} for hy_v3")
        if not self._has_lm_head():
            raise ValueError("hy_v3 GGUF has no output.weight; Hy-Memory does not tie embeddings")

        self.q4nx_tensors = {}

        for key, gguf_tensor in self.gguf_tensors.items():
            if gguf_tensor.name not in self.forward_name_map:
                print(f"[WARN] No name mapping for {gguf_tensor.name}, skipping")
                continue

            q4nx_name = self.forward_name_map[gguf_tensor.name]
            target = self.tensor_q4nx_type_map[gguf_tensor.name]
            print(f"Processing tensor: {gguf_tensor.name} with type "
                  f"{gguf_tensor.tensor_type.name} -> {q4nx_name} with dtype {target.name}")

            if target == GGMLQuantizationType.Q4_K:
                packed = self._pack_q4k_tensor(gguf_tensor)
            elif target == GGMLQuantizationType.Q8_0:
                packed = self._pack_lm_head(gguf_tensor)
            elif target == GGMLQuantizationType.F32:
                # expert bias: keep full precision, see class docstring
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                packed = torch.from_numpy(w).to(torch.float32).contiguous()
            elif q4nx_name.endswith("moe_router.weight"):
                # GGUF gives [n_expert, D]; the Qwen3.6 router reads [D, n_expert]
                packed = self._to_bf16(gguf_tensor).transpose(0, 1).contiguous()
            else:
                # embedding and norms
                packed = self._to_bf16(gguf_tensor)

            self.q4nx_tensors[q4nx_name] = packed

        # Every mapped GGUF tensor must have landed; a silent gap here would only
        # show up as garbage output on the NPU.  The name map spans layers 0..47
        # for every pattern, so layer 0's MoE names exist there without a GGUF
        # source -- only names whose GGUF tensor is present count.
        missing = [q4nx_name for gguf_name, q4nx_name in self.forward_name_map.items()
                   if gguf_name in self.gguf_tensors and q4nx_name not in self.q4nx_tensors]
        if missing:
            raise RuntimeError(f"{len(missing)} mapped tensors not written, e.g. {missing[:5]}")

        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
