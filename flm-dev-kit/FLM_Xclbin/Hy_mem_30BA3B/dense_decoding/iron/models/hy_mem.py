# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from ..model_config import ModelConfig
from ..constants import ModelArch, QuantType


class hy_memory_30b_a3b_dense_config(ModelConfig, model_arch=ModelArch.HY_MEMORY_30B_A3B_DENSE):
    """Hy-Memory-30B-A3B (hy_v3) -- the dense layer (layer 0) only.

    Layers 1-47 are MoE and run on ../moe_decoding.  Q/K norms run *before*
    RoPE, like Qwen3, so the default rope.o is right (not rope_hunyuan.o).

    32 Q heads over 4 KV heads = 8 per group: exactly one ATTN_IMPL_1x8x1
    segment, so q_heads_padding = 8 adds no pad heads.

    glu_slice = 512 (one proj round), because 2 * 6912 = 13824 does not split
    into 1024-value slices.  See models/hy-memory-30b-a3b-dense.h.
    """

    def __init__(self):
        super().__init__(
            name = "hy_memory_30b_a3b_dense",
            num_heads = 32,
            num_kv_heads = 4,
            intermediate_size = 6912,
            hidden_size = 2048,
            glu_slice = 512,
            vocab_size = 120832,
            head_dim = 128,
            has_qk_norm = True,
            # ATTN_IMPL_1x8x1: 8 Q heads per group, no padding needed
            q_heads_padding = 8,
            q4nx_row_block_size = 32,
            q4nx_col_block_size = 256,
            q4nx_group_size = 32,
            # mirrors QUANT_TYPE in models/<model>.h; sets the packed block size
            quant_type = QuantType.Q4_K,
        )
