# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from ..model_config import ModelConfig
from ..constants import ModelArch, QuantType

class hy_memory_30b_a3b_config(ModelConfig, model_arch=ModelArch.HY_MEMORY_30B_A3B):
    def __init__(self):
        super().__init__(
            name = "hy_memory_30b_a3b",
            num_heads = 32,
            num_kv_heads = 4,
            # experts and shared expert are 768 wide (expert_hidden_dim); one GLU
            # slice = one proj round of 512 (see models/hy-memory-30b-a3b.h)
            intermediate_size = 768,
            hidden_size = 2048,
            glu_slice = 512,
            vocab_size = 120832,
            head_dim = 128,
            # 4 attention CUs (2 columns x 2), one KV head each
            kv_head_per_cu = 1,
            num_attn_cus = 4,
            rope_partial_factor = 1.0,
            has_qk_norm = True,
            PROJ_QUANT_TYPE = QuantType.Q4_K,
            num_experts = 128,
            num_active_experts = 8,
        )
