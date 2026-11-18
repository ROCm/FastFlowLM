# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


from ..model_config import ModelConfig
from ..constants import ModelArch, QuantType

class qwen38_27b_config(ModelConfig, model_arch=ModelArch.QWEN3_8_27B):
    def __init__(self):
        super().__init__(
            name = "qwen38_27b",
            num_heads = 24,
            num_kv_heads = 4,
            intermediate_size = 17408,
            hidden_size = 5120,
            glu_slice = 1024,
            vocab_size = 248320,
            head_dim = 256,
            kv_head_per_round=1,
            rope_partial_factor = 0.25,
            has_qk_norm = True,
            num_attn_cus = 2,
            # 24/4 = 6 q heads per group, padded to 8 for the 1x8x1 attention CU.
            q_heads_padding = 8,
            PROJ_QUANT_TYPE = QuantType.Q4_K
        )
