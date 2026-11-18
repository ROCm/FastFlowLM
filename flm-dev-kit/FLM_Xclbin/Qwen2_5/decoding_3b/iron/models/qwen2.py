# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


from ..model_config import ModelConfig
from ..constants import ModelArch

class qwen2_3b_config(ModelConfig, model_arch=ModelArch.QWEN2_3B):
    def __init__(self):
        super().__init__(
            name = "qwen2_3b",
            num_heads = 16,
            num_kv_heads = 2,
            intermediate_size = (11008 + 256),
            hidden_size = 2048,
            glu_slice = 1024,
            vocab_size = 151936,
            head_dim = 128,
            has_qk_norm = False,
            has_attn_bias = True
        )

class qwen2vl_3b_config(ModelConfig, model_arch=ModelArch.QWEN2VL_3B):
    def __init__(self):
        super().__init__(
            name = "qwen2_3b",
            num_heads = 16,
            num_kv_heads = 2,
            intermediate_size = (11008 + 256),
            hidden_size = 2048,
            glu_slice = 1024,
            vocab_size = 151936,
            head_dim = 128,
            has_qk_norm = False,
            has_attn_bias = True
        )
