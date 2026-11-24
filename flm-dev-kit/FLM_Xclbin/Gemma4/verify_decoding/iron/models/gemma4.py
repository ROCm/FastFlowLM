# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


from ..model_config import ModelConfig
from ..constants import ModelArch

class gemma4_e2b_config(ModelConfig, model_arch=ModelArch.GEMMA4_E2B):
    def __init__(self):
        super().__init__(
            name = "gemma4_e2b",
            num_heads = 8,
            num_kv_heads = 1,
            intermediate_size = 6144,
            hidden_size = 1536,
            glu_slice = 1024,
            vocab_size = 262144,
            head_dim = 512,
            swa_head_dim=256,
            kv_head_per_round=1,
            has_qk_norm = True,
            num_attn_cus = 1,
            per_layer_input_dim=256,
            enable_double_wide_mlp=True
        )

class gemma4_e4b_config(ModelConfig, model_arch=ModelArch.GEMMA4_E4B):
    def __init__(self):
        super().__init__(
            name = "gemma4_e4b",
            num_heads = 8,
            num_kv_heads = 2,
            intermediate_size = 10240,
            hidden_size = 2560,
            glu_slice = 1024,
            vocab_size = 262144,
            head_dim = 512,
            swa_head_dim=256,
            kv_head_per_round=2,
            has_qk_norm = True,
            num_attn_cus = 1,
            per_layer_input_dim=256,
            enable_double_wide_mlp=False,
            attn_qk_link_with = "attn_qk_e4b.o",
            attn_kv_link_with = "attn_kv_e4b.o"
        )
