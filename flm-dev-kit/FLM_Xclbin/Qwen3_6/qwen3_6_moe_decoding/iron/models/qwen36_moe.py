# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


from ..model_config import ModelConfig
from ..constants import ModelArch, QuantType

class qwen36_35b_a3b_config(ModelConfig, model_arch=ModelArch.QWEN3_6_35B_A3B):
    def __init__(self):
        super().__init__(
            name = "qwen36_35b_a3b",
            num_heads = 16,
            num_kv_heads = 2,
            intermediate_size = 512,
            hidden_size = 2048,
            glu_slice = 1024,
            vocab_size = 248320,
            head_dim = 256,
            linear_num_heads = 32,
            linear_conv_kernel_size = 4,
            linear_qk_head_dim = 128,
            linear_v_head_dim = 128,
            linear_num_qk_heads = 16,
            kv_head_per_cu=1,
            rope_partial_factor = 0.25,
            num_delta_net_cts = 4,
            has_qk_norm = True,
            num_attn_cus = 2,
            PROJ_QUANT_TYPE = QuantType.Q4_K,
        )
        self.num_active_experts=8
        self.num_experts=256        
