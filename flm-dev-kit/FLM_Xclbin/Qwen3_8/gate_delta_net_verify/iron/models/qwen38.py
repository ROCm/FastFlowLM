# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


from ..model_config import ModelConfig
from ..constants import ModelArch, QuantType

class qwen38_27b_config(ModelConfig, model_arch=ModelArch.QWEN3_8_27B):
    def __init__(self):
        super().__init__(
            name = "qwen38_27b",
            intermediate_size = 17408,
            hidden_size = 5120,
            glu_slice = 1024,
            vocab_size = 248320,
            linear_num_heads = 48,
            linear_conv_kernel_size = 4,
            linear_qk_head_dim = 128,
            linear_v_head_dim = 128,
            linear_num_qk_heads = 16,
            num_delta_net_cts = 4,
            PROJ_QUANT_TYPE = QuantType.Q4_K,
            AB_PROJ_QUANT_TYPE = QuantType.Q8_0
        )
