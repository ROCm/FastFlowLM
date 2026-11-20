# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


from ..model_config import ModelConfig
from ..constants import ModelArch, QuantType

class qwen35_2b_config(ModelConfig, model_arch=ModelArch.QWEN3_5_2B):
    def __init__(self):
        super().__init__(
            name = "qwen35_2b",
            num_heads = 8,
            num_kv_heads = 2,
            intermediate_size = 6144,
            hidden_size = 2048,
            glu_slice = 1024,
            vocab_size = 248320,
            head_dim = 256,
            linear_num_heads = 16,
            linear_conv_kernel_size = 4,
            linear_qk_head_dim = 128,
            linear_v_head_dim = 128,
            linear_num_qk_heads = 16,
            kv_head_per_round=2,
            rope_partial_factor = 0.25,
            num_delta_net_cts = 4,
            has_qk_norm = True,
            num_attn_cus = 1,
            PROJ_QUANT_TYPE = QuantType.Q4_K,
            AB_PROJ_QUANT_TYPE = QuantType.Q8_0
        )

class qwen35_omni_config(ModelConfig, model_arch=ModelArch.QWEN3_5_OMNI):
    def __init__(self):
        super().__init__(
            name = "qwen35_2b",
            num_heads = 8,
            num_kv_heads = 2,
            intermediate_size = 6656,
            hidden_size = 2560,
            glu_slice = 1024,
            vocab_size = 248320,
            head_dim = 256,
            linear_num_heads = 16,
            linear_conv_kernel_size = 4,
            linear_qk_head_dim = 128,
            linear_v_head_dim = 128,
            linear_num_qk_heads = 16,
            kv_head_per_round=2,
            rope_partial_factor = 0.25,
            num_delta_net_cts = 4,
            has_qk_norm = True,
            num_attn_cus = 1,
            PROJ_QUANT_TYPE = QuantType.Q4_1,
            AB_PROJ_QUANT_TYPE = QuantType.Q4_1
        )

class qwen35_08b_config(ModelConfig, model_arch=ModelArch.QWEN3_5_08B):
    def __init__(self):
        super().__init__(
            name = "qwen35_08b",
            num_heads = 8,
            num_kv_heads = 2,
            intermediate_size = 3584,
            hidden_size = 1024,
            glu_slice = 1024,
            vocab_size = 248320,
            head_dim = 256,
            linear_num_heads = 16,
            linear_conv_kernel_size = 4,
            linear_qk_head_dim = 128,
            linear_v_head_dim = 128,
            linear_num_qk_heads = 16,
            kv_head_per_round=2,
            rope_partial_factor = 0.25,
            num_delta_net_cts = 4,
            has_qk_norm = True,
            num_attn_cus = 1,
            PROJ_QUANT_TYPE = QuantType.Q4_K,
            AB_PROJ_QUANT_TYPE = QuantType.Q8_0
        )

class qwen35_4b_config(ModelConfig, model_arch=ModelArch.QWEN3_5_4B):
    def __init__(self):
        super().__init__(
            name = "qwen35_4b",
            num_heads = 16,
            num_kv_heads = 4,
            intermediate_size = 9216,
            hidden_size = 2560,
            glu_slice = 1024,
            vocab_size = 248320,
            head_dim = 256,
            linear_num_heads = 32,
            linear_conv_kernel_size = 4,
            linear_qk_head_dim = 128,
            linear_v_head_dim = 128,
            linear_num_qk_heads = 16,
            kv_head_per_round=2,
            rope_partial_factor = 0.25,
            num_delta_net_cts = 4,
            has_qk_norm = True,
            num_attn_cus = 2,
            PROJ_QUANT_TYPE = QuantType.Q4_K,
            AB_PROJ_QUANT_TYPE = QuantType.Q8_0
        )


class qwen35_9b_config(ModelConfig, model_arch=ModelArch.QWEN3_5_9B):
    def __init__(self):
        super().__init__(
            name = "qwen35_9b",
            num_heads = 16,
            num_kv_heads = 4,
            intermediate_size = 12288,
            hidden_size = 4096,
            glu_slice = 1024,
            vocab_size = 248320,
            head_dim = 256,
            linear_num_heads = 32,
            linear_conv_kernel_size = 4,
            linear_qk_head_dim = 128,
            linear_v_head_dim = 128,
            linear_num_qk_heads = 16,
            kv_head_per_round=2,
            rope_partial_factor = 0.25,
            num_delta_net_cts = 4,
            has_qk_norm = True,
            num_attn_cus = 2,
            PROJ_QUANT_TYPE = QuantType.Q4_K,
            AB_PROJ_QUANT_TYPE = QuantType.Q8_0
        )