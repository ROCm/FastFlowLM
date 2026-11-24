# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


from ..model_config import ModelConfig
from ..constants import ModelArch, QuantType

class qwen3_0_6b_config(ModelConfig, model_arch=ModelArch.QWEN3_0_6B):
    def __init__(self):
        super().__init__(
            name = "qwen3_0_6b",
            num_heads = 16,
            num_kv_heads = 8,
            intermediate_size = 3072,
            hidden_size = 1024,
            glu_slice = 1024,
            vocab_size = 151936,
            head_dim = 128,
            has_qk_norm = True,
            # mirrors QUANT_TYPE in models/<model>.h; sets the packed block size
            quant_type = QuantType.Q4_1,
        )

class qwen3_1_7b_config(ModelConfig, model_arch=ModelArch.QWEN3_1_7B):
    def __init__(self):
        super().__init__(
            name = "qwen3_1_7b",
            num_heads = 16,
            num_kv_heads = 8,
            intermediate_size = 6144,
            hidden_size = 2048,
            glu_slice = 1024,
            vocab_size = 151936,
            head_dim = 128,
            has_qk_norm = True,
            # mirrors QUANT_TYPE in models/<model>.h; sets the packed block size
            quant_type = QuantType.Q4_1,
        )


class qwen3_4b_config(ModelConfig, model_arch=ModelArch.QWEN3_4B):
    def __init__(self):
        super().__init__(
            name = "qwen3_4b",
            num_heads = 32,
            num_kv_heads = 8,
            intermediate_size = 9728,
            hidden_size = 2560,
            glu_slice = 1024,
            vocab_size = 151936,
            head_dim = 128,
            has_qk_norm = True,
            # mirrors QUANT_TYPE in models/<model>.h; sets the packed block size
            quant_type = QuantType.Q4_1,
        )


class qwen3_8b_config(ModelConfig, model_arch=ModelArch.QWEN3_8B):
    def __init__(self):
        super().__init__(
            name = "qwen3_8b",
            num_heads = 32,
            num_kv_heads = 8,
            intermediate_size = 12288,
            hidden_size = 4096,
            glu_slice = 1024,
            vocab_size = 151936,
            head_dim = 128,
            has_qk_norm = True,
            # mirrors QUANT_TYPE in models/<model>.h; sets the packed block size
            quant_type = QuantType.Q4_1,
        )