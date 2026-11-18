# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


from ..model_config import ModelConfig
from ..constants import ModelArch, QuantType

class llama_3_2_1b_config(ModelConfig, model_arch=ModelArch.LLAMA_3_2_1B):
    def __init__(self):
        super().__init__(
            name = "llama_3_2_1b",
            num_heads = 32,
            num_kv_heads = 8,
            intermediate_size = 8192,
            hidden_size = 2048,
            glu_slice = 1024,
            vocab_size = 128256,
            head_dim = 64,
            q4nx_row_block_size = 32,
            q4nx_col_block_size = 256,
            q4nx_group_size = 32,
            # mirrors QUANT_TYPE in models/<model>.h; sets the packed block size
            quant_type = QuantType.Q4_1,
        )

class llama_3_2_3b_config(ModelConfig, model_arch=ModelArch.LLAMA_3_2_3B):
    def __init__(self):
        super().__init__(
            name = "llama_3_2_3b",
            num_heads = 24,
            num_kv_heads = 8,
            intermediate_size = 8192,
            hidden_size = 3072,
            glu_slice = 1024,
            vocab_size = 128256,
            head_dim = 128,
            q4nx_row_block_size = 32,
            q4nx_col_block_size = 256,
            q4nx_group_size = 32,
            # mirrors QUANT_TYPE in models/<model>.h; sets the packed block size
            quant_type = QuantType.Q4_1,
        )

class llama_3_1_8b_config(ModelConfig, model_arch=ModelArch.LLAMA_3_1_8B):
    def __init__(self):
        super().__init__(
            name = "llama_3_1_8b",
            num_heads = 32,
            num_kv_heads = 8,
            intermediate_size = 14336,
            hidden_size = 4096,
            glu_slice = 1024,
            vocab_size = 128256,
            head_dim = 128,
            q4nx_row_block_size = 32,
            q4nx_col_block_size = 256,
            q4nx_group_size = 32,
            # mirrors QUANT_TYPE in models/<model>.h; sets the packed block size
            quant_type = QuantType.Q4_1,
        )