# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


from ..model_config import ModelConfig
from ..constants import ModelArch, QuantType

class nanbeige_3b_config(ModelConfig, model_arch=ModelArch.NANBEIGE_3B):
    def __init__(self):
        super().__init__(
            name = "nanbeige_3b",
            num_heads = 20,
            num_kv_heads = 4,
            intermediate_size = 10496 + 256,
            hidden_size = 2560,
            glu_slice = 1024,
            vocab_size = 166144,
            head_dim = 128,
            q4nx_row_block_size = 32,
            q4nx_col_block_size = 256,
            q4nx_group_size = 32,
            q_heads_padding = 8,
            # mirrors QUANT_TYPE in models/<model>.h; sets the packed block size
            quant_type = QuantType.Q4_1,
        )