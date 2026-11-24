# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from ..model_config import ModelConfig
from ..constants import ModelArch, QuantType

class phi4_4b_config(ModelConfig, model_arch=ModelArch.PHI4_4B):
    def __init__(self):
        super().__init__(
            name = "phi4_4b",
            num_heads = 24,
            num_kv_heads = 8,
            intermediate_size = 8192,
            hidden_size = 3072,
            glu_slice = 1024,
            vocab_size = 200064,
            head_dim = 128,
            q4nx_row_block_size = 32,
            q4nx_col_block_size = 256,
            q4nx_group_size = 32,
            rope_link_with = "rope_phi4.o",
            # mirrors QUANT_TYPE in models/<model>.h; sets the packed block size
            quant_type = QuantType.Q4_1,
        )
