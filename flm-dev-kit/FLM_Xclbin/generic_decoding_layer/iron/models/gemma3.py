# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from ..model_config import ModelConfig
from ..constants import ModelArch

class gemma3_4b_config(ModelConfig, model_arch=ModelArch.GEMMA3_4B):
    def __init__(self):
        super().__init__(
            name = "gemma3_4b",
            num_heads = 8,
            num_kv_heads = 4,
            intermediate_size = 10240,
            hidden_size = 2560,
            glu_slice = 1024,
            vocab_size = 262208,
            head_dim = 256,
            has_qk_norm = True
        )