# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from .qwen3 import qwen3_0_6b_config
from .llama import llama_3_2_1b_config, llama_3_2_3b_config, llama_3_1_8b_config
from .phi4 import phi4_4b_config
from .gemma3 import gemma3_4b_config
from .nanbeige import nanbeige_3b_config
from .hunyuan import hunyuan_dense_1_8b_config
from .qwen3_asr import qwen3_asr_1_7b_config, qwen3_asr_1_7b_mix_config

__all__ = [
    "qwen3_0_6b_config",
    "llama_3_2_1b_config",
    "llama_3_2_3b_config",
    "llama_3_1_8b_config",
    "phi4_4b_config",
    "gemma3_4b_config",
    "nanbeige_3b_config",
    "hunyuan_dense_1_8b_config",
    "qwen3_asr_1_7b_config",
    "qwen3_asr_1_7b_mix_config",
]