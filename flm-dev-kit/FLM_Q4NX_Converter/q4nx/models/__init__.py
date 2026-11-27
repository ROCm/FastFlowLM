# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""
Model converters for different architectures.
Import this module to automatically register all available model converters.
"""

from .qwen3vl import Qwen3VL
from .llama import Llama
from .lfm2 import LFM2
from .lfm3 import LFM3
from .qwen3 import Qwen3
from .qwen35 import Qwen35, Qwen35_9B, Qwen35_2B, Qwen35_08B
from .qwen35_27b import Qwen35_27B
from .qwen38 import Qwen38_27B
from .qwen35_omni import Qwen35Omni
from .qwen2 import Qwen2
from .qwen2vl import Qwen2VL
from .gemma3 import Gemma3
from .gemma4 import Gemma4
from .gemma4_12b import Gemma4_12B
from .phi4 import Phi4
from .gpt_oss import GPTOSS
from .nanbeige import Nanbeige
from .qwen35_moe import QWEN35_A3B
from .hunyuan_dense import HunyuanDense
from .minicpm_v import MiniCPM_V_1B
from .minicpm_ocr import MiniCPM_OCR_1B
from .hy_mem import HyMemory30BA3B
from .minicpm_v_35b import MiniCPM_V_35B_A3B
from .dflash import DFlash
from .qwen38_dflash2 import Qwen38DFlash2
from .qwen3_asr import Qwen3ASR
from .inkling2 import Inkling2Loop
__all__ = ['Qwen3VL', 'Llama', 'LFM2', 'LFM3', 'Qwen3', 'Qwen2', 'Qwen2VL', 'Gemma3', 'Gemma4', 'Gemma4_12B', 'Phi4', 'GPTOSS', 'Nanbeige', 'Qwen35', 'Qwen35_2B', 'Qwen35_9B', 'Qwen35_08B', 'Qwen35_27B', 'Qwen35Omni', 'QWEN35_A3B', 'HunyuanDense', 'Qwen38_27B', 'MiniCPM_V_1B', 'MiniCPM_OCR_1B', 'DFlash', 'Qwen38DFlash2','MiniCPM_V_35B_A3B', 'HyMemory30BA3B', 'Qwen3ASR', 'Inkling2Loop']
