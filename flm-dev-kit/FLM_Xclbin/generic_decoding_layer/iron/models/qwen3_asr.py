# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from ..model_config import ModelConfig
from ..constants import ModelArch, QuantType


class qwen3_asr_1_7b_config(ModelConfig, model_arch=ModelArch.QWEN3_ASR_1_7B):
    """Qwen3-ASR-1.7B text decoder (Confucius4-R2T2): qwen3_1_7b_config with Q8_0 weights.

    Q8_0 because 4-bit RTN breaks this fine-tune; the packing stride (PARALLEL_ROWS = 32) lives in models/qwen3-asr-1.7b.h.
    """

    def __init__(self):
        super().__init__(
            name = "qwen3_asr_1_7b",
            num_heads = 16,
            num_kv_heads = 8,
            intermediate_size = 6144,
            hidden_size = 2048,
            glu_slice = 1024,
            vocab_size = 151936,
            head_dim = 128,
            has_qk_norm = True,
            # mirrors QUANT_TYPE in models/<model>.h; sets the packed block size
            quant_type = QuantType.Q8_0,
        )


class qwen3_asr_1_7b_mix_config(ModelConfig, model_arch=ModelArch.QWEN3_ASR_1_7B_MIX):
    """qwen3_asr_1_7b_config with mixed Q4_1 / Q8_0 projections: one 10240-byte weight unit per DMA (models/qwen3-asr-1.7b-mix.h)."""

    def __init__(self):
        super().__init__(
            name = "qwen3_asr_1_7b_mix",
            num_heads = 16,
            num_kv_heads = 8,
            intermediate_size = 6144,
            hidden_size = 2048,
            glu_slice = 1024,
            vocab_size = 151936,
            head_dim = 128,
            has_qk_norm = True,
            quant_type = QuantType.Q41_Q80_MIX,
        )
