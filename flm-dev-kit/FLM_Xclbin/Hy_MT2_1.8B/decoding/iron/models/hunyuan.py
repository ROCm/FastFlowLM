# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from ..model_config import ModelConfig
from ..constants import ModelArch, QuantType


class hunyuan_dense_1_8b_config(ModelConfig, model_arch=ModelArch.HUNYUAN_DENSE_1_8B):
    """Hy-MT2-1.8B -- the `hunyuan-dense` GGUF architecture.

    A plain Llama-shaped GQA decoder; the one thing the array has to know about
    is that the per-head Q/K RMS norms run *after* RoPE rather than before, so
    the RoPE tile links against the rope_hunyuan.o variant.  See
    the HunYuan V1 dense model card / HF modeling code and kernels/rope_hunyuan.cc.

    4 KV heads with 16/4 = 4 Q heads per group.  The array runs this as
    ATTN_IMPL_1x8x1 -- the 1x4x1 kernel bodies only ever emitted two Q heads per
    CU (see models/hunyuan-dense-1.8b.h) -- so the group is padded out to 8 and
    q_heads_padding has to say so: it sizes the RoPE tile's Q buffer, and
    ModelConfig.PADDED_DQ sizes everything downstream of it.
    """

    def __init__(self):
        super().__init__(
            name = "hunyuan_dense_1_8b",
            num_heads = 16,
            num_kv_heads = 4,
            intermediate_size = 6144,
            hidden_size = 2048,
            glu_slice = 1024,
            vocab_size = 120818,
            head_dim = 128,
            has_qk_norm = True,
            # ATTN_IMPL_1x8x1: 4 Q heads per group padded out to a segment of 8
            q_heads_padding = 8,
            q4nx_row_block_size = 32,
            q4nx_col_block_size = 256,
            q4nx_group_size = 32,
            rope_link_with = "rope_hunyuan.o",
            # mirrors QUANT_TYPE in models/<model>.h; sets the packed block size
            quant_type = QuantType.Q4_K,
        )
