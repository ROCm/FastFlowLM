# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from enum import IntEnum, auto

# Packet buffer layout, mirroring kernels/aie_array_layout.h -- change both or
# neither.  A packet buffer is [pad][one uint32 stream packet header][payload],
# and the DMA sends the header word plus the payload as one contiguous run, so
# the dma_bd offset is PKT_ID_OFFSET and its len is the payload plus the two
# bf16 the header word occupies.
#
# The payload offset has to keep the payload 64-byte aligned: the kernels reach
# into it with 512-bit vector ops, and AIE2P silently truncates a 512-bit access
# down to the 64-byte boundary below it.  See the note in that header.
PKT_PAYLOAD_OFFSET = 32              # bf16 elements
PKT_ID_OFFSET      = PKT_PAYLOAD_OFFSET - 2

class proj_pkt_id(IntEnum):
    to_rope = 1
    to_rms_norm = 4
    to_glu = 8

class rms_pkt_id(IntEnum):
    to_proj = 0
    to_st = 1

class it1_pkt_id(IntEnum):
    to_rms  = 0
    to_rope = 1

class kv_out_pkt_id(IntEnum):
    to_it0 = 14
    to_it7 = 15

class aux_to_proj_x_in_pkt_id(IntEnum):
    to_out_proj = 2
    to_down_proj = 0

class QuantType(IntEnum):
    """Projection weight format, mirroring kernels/models/all_models.h.

    Values match that header so the two can be compared by eye.  A model's IRON
    config and its kernel header must pick the same format -- a mismatch is a
    wrong DMA length, i.e. silently misread weights -- so ModelConfig reads
    QUANT_TYPE back out of models/<model>.h and refuses to build on a
    disagreement.
    """
    Q4_1 = 0
    Q8_0 = 1
    Q4_K = 2
    Q4_0 = 3


# Packed bytes per weight, as (numerator, denominator).  Same numbers the
# static_asserts in kernels/model_spec.h pin the block structs to.
QuantBytesPerWeight: dict[QuantType, tuple[int, int]] = {
    QuantType.Q4_0: (9, 16),    # 4.5   bpw: 4 + 16/32
    QuantType.Q4_1: (5, 8),     # 5     bpw: 4 + 16/32 + 16/32
    QuantType.Q4_K: (37, 64),   # 4.625 bpw: 4 + 8/32 + 8/32 + 2*16/256
    QuantType.Q8_0: (17, 16),   # 8.5   bpw: 8 + 16/32
}


class ModelArch(IntEnum):
    LLAMA_3_2_1B = auto()
    LLAMA_3_2_3B = auto()
    LLAMA_3_1_8B = auto()
    QWEN3_0_6B = auto()
    QWEN3_1_7B = auto()
    QWEN3_4B = auto()
    QWEN3_8B = auto()
    GEMMA3_4B = auto()
    PHI4_4B  = auto()
    NANBEIGE_3B = auto()
    HUNYUAN_DENSE_1_8B = auto()
    HY_MEMORY_30B_A3B_DENSE = auto()

ModelArchNames: dict[ModelArch, str] = {
    ModelArch.LLAMA_3_2_1B: "LLAMA_3_2_1B",
    ModelArch.LLAMA_3_2_3B: "LLAMA_3_2_3B",
    ModelArch.LLAMA_3_1_8B: "LLAMA_3_1_8B",
    ModelArch.QWEN3_0_6B: "QWEN3_0_6B",
    ModelArch.QWEN3_1_7B: "QWEN3_1_7B",
    ModelArch.QWEN3_4B: "QWEN3_4B",
    ModelArch.QWEN3_8B: "QWEN3_8B",
    ModelArch.GEMMA3_4B: "GEMMA3_4B",
    ModelArch.PHI4_4B:  "PHI4_4B",
    ModelArch.NANBEIGE_3B: "NANBEIGE_3B",
    ModelArch.HUNYUAN_DENSE_1_8B: "HUNYUAN_DENSE_1_8B",
    ModelArch.HY_MEMORY_30B_A3B_DENSE: "HY_MEMORY_30B_A3B_DENSE",
}