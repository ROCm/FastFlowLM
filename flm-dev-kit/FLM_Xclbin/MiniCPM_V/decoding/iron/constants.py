# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from enum import IntEnum, auto

class proj_pkt_id(IntEnum):
    to_rope = 17
    to_conv = 4
    to_aux = 8
    to_glu = 15
    to_out = 12

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
    to_down_proj = 9

class ModelArch(IntEnum):
    QWEN3_5_2B = auto()
    QWEN3_5_4B = auto()
    QWEN3_5_9B = auto()
    QWEN3_5_08B = auto()
    QWEN3_5_OMNI = auto()

class QuantType(IntEnum):
    Q4_1 = auto()
    Q8_0 = auto()
    Q4_K = auto()
    
ModelArchNames: dict[ModelArch, str] = {
    ModelArch.QWEN3_5_2B: "QWEN3_5_2B",
    ModelArch.QWEN3_5_4B: "QWEN3_5_4B",
    ModelArch.QWEN3_5_9B: "QWEN3_5_9B",
    ModelArch.QWEN3_5_08B: "QWEN3_5_08B",
    ModelArch.QWEN3_5_OMNI: "QWEN3_5_OMNI"
}