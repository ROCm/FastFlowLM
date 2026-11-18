# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from enum import IntEnum, auto

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

class ModelArch(IntEnum):
    QWEN2_3B = auto()
    QWEN2VL_3B = auto()

ModelArchNames: dict[ModelArch, str] = {
    ModelArch.QWEN2_3B: "QWEN2_3B",
    ModelArch.QWEN2VL_3B: "QWEN2VL_3B",
}