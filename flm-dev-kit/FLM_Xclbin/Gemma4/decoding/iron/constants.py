# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from enum import IntEnum, auto

class proj_pkt_id(IntEnum):
    to_rope = 1
    to_swa_rope = 2
    to_rms = 4
    to_glu = 8


class to_proj_x_in_pkt_id(IntEnum):
    from_rms = 0
    from_attn = 0
    from_swa_attn = 0
    from_glu = 22

class rms_pkt_id(IntEnum):
    to_proj = 0
    to_st = 5

class it1_pkt_id(IntEnum):
    to_rms  = 0
    to_rope = 1


class ModelArch(IntEnum):
    GEMMA4_E2B = auto()
    GEMMA4_E4B = auto()

ModelArchNames: dict[ModelArch, str] = {
    ModelArch.GEMMA4_E2B: "GEMMA4_E2B",
    ModelArch.GEMMA4_E4B: "GEMMA4_E4B",
}