# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from enum import IntEnum, auto

class proj_pkt_id(IntEnum):
    to_rope = 1
    to_aux = 8
    to_glu = 15

    x_to_column_0 = 29
    x_to_column_1 = 25
    x_to_column_2 = 22
    x_to_column_3 = 31



class rms_pkt_id(IntEnum):
    to_proj = 0
    to_st = 1
    to_glu = 7
    from_glu = 11

class it1_pkt_id(IntEnum):
    to_rms  = 0
    to_rope = 1

# packet ids into the proj engine's x input (MT1), besides rms_pkt_id.to_proj
class proj_x_pkt_id(IntEnum):
    from_attn = 2       # attention output, for o_proj
    from_glu = 9        # GLU: shared expert h, routed up/gate x, routed experts' h

class router_w_pkt_id(IntEnum):
    to_router_w = 10

# RoPE -> attention CU i sends Q with packet id ROPE_Q_PKT_ID_BASE + i. 16..19 are
# used by nothing else and form an aligned group of 4, so if the router merges
# them into one masked rule (mask 0x1C, value 16) that rule matches only Q. With
# ids 0..3 the merged rule also caught rms -> proj (id 0) and the normal build
# could not route at all.
ROPE_Q_PKT_ID_BASE = 16


class ModelArch(IntEnum):
    HY_MEMORY_30B_A3B = auto()

class QuantType(IntEnum):
    Q4_1 = auto()
    Q8_0 = auto()
    Q4_K = auto()

ModelArchNames: dict[ModelArch, str] = {
    ModelArch.HY_MEMORY_30B_A3B: "HY_MEMORY_30B_A3B"
}
