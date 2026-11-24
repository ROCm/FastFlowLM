# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from enum import IntEnum, auto

class proj_pkt_id(IntEnum):
    to_rope = 1
    to_conv = 4
    to_aux = 8
    to_glu = 15  
    
    
    
    x_to_column_0 = 29
    x_to_column_1 = 25
    x_to_column_2 = 22
    x_to_column_3 = 31
    


class x_repeat_id(IntEnum):
    x_repeat_to_aux_x_proj_id = 18

class rms_pkt_id(IntEnum):
    to_proj = 0
    to_st = 1
    to_glu = 7
    from_glu = 11
    
class it1_pkt_id(IntEnum):
    to_rms  = 0
    to_rope = 1

class kv_out_pkt_id(IntEnum):
    to_it0 = 14
    to_it7 = 15

class aux_to_proj_x_in_pkt_id(IntEnum):
    to_out_proj = 2
    glu_to_aux_x_in = 9
    
    # glu_to_aux_x_in_no_repeat = 20


class linear_aux_id(IntEnum):
    to_ab_proj_weight = 3    
    to_router_proj_weight = 10

    
class ModelArch(IntEnum):
    QWEN3_6_35B_A3B = auto()

class QuantType(IntEnum):
    Q4_1 = auto()
    Q8_0 = auto()
    Q4_K = auto()

ModelArchNames: dict[ModelArch, str] = {
    ModelArch.QWEN3_6_35B_A3B: "QWEN3_6_35B_A3B"
}