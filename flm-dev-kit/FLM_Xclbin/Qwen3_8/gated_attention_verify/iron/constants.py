# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import os
from enum import IntEnum, auto

# Query rows per forward pass, set by the top-level Makefile. The AIE kernels
# and the host read the same number as a -DL_TOKENS define, so all three
# languages move together.
L_TOKENS = int(os.environ.get("L_TOKENS", "8"))

class proj_pkt_id(IntEnum):
  
    to_rms = 2    # o,      -> the residual's memtile
    to_rope = 4   # qkv,    -> the RoPE tile (today: straight to DDR)
    to_aux = 8    # z,      -> the gate's memtile
    to_glu = 16   # up|gate -> the GLU tile

class proj_x_pkt_id(IntEnum):
   
    # Numerically the same as aux_to_proj_x_in_pkt_id.to_out_proj, which the
    # gate tile stamps *inbound* to the replay tile -- different ports, so the
    # collision is only in the value.
    from_replay = 2

class rms_pkt_id(IntEnum):
   
    to_proj = 0
    to_st = 1
    to_out = 4

class it1_pkt_id(IntEnum):
    to_rms  = 0
    to_rope = 1

class aux_to_rms_pkt_id(IntEnum):
   
    to_residual = 3

class aux_to_proj_x_in_pkt_id(IntEnum):
    """Ids stamped inbound to proj_x_replay_mt by whoever produces an x.

    Landing channels there must be packet-switched: the rms tile already reaches
    the tile that way, and a switchbox will not drive one DMA channel from both a
    circuit and a packet flow. Only has to differ from the ids sharing its route.
    """
    to_out_proj = 2
    to_down_proj = 19

class ModelArch(IntEnum):
    QWEN3_8_27B = auto()

class QuantType(IntEnum):
    Q4_1 = auto()
    Q8_0 = auto()
    Q4_K = auto()
    
ModelArchNames: dict[ModelArch, str] = {
    ModelArch.QWEN3_8_27B: "QWEN3_8_27B"
}