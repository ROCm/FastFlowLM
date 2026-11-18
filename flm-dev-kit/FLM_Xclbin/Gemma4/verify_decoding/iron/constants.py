# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import os
from enum import IntEnum, auto

# Rows per dispatch, set by the Makefile (-DL_TOKENS for the kernels reads the same number).
L_TOKENS = int(os.environ.get("L_TOKENS", "1"))
# The projection core exists in two forms: the shipping single-token core (L_TOKENS == 1) and the 8-token mmul core.
# Any 1 < L_TOKENS <= 8 runs the 8-token core with unused rows.
KERNEL_L = 1 if L_TOKENS == 1 else 8

class proj_pkt_id(IntEnum):
    to_rope = 1
    to_swa_rope = 2
    to_rms = 4
    to_glu = 8


class to_proj_x_in_pkt_id(IntEnum):
    from_rms = 0
    # one packet per token at L_TOKENS > 1, stamped with the ids of the o landing buffers of the x replay tile (proj_x_in_pkt_id)
    from_attn = 0 if L_TOKENS == 1 else 17
    from_swa_attn = 0 if L_TOKENS == 1 else 18
    from_glu = 22

class proj_x_in_pkt_id(IntEnum):
    """Ids a producer stamps inbound to the x replay tile (projonly design: the shim). One landing channel per id."""
    to_main = 16     # qkv, up|gate (time-multiplexed on one buffer)
    to_o_global = 17
    to_o_swa = 18
    to_down = 19

class proj_x_pkt_id(IntEnum):
    """The one id the replay channels merge under on their way to the x FIFO (see proj_x_replay_mt)."""
    from_replay = 20

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