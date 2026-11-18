# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import os
from enum import IntEnum, auto

# Query rows per forward pass. The MTP draft head is decode-shape only -- one
# token per step -- so this is 1 here, unlike ../gated_attention_verify, which
# runs 8. The AIE kernels and the host read the same number as a -DL_TOKENS
# define, so all three languages move together.
L_TOKENS = int(os.environ.get("L_TOKENS", "1"))


class proj_pkt_id(IntEnum):
    """Where a projection's output goes, stamped by the mvm core.

    One-hot: the switchboxes demux several logical streams off one physical
    MM2S by masking on these, so two ids sharing a bit would both match.
    """
    # fc_out *and* down, both north to the rms tile's x FIFO. They share an id
    # because they share a destination channel and the tile's two-BD ping/pong
    # chain tells them apart by arrival order -- the same trade z and o make
    # below, and the same one the tile's own two norms make on the way out.
    #
    # **down had an id of its own, to_out = 1, while it was a DDR tap, and
    # reusing it here would hang the design.** Its route now ends at CT[0][2]
    # rather than at IT[2], so it enters column 2's switchbox from the south --
    # where it1_pkt_id.to_rope, also 1, is already passing through northbound to
    # the RoPE tile. One id, one slave port, two destinations is what a
    # switchbox cannot express. Sharing to_rms costs no new rule anywhere,
    # because fc_out already walks this exact route.
    to_rms  = 2   # fc_out, then down -> the rms/residual tile
    to_rope = 4   # q|k|v           -> the RoPE tile
    # z *and* o, both to MT[2]. They share an id because they share a
    # destination channel and the memtile's BD chain tells them apart by arrival
    # order -- z lands in the gate buffers, o in the out buffers behind them.
    # Giving o an id of its own would cost a second rule on every switchbox the
    # y tree traverses and buy nothing.
    to_aux  = 8   # z, then o       -> the gate's memtile
    to_glu  = 16  # up|gate         -> the GLU tile


class proj_x_pkt_id(IntEnum):
    """Ids stamped inbound to the projection engine's x FIFO."""
    from_passthrough = 2

class pass_pkt_id(IntEnum):
    to_proj = 0
    
class rms_pkt_id(IntEnum):
    """What the rms tile is emitting on its Y port.

    Two destination sets on one master port: west to the projection engine for
    the two norms the mvm consumes (phases 1 and 2, told apart downstream by
    arrival order), and south to IT[2] for h2, the layer output (phase 3).
    Distinct bits, since both leave the same port.
    """
    to_proj = 0
    to_st   = 1
    to_out  = 4


class aux_to_rms_pkt_id(IntEnum):
    """o, from MT[2] north to the rms tile's x FIFO -- the attention residual.

    **Bit 5, and not a small id.** This packet enters the rms tile's south port,
    which already carries it1_pkt_id.to_rms (0, input_layernorm's weight) and
    it1_pkt_id.to_rope (1, passing through to the RoPE tile above). The router
    widens a rule's mask to cover the ids leaving a port in the same direction,
    so an id sharing a bit with either of those could swallow their packets --
    and the weight one lands on a *different* channel of the same tile, which is
    the case a mis-set mask gets wrong silently.
    """
    to_residual = 32


class it1_pkt_id(IntEnum):
    to_rms  = 0
    to_rope = 1
    to_pass = 2


class kv_out_pkt_id(IntEnum):
    """Which attention column *and which plane* a k|v half is bound for.

    Stamped by the RoPE tile. One id per column, not per CU: the two heads a
    column owns travel together and its memtile splits them. The append has to
    land on that column's own shim so npu_dma_wait can order it against the
    replay -- see iron/RoPE.py.

    **Four ids, not two: k and v are separate flows now.** They land on
    different S2MM channels of the same shim, which is what lets the host walk
    each plane with its own BD instead of describing both in one descriptor
    whose "k or v" stride is the whole k plane. That stride was a 20-bit field
    and it capped the cache near MAX_L = 2048 -- silently, since
    npu_instr_utils prints and masks rather than faulting.

    The layout of the four is load-bearing. Both columns' packets share the
    physical link east out of column 2, and column 4's switchbox has to drop
    12 and 14 while passing 13 and 15 through to column 5. Even ids go to
    column 0, odd to column 1, exactly as ../gated_attention_verify does with
    4..7 -- a router that widens a rule's mask over the block still splits it
    on the low bit. Ids 12..15 rather than that design's 4..7 because this
    tree's projection engine sits in columns 6 and 7 and its westbound flows
    (to_rms 2, to_rope 4, to_aux 8, to_glu 16) transit these same switchboxes.
    """
    k_to_col_0 = 12
    k_to_col_1 = 13
    v_to_col_0 = 14
    v_to_col_1 = 15


class aux_to_proj_x_in_pkt_id(IntEnum):
    to_out_proj  = 2
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
