# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import os
from enum import IntEnum, auto

# Query rows per forward pass, set by the top-level Makefile. The AIE kernels
# and the host read the same number as a -DL_TOKENS define, so all three
# languages move together.
L_TOKENS = int(os.environ.get("L_TOKENS", "8"))

class proj_pkt_id(IntEnum):
    """The engine's y tree. One-hot, and the values are load-bearing.

    Every id is the only one in the tree with its bit set, so the mask that
    splits it off a shared slave port is that bit alone and cannot match the
    others. Give two of them a shared bit and the rule generator either lets one
    swallow the other or refuses the port outright -- see the pkt_ids comment in
    proj_tile.py. Mirrors the pkt_id_to_* constants in aie_array_layout.h.
    """
    to_rms = 2    # o and down, -> MT[3]
    to_conv = 4   # qkv,        -> MT[5]
    to_aux = 8    # z,          -> MT[3]
    to_glu = 16   # up|gate,    -> CT[3][5]

class proj_x_pkt_id(IntEnum):
    """The id proj_x_replay_mt's replay channels stamp on their way to its own
    FIFO. It never leaves that tile.

    x reaches the 16 mvm cores on a circuit flow, so there is nothing for an id
    to disambiguate at the cores. The only merge left is inside the replay tile,
    where one replay channel per activation width feeds a single FIFO, and every
    one of them shares this id.

    The sharing is load-bearing. A distinct id per activation is its own packet
    flow, so the pathfinder grows each its own multicast tree; the trees diverge,
    and the arbiter allocator then ties one master port to two arbiters, which
    AIEDialect.cpp:2047 rejects. One id, one port, one tree per hop.
    """
    # Numerically the same as aux_to_proj_x_in_pkt_id.to_out_proj, which the
    # gate tile stamps *inbound* to that tile -- different ports, so the
    # collision is only in the value.
    from_replay = 2

class rms_pkt_id(IntEnum):
    """The ids the rms kernel stamps on its y rows, one per phase.

    The kernel writes them into the payload itself (rms_residual.cc), so the
    phase the tile is in picks the id:

      to_proj  phase 1, the input norm       -> MT[2], qkv|z's x
      to_st    phase 2, the post-attn norm   -> MT[2], up|gate's x
      to_out   phase 3, h2, no norm          -> IT[3], DDR

    Two *destination sets*, and that is the load-bearing part. Both norms land
    on the same MT[2] port (told apart by arrival order, not by id), so
    AIECreatePacketFlows merges to_proj and to_st into one group and the port
    carries exactly two masksets: west to MT[2], south to the shim. A *third*
    set off this master port does not route, whatever the id or the destination
    -- the allocator ties the port to two arbiters and AIEDialect.cpp:2047
    rejects it. That is why the post-attention norm has no DDR tap of its own.

    The values only have to be distinct; 4 rather than 2 keeps them clear of
    proj_x_pkt_id.from_replay, which shares the column.
    """
    to_proj = 0
    to_st = 1
    to_out = 4

class it1_pkt_id(IntEnum):
    to_rms  = 0

class aux_to_proj_x_in_pkt_id(IntEnum):
    """What the gate and GLU tiles stamp on their way into MT[2]'s x ports."""
    to_out_proj = 2    # the gated rows, o's x
    to_down_proj = 19  # silu(gate) * up, down's x

class aux_to_rms_pkt_id(IntEnum):
    """The id MT[3] stamps on o's and down's rows for their one hop north into
    the rms tile's x port.

    That port is a merge: the shim sends the layer input on it under
    it1_pkt_id.to_rms, MT[3] sends each residual's second operand under this one.
    Both land on the same S2MM channel and the rms tile cannot tell them apart --
    it does not need to, the phases are strictly ordered by the layer's data
    dependencies. The ids differ only so the router has one rule per source; 3
    keeps it out of rms_pkt_id's values, which share the column.
    """
    to_residual = 3

class ModelArch(IntEnum):
    QWEN3_8_27B = auto()

class QuantType(IntEnum):
    Q4_1 = auto()
    Q8_0 = auto()
    Q4_K = auto()
    
ModelArchNames: dict[ModelArch, str] = {
    ModelArch.QWEN3_8_27B: "QWEN3_8_27B"
}