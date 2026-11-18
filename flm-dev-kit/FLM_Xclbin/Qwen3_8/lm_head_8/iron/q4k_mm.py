# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils
from aie.utils.trace import MemTilePortEvent, MemTileEvent, PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from typing import List
from utils import mlir_context, tile_with_location, connect_dma, connect_port, lock_acquire, lock_release

# IT0's MM2S 0 carries column 0's second weight stream alongside the control
# traffic, so its flow is packet-switched; every other column gets a plain circuit
# flow on its own shim tile's MM2S 0. See lm_head_column.
#
# There is no X_PKT_ID at L = 8: x no longer rides a shared shim port at all, it
# goes into each column's memtile on that column's own channel.
W_PKT_ID = 1

# Ping-pong depth of each weight group in the memtile. Two is enough to keep both
# of a group's output channels busy; more just buys slack against jitter.
#
# Ping-pong depth of each weight group's memtile staging.
W_SLOTS_A = 2
W_SLOTS_B = 2

# k-block pairs per group-a memtile slot. A "unit" is one k-block for core 0
# followed by one for core 1, which is what the DDR layout already interleaves
# (reorder_cpy_q80), so widening a slot needs no host change: the slot simply
# spans UNITS_PER_SLOT_A consecutive units and the two MM2S channels stride
# through it.
#
# This exists to shorten the group-a S2MM ring. That ring has to tile exactly one
# round so it stays in phase with x, which costs 1 + K_BLOCKS/UNITS_PER_SLOT_A
# BDs. At 1 unit per slot that is 21 BDs and does not fit; at 2 it is 11.
#
# BD ids in a memtile are banked by channel parity -- even channels may only use
# ids 0-23 and odd channels 24-47 (AIETargetModel.h:469, isBdChannelAccessible).
# So the budget that matters is 24 per bank, not 48 overall, and a hand count of
# the total will mislead.
#
# It cannot simply be made as large as the banks allow: K_BLOCKS/UNITS_PER_SLOT_A
# must stay a multiple of W_SLOTS_A, or the output ring ends a round part-way
# through its own cycle and reads the next round's data out of the wrong buffer.
# 4 looks cheaper than 2 here and is wrong for exactly that reason -- 20/4 = 5
# slots per round against 2 buffers leaves the MM2S chains half a lap out at
# every round boundary. It fails late, not immediately: the first slot of round 1
# is correct, which is why the y buffer opens with the right three values and
# then diverges.
UNITS_PER_SLOT_A = 2

def lm_head_tile(tile: tile_with_location, mctx: mlir_context):

    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    qmm = mctx.qmm
    x_ty = mctx.x_ty
    w_ty = mctx.w_ty
    y_ty = mctx.y_ty

    # x is one column block wide now, not one whole row: at L = 8 a resident row
    # would be 80 KB against a 64 KB core memory, so x streams and ping/pongs the
    # same way w does. X_LANES * 256 rather than L * 256, because the last lane
    # carries model.norm.weight for the kernel to fold in as it loads x. See the
    # header comment in kernel/qmm.cc.
    x_buffer_0 = aie.buffer(tile = ct, datatype = x_ty, name = f"x_0_{ct_row}_{ct_col}")
    x_buffer_1 = aie.buffer(tile = ct, datatype = x_ty, name = f"x_1_{ct_row}_{ct_col}")
    w_buffer_0 = aie.buffer(tile = ct, datatype = w_ty, name = f"w_0_{ct_row}_{ct_col}")
    w_buffer_1 = aie.buffer(tile = ct, datatype = w_ty, name = f"w_1_{ct_row}_{ct_col}")
    y_buffer_0 = aie.buffer(tile = ct, datatype = y_ty, name = f"y_0_{ct_row}_{ct_col}")
    y_buffer_1 = aie.buffer(tile = ct, datatype = y_ty, name = f"y_1_{ct_row}_{ct_col}")

    x_prod_lock = aie.lock(ct, lock_id = 0, init = 2)
    x_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    w_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    w_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    y_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    y_cons_lock = aie.lock(ct, lock_id = 5, init = 0)

    @aie.core(ct, link_with = "qmm.o", stack_size = 1024 * 8)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            qmm(y_buffer_0, y_buffer_1, w_buffer_0, w_buffer_1, x_buffer_0, x_buffer_1)

    @aie.mem(ct)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

        with block[1]:
            lock_acquire(x_prod_lock, 1)
            aie.dma_bd(x_buffer_0)
            lock_release(x_cons_lock, 1)
            aie.next_bd(block[2])

        with block[2]:
            lock_acquire(x_prod_lock, 1)
            aie.dma_bd(x_buffer_1)
            lock_release(x_cons_lock, 1)
            aie.next_bd(block[1])

        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])

        with block[4]:
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w_buffer_0)
            lock_release(w_cons_lock, 1)
            aie.next_bd(block[5])
        
        with block[5]:
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w_buffer_1)
            lock_release(w_cons_lock, 1)
            aie.next_bd(block[4])
            
        with block[6]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[7], chain=block[9])

        with block[7]:
            lock_acquire(y_cons_lock, 1)
            aie.dma_bd(y_buffer_0)
            lock_release(y_prod_lock, 1)
            aie.next_bd(block[8])

        with block[8]:
            lock_acquire(y_cons_lock, 1)
            aie.dma_bd(y_buffer_1)
            lock_release(y_prod_lock, 1)
            aie.next_bd(block[7])
        
        with block[9]:
            aie.EndOp()

def _build_dma_chains(block, chains):
    """Emit a memtile/mem DMA body from a flat description of its channels.

    ``chains`` is a list of ``(direction, channel, bds)`` or
    ``(direction, channel, bds, n_prologue)`` where each bd is
    ``(buffer, offset, length, acquire_lock, release_lock)`` and ``length = None``
    means the whole buffer. The channels are chained together the way
    aie.dma_start's ``chain`` argument wants. Writing this out by hand is 60-odd
    near-identical blocks; the two weight groups below are structurally identical,
    so it is worth generating.

    The first ``n_prologue`` bds of a channel run exactly once; the rest form the
    ring that repeats forever. The ring closes onto bd ``n_prologue``, so the
    prologue is never revisited. That is what lets the one-shot x fill share the
    group-a weight S2MM channel -- all six memtile S2MM channels are already
    committed (two weights, four y), and x is 90 KB against 169 MB of weights, so
    it is far cheaper to prepend it here than to give up a weight channel.
    """
    # normalise to 4-tuples so callers can omit n_prologue
    chains = [c if len(c) == 4 else (*c, 0) for c in chains]

    starts = [0]
    bd_ids = []
    next_id = 1
    for _, _, bds, _ in chains:
        bd_ids.append(list(range(next_id, next_id + len(bds))))
        next_id += len(bds)
        starts.append(next_id)
        next_id += 1
    end_id = starts.pop()

    for ci, (direction, channel, bds, n_prologue) in enumerate(chains):
        nxt = block[starts[ci + 1]] if ci + 1 < len(chains) else block[end_id]

        def emit_start():
            aie.dma_start(
                channel_dir = direction,
                channel_index = channel,
                dest = block[bd_ids[ci][0]],
                chain = nxt,
            )

        if ci == 0:
            # block[0] is the body's entry block, which is already open.
            emit_start()
        else:
            with block[starts[ci]]:
                emit_start()

        for bi, (buf, offset, length, acq, rel) in enumerate(bds):
            with block[bd_ids[ci][bi]]:
                # A lock may be given as (lock, n) to move n credits at once. That
                # is how a wide staging slot is handed to the several BDs that
                # read it: the filler releases n, each reader takes one.
                #
                # Both an acquire and a release are mandatory on a memtile bd --
                # AIERT.cpp:304 asserts otherwise -- so there is no "no handshake
                # on this edge" option here.
                acq_lock, acq_n = acq if isinstance(acq, tuple) else (acq, 1)
                rel_lock, rel_n = rel if isinstance(rel, tuple) else (rel, 1)
                lock_acquire(acq_lock, acq_n)
                if length is None:
                    aie.dma_bd(buf)
                else:
                    aie.dma_bd(buf, offset = offset, len = length)
                lock_release(rel_lock, rel_n)
                # The ring wraps onto n_prologue, not 0, so any prologue bds run
                # once and are then stepped over forever.
                nxt_bi = bi + 1
                if nxt_bi == len(bds):
                    nxt_bi = n_prologue
                aie.next_bd(block[bd_ids[ci][nxt_bi]])

    with block[end_id]:
        aie.EndOp()


def lm_head_memtile(tile: tile_with_location, mctx: mlir_context):

    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    w_mem_ty  = mctx.w_mem_ty
    wa_mem_ty = mctx.wa_mem_ty
    y_mem_ty  = mctx.y_mem_ty
    x_mem_ty  = mctx.x_mem_ty

    W_BLOCK_SIZE        = mctx.W_BLOCK_SIZE
    Q4NX_ROW_BLOCK_SIZE = mctx.Q4NX_ROW_BLOCK_SIZE
    X_BLOCK_SIZE        = mctx.X_BLOCK_SIZE
    K_BLOCKS            = mctx.K_BLOCKS
    ROUNDS              = mctx.ROUNDS
    L_TOKENS            = mctx.L_TOKENS

    # The four cores of a column are fed by two independent weight streams rather
    # than one: a single shim->memtile channel saturates at roughly a quarter of what
    # the four cores can consume, which is what made this kernel purely data-movement
    # bound. Group a serves cores 0/1, group b serves cores 2/3, and each group has
    # its own shim MM2S channel behind it. A slot is therefore two k-blocks wide, not
    # four, and the host lays the weights out in DDR as two contiguous halves per
    # column-round to match (see reorder_cpy in host/host.cpp).
    # Group a's slot is UNITS_PER_SLOT_A units wide rather than one, purely to
    # keep its S2MM ring short enough to fit the even BD bank -- see the constant.
    w_a = [aie.buffer(tile = mt, datatype = wa_mem_ty, name = f"wa_{i}_{mt_row}_{mt_col}") for i in range(W_SLOTS_A)]
    w_b = [aie.buffer(tile = mt, datatype = w_mem_ty, name = f"wb_{i}_{mt_row}_{mt_col}") for i in range(W_SLOTS_B)]
    y_buffers = [
        aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_0_{mt_row}_{mt_col}"),
        aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_1_{mt_row}_{mt_col}"),
    ]
    # All L rows of x for this column plus model.norm.weight, [k_block][lane][256]
    # with X_LANES = L + 1 lanes. Refilled once per ROUND rather than once per
    # dispatch: the host re-sends x every round so that x is an ordinary streamed
    # operand with the same lifetime as a weight slot.
    # 9 * 5120 * 2 B = 90 KB of the memtile's 512 KB.
    #
    # The norm weight is a lane of x rather than a buffer of its own for the same
    # reason x is per-round: it would otherwise need a one-shot fill, and a
    # one-shot fill is precisely the bug below.
    #
    # The one-shot prologue this replaces could not survive a second dispatch.
    # Its fill BD ran once and the ring then stepped over it forever, so on
    # dispatch 2 the host's x transfer landed in the *weight* BDs behind it and
    # was dequantised as Q8_0 -- NaN/Inf logits, with the command still reporting
    # COMPLETED. A per-round slot has no such edge: every dispatch is just more
    # rounds, and there is no state that has to be rewound between them.
    x_buffer = aie.buffer(tile = mt, datatype = x_mem_ty, name = f"x_{mt_row}_{mt_col}")

    # Each group hands a slot down its two output channels in turn and only then
    # returns it to the producer, so a slot is never refilled while either core is
    # still reading it.
    # Group a counts in per-BD credits rather than slots: a slot is handed to
    # UNITS_PER_SLOT_A reader BDs on each output channel, so the free pool holds
    # W_SLOTS_A * UNITS_PER_SLOT_A. Still far below the 63 lock ceiling.
    wa_prod_lock   = aie.lock(mt, lock_id = 0, init = W_SLOTS_A * UNITS_PER_SLOT_A)
    wa_cons_lock_0 = aie.lock(mt, lock_id = 1, init = 0)
    wa_cons_lock_1 = aie.lock(mt, lock_id = 2, init = 0)
    wb_prod_lock   = aie.lock(mt, lock_id = 3, init = W_SLOTS_B)
    wb_cons_lock_0 = aie.lock(mt, lock_id = 4, init = 0)
    wb_cons_lock_1 = aie.lock(mt, lock_id = 5, init = 0)
    y_prod_lock_0 = aie.lock(mt, lock_id = 6, init = 2)
    y_prod_lock_1 = aie.lock(mt, lock_id = 7, init = 0)
    y_prod_lock_2 = aie.lock(mt, lock_id = 8, init = 0)
    y_prod_lock_3 = aie.lock(mt, lock_id = 9, init = 0)
    y_cons_lock   = aie.lock(mt, lock_id = 10, init = 0)
    # x is now an ordinary per-round producer/consumer pair, exactly like a weight
    # group: the shim fills a slot, the replay channel drains it to the cores and
    # hands the slot back. Both counts stay within 1, well under the 63 lock
    # ceiling (up-gate-must-be-resident-lock-63), because the handshake is once per
    # ROUND rather than once per replayed chunk.
    x_prod_lock = aie.lock(mt, lock_id = 11, init = 1)
    x_cons_lock = aie.lock(mt, lock_id = 12, init = 0)

    # A core's y contribution is L * 32, not 32: it emits [token][32] for its one
    # 32-row weight block. The four cores' slices sit back to back, so the shared
    # buffer is [core][token][32] and the shim sees [round][col][core][token][32].
    Y = L_TOKENS * Q4NX_ROW_BLOCK_SIZE
    X = X_BLOCK_SIZE

    # x and group-a weights share shim channel 1, and the ring must tile exactly
    # one ROUND of that channel's traffic: ONE x BD followed by K_BLOCKS weight
    # BDs. The ratio is not 1:1 -- the host sends one x transfer and one *weight*
    # transfer per round, but a round's weight transfer is K_BLOCKS memtile slots
    # wide while x is a single slot. A 1:1 ring drifts by K_BLOCKS-1 slots every
    # round, which hands x's 90 KB to a weight BD to be dequantised as Q8_0: NaN
    # logits, ERT state 8, and a 60 s dispatch. That was the first attempt here.
    #
    # The K_BLOCKS weight BDs cycle over only W_SLOTS_A distinct buffers, so this
    # costs BDs but not memory.
    #
    # This is what makes the design re-dispatchable. The previous layout put x in
    # a one-shot PROLOGUE bd that the ring stepped over after the first pass, so a
    # second dispatch's x fell through into the weight BDs -- same corruption, but
    # only from dispatch 2 onward. Here there is no prologue and no per-dispatch
    # state: round 244 is indistinguishable from round 4.
    assert K_BLOCKS % UNITS_PER_SLOT_A == 0, "group-a slot must tile a round"
    SLOTS_PER_ROUND_A = K_BLOCKS // UNITS_PER_SLOT_A
    # ...and a round must be a whole number of laps of the buffer cycle, or the
    # output channels start round r+1 mid-cycle and read the wrong buffer.
    assert SLOTS_PER_ROUND_A % W_SLOTS_A == 0, "group-a ring must close per round"
    # The filler takes UNITS_PER_SLOT_A credits (one per BD that will read the
    # slot) and releases the same number downstream, so the counts balance at
    # every edge of the group-a ring.
    U = UNITS_PER_SLOT_A
    xa_ring = [(x_buffer, 0, None, x_prod_lock, x_cons_lock)]
    for i in range(SLOTS_PER_ROUND_A):
        xa_ring.append((w_a[i % W_SLOTS_A], 0, None, (wa_prod_lock, U), (wa_cons_lock_0, U)))

    # Channel assignment is constrained by the BD banks, not just by count. A
    # memtile's 48 BDs are split into two fixed halves: EVEN channels may only use
    # ids 0-23 and ODD channels only 24-47 (AIETargetModel.h:469,
    # isBdChannelAccessible). So "42 BDs total, under 48" is not a sufficient
    # check -- the 21-BD x+wa ring plus the odd-channel weight and y chains came
    # to 31 in the odd bank while the even bank held 11, and aiecc refused it.
    #
    # The big ring therefore goes on EVEN S2MM 0, and group b, which is small,
    # takes the odd S2MM 1. Both banks then sit at 21/24 and 11/24.
    chains = [
        # x + weight group a: in from the shim (even channel, for the bank), and
        # the weights out to cores 0 and 1.
        (aie.DMAChannelDir.S2MM, 0, xa_ring),
        # A wide slot holds UNITS_PER_SLOT_A units of [core0 k-block | core1
        # k-block], so each output channel walks its own k-blocks at a stride of
        # one unit.
        #
        # Every memtile BD must carry BOTH an acquire and a release -- AIERT.cpp
        # asserts on a bd with only one, so "lock on the first BD, unlock on the
        # last" is not expressible. Instead the filler releases UNITS_PER_SLOT_A
        # credits for the slot and each BD takes exactly one, which gives the same
        # mutual exclusion with a per-BD handshake.
        (aie.DMAChannelDir.MM2S, 1, [
            (b, u * 2 * W_BLOCK_SIZE, W_BLOCK_SIZE, wa_cons_lock_0, wa_cons_lock_1)
            for b in w_a for u in range(UNITS_PER_SLOT_A)]),
        (aie.DMAChannelDir.MM2S, 2, [
            (b, (u * 2 + 1) * W_BLOCK_SIZE, W_BLOCK_SIZE, wa_cons_lock_1, wa_prod_lock)
            for b in w_a for u in range(UNITS_PER_SLOT_A)]),
        # x replay: one BD per slot, each streaming that slot's whole 90 KB to all
        # four cores of the column -- exactly the K_BLOCKS column blocks one round
        # consumes. x is laid out [k_block][lane][256], contiguous in the order
        # the k-loop wants, so splitting it into column blocks is the *consumer's*
        # business and the cores' own 2048-element ping/pong BDs do it. One BD per
        # k-block would be 20 per slot and would blow the 48-BD allocator.
        #
        # Unlike the old free-running ring this one handshakes: it takes a filled
        # slot and returns it, so the shim cannot overwrite a slot that is still
        # being replayed, and the cores cannot run ahead into a slot that has not
        # been filled yet.
        (aie.DMAChannelDir.MM2S, 0,
            [(x_buffer, 0, None, x_cons_lock, x_prod_lock)]),
        # Weight group b: in from the shim on the odd channel, out to cores 2/3.
        (aie.DMAChannelDir.S2MM, 1, [(b, 0, None, wb_prod_lock, wb_cons_lock_0) for b in w_b]),
        (aie.DMAChannelDir.MM2S, 3, [(b, 0, W_BLOCK_SIZE, wb_cons_lock_0, wb_cons_lock_1) for b in w_b]),
        (aie.DMAChannelDir.MM2S, 4, [(b, W_BLOCK_SIZE, W_BLOCK_SIZE, wb_cons_lock_1, wb_prod_lock) for b in w_b]),
        # y: each core drops its 16 logits into its own quarter of the shared buffer,
        # in core order, and the last one hands the full buffer to the shim.
        (aie.DMAChannelDir.S2MM, 2, [(b, 0,     Y, y_prod_lock_0, y_prod_lock_1) for b in y_buffers]),
        (aie.DMAChannelDir.S2MM, 3, [(b, Y,     Y, y_prod_lock_1, y_prod_lock_2) for b in y_buffers]),
        (aie.DMAChannelDir.S2MM, 4, [(b, Y * 2, Y, y_prod_lock_2, y_prod_lock_3) for b in y_buffers]),
        (aie.DMAChannelDir.S2MM, 5, [(b, Y * 3, Y, y_prod_lock_3, y_cons_lock)   for b in y_buffers]),
        (aie.DMAChannelDir.MM2S, 5, [(b, 0, None, y_cons_lock, y_prod_lock_0) for b in y_buffers]),
    ]

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        _build_dma_chains(block, chains)


def lm_head_column(CTs: List[tile_with_location], MT: tile_with_location, IT: tile_with_location, mctx: mlir_context, w_pkt_id: int = -1):

    lm_head_memtile(MT, mctx)

    # Both of this column's shim MM2S channels carry weights, and shim channel 1
    # also carries one x slot per round, ahead of each group-a weight transfer.
    #
    # The shim channel roles are unchanged from the L = 1 design -- shim 1 is
    # x + group a, shim 0 is group b -- but they now land on the OPPOSITE memtile
    # channels, because a memtile's BD ids are banked by channel parity and the
    # 21-BD x+wa ring only fits in the even bank (see lm_head_memtile).
    #
    # x is NOT broadcast from a shim port any more. At L = 8 it has to be re-sent
    # per round, and the L = 1 design shared IT[0].MM2S[0] between the x
    # broadcast and column 0's group-b weights. A broadcast advances only as fast
    # as its slowest consumer, so cores 2/3 of column 0 would be waiting on
    # weights queued behind x on that very port while the port waited on them --
    # a dependency cycle. It is safe at L = 1 only because x is sent once, before
    # any weight transfer is queued. Each column now takes its own copy of x into
    # its memtile, which sinks the whole 90 KB and frees the shim port.
    connect_dma(IT, MT, 1, 0)
    connect_dma(IT, MT, 0, 1, pkt_id = w_pkt_id)

    for i, proj_core in enumerate(CTs):
        lm_head_tile(proj_core, mctx)
        # x: memtile MM2S 0, replayed per k-block to all four cores.
        connect_dma(MT, proj_core, 0, 0)
        connect_dma(MT, proj_core, i + 1, 1)

        connect_dma(proj_core, MT, 0, i + 2)

    connect_dma(MT, IT, 5, 0)
