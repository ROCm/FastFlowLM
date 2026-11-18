# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""dequant_mm DMA chains and locks (from the former ../fused_prefill, deleted 2026-11-12), on arenas.

Same chains, locks, lengths and dimensions as the original; every buffer is now
(arena, offset) from layout.py instead of its own aie.buffer. Only built into the
"dq" variant, whose configuration the host replays.
"""
import aie.dialects.aie as aie

from utils import lock_acquire, lock_release
from layout import CT_DQ_E, CT_DQ_O, MT_DQ, elems

# kernel/dq/common.hpp: NPU_M_TILE 32, NPU_K_TILE 512, NPU_N_TILE 128
m, k, n = 32, 512, 128
m_slice, k_slice, n_slice, y_slice = 16, 256, 32, 64
QW_BYTES = 4736                                 # one q4_k 32 x 256 block
QW_MEM_BYTES = (k // k_slice) * (n // n_slice) * QW_BYTES   # 8 blocks per mem tile buffer: 37888 B
X_SENDS = k // k_slice                          # x crosses to the cores once per k_slice
# x is bfp16 tiles made by the host: per 16 tokens, [Din / 8][2 row blocks][72 B]. One core
# x buffer = 16 tokens x k_slice = 4608 B, already in the order the kernel reads it.
X_GROUP = 16
X_TILE_BYTES = X_GROUP * k_slice * 9 // 8        # 4608 B


def bd(arena, off_bytes, length, dims=None):
    """aie.dma_bd on a bf16 arena at a byte offset; length in bf16 elements"""
    if dims is None:
        aie.dma_bd(arena, offset=elems(off_bytes), len=length)
    else:
        aie.dma_bd(arena, offset=elems(off_bytes), len=length, dimensions=dims)


def dq_core_pair(arena_e, arena_o, tile_e, tile_o):
    """chains + locks of one (even, odd) core pair"""

    x_prod = aie.lock(tile_e, lock_id=0, init=2)
    x_cons = aie.lock(tile_e, lock_id=1, init=0)
    qw_prod = aie.lock(tile_e, lock_id=2, init=4)
    qw_cons = aie.lock(tile_e, lock_id=3, init=0)
    y_prod = aie.lock(tile_e, lock_id=4, init=2)
    y_cons = aie.lock(tile_e, lock_id=5, init=0)
    aie.lock(tile_e, lock_id=6, init=0)      # sync, used by the cores only
    x_prod_o = aie.lock(tile_o, lock_id=0, init=2)
    x_cons_o = aie.lock(tile_o, lock_id=1, init=0)
    y_prod_o = aie.lock(tile_o, lock_id=4, init=2)
    y_cons_o = aie.lock(tile_o, lock_id=5, init=0)
    aie.lock(tile_o, lock_id=6, init=0)      # sync

    E = CT_DQ_E

    @aie.mem(tile_e)
    def mem_e(block):
        aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
        for i, buf in enumerate(("x0", "x1")):
            with block[1 + i]:
                lock_acquire(x_prod, 1)
                bd(arena_e, E[buf], X_TILE_BYTES // 2)
                lock_release(x_cons, 1)
                aie.next_bd(block[2 - i])
        with block[3]:
            aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest=block[4], chain=block[6])
        for i, buf in enumerate(("qw0", "qw1")):
            with block[4 + i]:
                lock_acquire(qw_prod, 2)
                bd(arena_e, E[buf], QW_BYTES // 2)
                lock_release(qw_cons, 2)
                aie.next_bd(block[5 - i])
        with block[6]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest=block[7], chain=block[9])
        for i, buf in enumerate(("y0", "y1")):
            with block[7 + i]:
                lock_acquire(y_cons, 1)
                bd(arena_e, E[buf], y_slice)
                lock_release(y_prod, 1)
                aie.next_bd(block[8 - i])
        with block[9]:
            aie.EndOp()

    O = CT_DQ_O

    @aie.mem(tile_o)
    def mem_o(block):
        aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
        for i, buf in enumerate(("x0", "x1")):
            with block[1 + i]:
                lock_acquire(x_prod_o, 1)
                bd(arena_o, O[buf], X_TILE_BYTES // 2)
                lock_release(x_cons_o, 1)
                aie.next_bd(block[2 - i])
        with block[3]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest=block[4], chain=block[6])
        for i, buf in enumerate(("y0", "y1")):
            with block[4 + i]:
                lock_acquire(y_cons_o, 1)
                bd(arena_o, O[buf], y_slice)
                lock_release(y_prod_o, 1)
                aie.next_bd(block[5 - i])
        with block[6]:
            aie.EndOp()


def dq_memtile(arena, tile, m=m, split=False):
    """mem tile chains + locks: x to a core row, y from 4 cores, q4_k weights

    m      tokens per core row per round (32, or 16 for the 32-row plan)
    split  the two core pairs of the column get different weights: the shim
           sends two 128-row lanes interleaved per 256-column step, A (first
           lane) goes out on MM2S 2 to core row 0, B on MM2S 3 to core row 2
    """
    M = MT_DQ
    x_prod = aie.lock(tile, lock_id=0, init=2)
    x_cons = [aie.lock(tile, lock_id=1 + i, init=0) for i in range(X_SENDS)]
    y_base = 1 + X_SENDS
    y_prod = [aie.lock(tile, lock_id=y_base, init=2)] + \
             [aie.lock(tile, lock_id=y_base + i, init=0) for i in (1, 2, 3)]
    y_cons = aie.lock(tile, lock_id=y_base + 4, init=0)
    qw_prod = aie.lock(tile, lock_id=y_base + 5, init=2)
    qw_cons = aie.lock(tile, lock_id=y_base + 6, init=0)
    if split:
        qwb_prod = aie.lock(tile, lock_id=y_base + 7, init=2)
        qwb_cons = aie.lock(tile, lock_id=y_base + 8, init=0)

    y_in_dims = [(m // 8, 8 * n), (n // 8, 8), (8, n), (8, 1)]

    b_x_mm2s = 4
    b_y_s2mm = b_x_mm2s + 2 * X_SENDS
    b_y_mm2s = b_y_s2mm + 12
    b_qw_s2mm = b_y_mm2s + 3
    b_qw_mm2s = b_qw_s2mm + (5 if split else 3)
    b_qwb_mm2s = b_qw_mm2s + 3
    b_end = b_qwb_mm2s + (3 if split else 0)

    @aie.memtile_dma(tile)
    def body(block):
        # x in from the shim: one whole (m, k) tile per BD, bfp16, as [m / 16][k / k_slice][4608 B]
        groups = m // X_GROUP
        aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
        for i, buf in enumerate(("x0", "x1")):
            with block[1 + i]:
                lock_acquire(x_prod, 1)
                bd(arena, M[buf], groups * X_SENDS * X_TILE_BYTES // 2)
                lock_release(x_cons[0], 1)
                aie.next_bd(block[2 - i])
        # x out to the cores of this row: send j = slice j of every 16-token group
        # (the 4608 B run is split 9 x 512 B: a dimension holds at most 1023 words)
        with block[3]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest=block[b_x_mm2s], chain=block[b_y_s2mm])
        for b, buf in enumerate(("x0", "x1")):
            for send in range(X_SENDS):
                seq = b * X_SENDS + send
                with block[b_x_mm2s + seq]:
                    lock_acquire(x_cons[send], 1)
                    bd(arena, M[buf] + send * X_TILE_BYTES, groups * X_TILE_BYTES // 2,
                       [(groups, X_SENDS * X_TILE_BYTES // 2), (9, X_TILE_BYTES // 18), (X_TILE_BYTES // 18, 1)])
                    lock_release(x_prod if send == X_SENDS - 1 else x_cons[send + 1], 1)
                    aie.next_bd(block[b_x_mm2s + (seq + 1) % (2 * X_SENDS)])
        # y in from the 4 cores of a row (S2MM 1..4), each a quarter of the slab
        for c in range(4):
            head = b_y_s2mm + 3 * c
            nxt = block[head + 3] if c < 3 else block[b_y_mm2s]
            with block[head]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 1 + c, dest=block[head + 1], chain=nxt)
            acq = y_prod[c]
            rel = y_prod[c + 1] if c < 3 else y_cons
            for i, buf in enumerate(("y0", "y1")):
                with block[head + 1 + i]:
                    lock_acquire(acq, 1)
                    bd(arena, M[buf] + 2 * c * m * n, m * n, y_in_dims)
                    lock_release(rel, 1)
                    aie.next_bd(block[head + 2 - i])
        # y out to the shim: the 4 column slabs interleaved into (m, 4n)
        with block[b_y_mm2s]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest=block[b_y_mm2s + 1], chain=block[b_qw_s2mm])
        for i, buf in enumerate(("y0", "y1")):
            with block[b_y_mm2s + 1 + i]:
                lock_acquire(y_cons, 1)
                bd(arena, M[buf], 4 * m * n, [(m, n), (4, m * n), (n, 1)])
                lock_release(y_prod[0], 1)
                aie.next_bd(block[b_y_mm2s + 2 - i])
        # q4_k weights: straight relay, one (k, n) tile per buffer
        if not split:
            with block[b_qw_s2mm]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 5, dest=block[b_qw_s2mm + 1], chain=block[b_qw_mm2s])
            for i, buf in enumerate(("qw0", "qw1")):
                with block[b_qw_s2mm + 1 + i]:
                    lock_acquire(qw_prod, 1)
                    bd(arena, M[buf], QW_MEM_BYTES // 2)
                    lock_release(qw_cons, 1)
                    aie.next_bd(block[b_qw_s2mm + 2 - i])
            with block[b_qw_mm2s]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 2, dest=block[b_qw_mm2s + 1], chain=block[b_end])
            for i, buf in enumerate(("qw0", "qw1")):
                with block[b_qw_mm2s + 1 + i]:
                    lock_acquire(qw_cons, 1)
                    bd(arena, M[buf], QW_MEM_BYTES // 2)
                    lock_release(qw_prod, 1)
                    aie.next_bd(block[b_qw_mm2s + 2 - i])
        else:
            # split: each qw buffer holds A then B, one 256-column step each
            lane = QW_MEM_BYTES // 2
            order = [("qw0", 0, qw_prod, qw_cons), ("qw0", lane, qwb_prod, qwb_cons),
                     ("qw1", 0, qw_prod, qw_cons), ("qw1", lane, qwb_prod, qwb_cons)]
            with block[b_qw_s2mm]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 5, dest=block[b_qw_s2mm + 1], chain=block[b_qw_mm2s])
            for i, (buf, off, prod, cons) in enumerate(order):
                with block[b_qw_s2mm + 1 + i]:
                    lock_acquire(prod, 1)
                    bd(arena, M[buf] + off, lane // 2)
                    lock_release(cons, 1)
                    aie.next_bd(block[b_qw_s2mm + 1 + (i + 1) % 4])
            for ch, head, off, prod, cons, nxt in ((2, b_qw_mm2s, 0, qw_prod, qw_cons, b_qwb_mm2s),
                                                   (3, b_qwb_mm2s, lane, qwb_prod, qwb_cons, b_end)):
                with block[head]:
                    aie.dma_start(aie.DMAChannelDir.MM2S, ch, dest=block[head + 1], chain=block[nxt])
                for i, buf in enumerate(("qw0", "qw1")):
                    with block[head + 1 + i]:
                        lock_acquire(cons, 1)
                        bd(arena, M[buf] + off, lane // 2)
                        lock_release(prod, 1)
                        aie.next_bd(block[head + 2 - i])
        with block[b_end]:
            aie.EndOp()
