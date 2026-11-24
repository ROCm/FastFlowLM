# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""dequant_mm (FLM_Xclbin/dequant_mm_512x512x512) DMA chains and locks, on arenas.

Same chains, locks and dimensions as the original dequant_mm_tile.py; every
buffer is now (arena, offset) from layout.py instead of its own aie.buffer.
Only built into the "dq" variant (its configuration is replayed by the host).
"""
import aie.dialects.aie as aie

from utils import lock_acquire, lock_release
from layout import CT_DQ_E, CT_DQ_O, MT_DQ, elems

# shapes (bf16 elements unless noted): dequant_mm_aie.py's with M and K halved
# (kernel/dq/common.hpp NPU_M_TILE 64, NPU_K_TILE 256)
m, k, n = 64, 256, 64
m_slice, k_slice, n_slice, y_slice = 16, 256, 32, 64
QW_BYTES = int(k_slice * n_slice * 4.625 // 8)        # one q4_k block: 4736 B
QW_MEM_BYTES = int(k * n * 4.625 // 8)                # 2 blocks per mem tile buffer
X_CT = m_slice * 128                                  # x slice per core buffer


def bd(arena, off_bytes, length, dims=None):
    """aie.dma_bd on a bf16 arena at a byte offset"""
    if dims is None:
        aie.dma_bd(arena, offset=elems(off_bytes), len=length)
    else:
        aie.dma_bd(arena, offset=elems(off_bytes), len=length, dimensions=dims)


def dq_core_pair(arena_e, arena_o, tile_e, tile_o):
    """chains + locks of one (even, odd) core pair"""
    x_dims = [(m_slice, 8), (128 // 8, m_slice * 8), (8, 1)]

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
        with block[1]:
            lock_acquire(x_prod, 1)
            bd(arena_e, E["x0"], X_CT, x_dims)
            lock_release(x_cons, 1)
            aie.next_bd(block[2])
        with block[2]:
            lock_acquire(x_prod, 1)
            bd(arena_e, E["x1"], X_CT, x_dims)
            lock_release(x_cons, 1)
            aie.next_bd(block[1])
        with block[3]:
            aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest=block[4], chain=block[6])
        with block[4]:
            lock_acquire(qw_prod, 2)
            bd(arena_e, E["qw0"], QW_BYTES // 2)
            lock_release(qw_cons, 2)
            aie.next_bd(block[5])
        with block[5]:
            lock_acquire(qw_prod, 2)
            bd(arena_e, E["qw1"], QW_BYTES // 2)
            lock_release(qw_cons, 2)
            aie.next_bd(block[4])
        with block[6]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest=block[7], chain=block[9])
        with block[7]:
            lock_acquire(y_cons, 1)
            bd(arena_e, E["y0"], y_slice)
            lock_release(y_prod, 1)
            aie.next_bd(block[8])
        with block[8]:
            lock_acquire(y_cons, 1)
            bd(arena_e, E["y1"], y_slice)
            lock_release(y_prod, 1)
            aie.next_bd(block[7])
        with block[9]:
            aie.EndOp()

    O = CT_DQ_O

    @aie.mem(tile_o)
    def mem_o(block):
        aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
        with block[1]:
            lock_acquire(x_prod_o, 1)
            bd(arena_o, O["x0"], X_CT, x_dims)
            lock_release(x_cons_o, 1)
            aie.next_bd(block[2])
        with block[2]:
            lock_acquire(x_prod_o, 1)
            bd(arena_o, O["x1"], X_CT, x_dims)
            lock_release(x_cons_o, 1)
            aie.next_bd(block[1])
        with block[3]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest=block[4], chain=block[6])
        with block[4]:
            lock_acquire(y_cons_o, 1)
            bd(arena_o, O["y0"], y_slice)
            lock_release(y_prod_o, 1)
            aie.next_bd(block[5])
        with block[5]:
            lock_acquire(y_cons_o, 1)
            bd(arena_o, O["y1"], y_slice)
            lock_release(y_prod_o, 1)
            aie.next_bd(block[4])
        with block[6]:
            aie.EndOp()


def dq_memtile(arena, tile):
    """mem tile chains + locks (x to a core row, y from 4 cores, q4_k weights)"""
    M = MT_DQ
    x_prod = aie.lock(tile, lock_id=0, init=2)
    x_cons = [aie.lock(tile, lock_id=i, init=0) for i in (1, 2)]
    y_prod = [aie.lock(tile, lock_id=5, init=2)] + [aie.lock(tile, lock_id=i, init=0) for i in (6, 7, 8)]
    y_cons = aie.lock(tile, lock_id=9, init=0)
    qw_prod = aie.lock(tile, lock_id=10, init=2)
    qw_cons = aie.lock(tile, lock_id=11, init=0)

    x_out_dims = [(m // 16, 16 * k), (2, 128), (16, k), (128, 1)]
    y_in_dims = [(m // 8, 8 * n), (8, 8), (8, n), (8, 1)]

    @aie.memtile_dma(tile)
    def body(block):
        # x in: whole 128 x 512 slab, ping / pong
        aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
        for i, buf in enumerate(("x0", "x1")):
            with block[1 + i]:
                lock_acquire(x_prod, 1)
                bd(arena, M[buf], m * k)
                lock_release(x_cons[0], 1)
                aie.next_bd(block[2 - i])
        # x out: each slab twice, cons_0 -> cons_1 -> prod. The cores read x
        # once per weight slice p (n_div_n_slice = 64 / 32 = 2), so every slab
        # goes out once per p. With k = one Q4_K block there is no second k half
        # to send (the 512-wide version had 4 sends: 2 halves x 2 p).
        with block[3]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest=block[4], chain=block[12])
        steps = [(0, 1), (1, None)]
        for b, buf in enumerate(("x0", "x1")):
            for s, (acq, rel) in enumerate(steps):
                idx = 4 + 2 * b + s
                with block[idx]:
                    lock_acquire(x_cons[acq], 1)
                    bd(arena, M[buf], m * k, x_out_dims)
                    lock_release(x_prod if rel is None else x_cons[rel], 1)
                    aie.next_bd(block[idx + 1] if idx < 7 else block[4])
        # y in from the 4 cores of a row (S2MM 1..4), each a quarter of the slab
        chain_heads = [12, 15, 18, 21]
        for q in range(4):
            head = chain_heads[q]
            with block[head]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 1 + q, dest=block[head + 1],
                              chain=block[head + 3] if q < 3 else block[24])
            acq = y_prod[q]
            rel = y_prod[q + 1] if q < 3 else y_cons
            for i, buf in enumerate(("y0", "y1")):
                with block[head + 1 + i]:
                    lock_acquire(acq, 1)
                    bd(arena, M[buf] + 2 * q * m * n, m * n, y_in_dims)
                    lock_release(rel, 1)
                    aie.next_bd(block[head + 2 - i])
        # y out to the shim
        with block[24]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest=block[25], chain=block[27])
        for i, buf in enumerate(("y0", "y1")):
            with block[25 + i]:
                lock_acquire(y_cons, 1)
                bd(arena, M[buf], 4 * m * n, [(m, n), (4, m * n), (n, 1)])
                lock_release(y_prod[0], 1)
                aie.next_bd(block[26 - i])
        # q4_k weights in / out
        with block[27]:
            aie.dma_start(aie.DMAChannelDir.S2MM, 5, dest=block[28], chain=block[30])
        for i, buf in enumerate(("qw0", "qw1")):
            with block[28 + i]:
                lock_acquire(qw_prod, 1)
                bd(arena, M[buf], QW_MEM_BYTES // 2)
                lock_release(qw_cons, 1)
                aie.next_bd(block[29 - i])
        with block[30]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 2, dest=block[31], chain=block[33])
        for i, buf in enumerate(("qw0", "qw1")):
            with block[31 + i]:
                lock_acquire(qw_cons, 1)
                bd(arena, M[buf], QW_MEM_BYTES // 2)
                lock_release(qw_prod, 1)
                aie.next_bd(block[32 - i])
        with block[33]:
            aie.EndOp()
