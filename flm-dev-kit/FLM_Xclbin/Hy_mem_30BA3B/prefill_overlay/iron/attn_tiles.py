# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Prefill attention (Hy_mem_30BA3B/attention_prefill) DMA chains and locks, on arenas.

Same chains, locks and dimensions as the original attn_tile.py; every buffer is
now (arena, offset) from layout.py. Only built into the "attn" variant.
"""
import aie.dialects.aie as aie

from utils import lock_acquire, lock_release
from layout import CT_AT, MT_AT_EVEN, MT_AT_ODD, elems

d = 128           # head dim
lq = 16
lk = 16
lk_mt = 128
lq_mt = 64
lq_ct = 32


def bd(arena, off_bytes, length, dims=None, bd_id=None):
    if dims is None:
        aie.dma_bd(arena, offset=elems(off_bytes), len=length, bd_id=bd_id)
    else:
        aie.dma_bd(arena, offset=elems(off_bytes), len=length, dimensions=dims, bd_id=bd_id)


class BdIds:
    """Fixed BD ids for attention, away from the ones aiecc gives dequant_mm
    (which counts up from 0 / 24). Then most BDs of both ops can stay loaded
    and a switch rewrites only the ids both ops use (host: overlay_host.hpp).
    A mem tile's even channels use BDs 0..23, its odd channels 24..47."""

    def __init__(self, even_first, odd_first):
        self.next = {0: even_first, 1: odd_first}

    def take(self, channel):
        i = self.next[channel % 2]
        self.next[channel % 2] += 1
        return i


CT_BD_FIRST = 8                           # dequant_mm's core BDs are 0..5
MT_EVEN_BD_FIRST = dict(even=14, odd=36)  # attn_memtile_even uses 10 + 12
MT_ODD_BD_FIRST = dict(even=16, odd=40)   # attn_memtile_odd uses 8 + 8


def attn_core(arena, tile):
    q_prod = aie.lock(tile, lock_id=0, init=1)
    q_cons = aie.lock(tile, lock_id=1, init=0)
    in_prod = aie.lock(tile, lock_id=2, init=2)
    in_cons = aie.lock(tile, lock_id=3, init=0)
    o_prod = aie.lock(tile, lock_id=4, init=2)
    o_cons = aie.lock(tile, lock_id=5, init=0)
    A = CT_AT
    ids = iter(range(CT_BD_FIRST, CT_BD_FIRST + 5))

    @aie.mem(tile)
    def body(block):
        aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
        with block[1]:
            lock_acquire(q_prod, 1)
            bd(arena, A["q"], lq_ct * d, bd_id=next(ids))
            lock_release(q_cons, 1)
            aie.next_bd(block[1])
        with block[2]:
            aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest=block[3], chain=block[5])
        for i, buf in enumerate(("in0", "in1")):
            with block[3 + i]:
                lock_acquire(in_prod, 1)
                bd(arena, A[buf], lk * d, bd_id=next(ids))
                lock_release(in_cons, 1)
                aie.next_bd(block[4 - i])
        with block[5]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest=block[6], chain=block[8])
        for i, buf in enumerate(("o0", "o1")):
            with block[6 + i]:
                lock_acquire(o_cons, 1)
                bd(arena, A[buf], 64, bd_id=next(ids))
                lock_release(o_prod, 1)
                aie.next_bd(block[7 - i])
        with block[8]:
            aie.EndOp()


def _o_in_chains(block, first, arena, M, o_prod, o_cons, end_chain, ids):
    """o from 4 cores (S2MM 2..5) into the two o buffers, then o out on MM2S 5"""
    o_dims = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)]
    heads = [first, first + 3, first + 6, first + 9]
    for q in range(4):
        head = heads[q]
        with block[head]:
            aie.dma_start(aie.DMAChannelDir.S2MM, 2 + q, dest=block[head + 1], chain=block[head + 3])
        acq = o_prod[q]
        rel = o_prod[q + 1] if q < 3 else o_cons
        for i, buf in enumerate(("o0", "o1")):
            with block[head + 1 + i]:
                lock_acquire(acq, 1)
                bd(arena, M[buf] + 2 * q * lq * d, lq * d, o_dims, bd_id=ids.take(2 + q))
                lock_release(rel, 1)
                aie.next_bd(block[head + 2 - i])
    out = first + 12
    with block[out]:
        aie.dma_start(aie.DMAChannelDir.MM2S, 5, dest=block[out + 1], chain=block[end_chain])
    for i, buf in enumerate(("o0", "o1")):
        with block[out + 1 + i]:
            lock_acquire(o_cons, 1)
            bd(arena, M[buf], 4 * lq * d, bd_id=ids.take(5))
            lock_release(o_prod[0], 1)
            aie.next_bd(block[out + 2 - i])


def attn_memtile_even(arena, tile):
    """q in (S2MM 0, 1) -> 4 core slices (MM2S 0..3); o back from 4 cores"""
    M = MT_AT_EVEN
    q_prod_0 = aie.lock(tile, lock_id=0, init=2)
    q_cons_0 = aie.lock(tile, lock_id=1, init=0)
    q_cons_1 = aie.lock(tile, lock_id=3, init=0)
    q_prod_1 = aie.lock(tile, lock_id=4, init=2)
    q_cons_2 = aie.lock(tile, lock_id=5, init=0)
    q_cons_3 = aie.lock(tile, lock_id=6, init=0)
    o_prod = [aie.lock(tile, lock_id=7, init=2)] + [aie.lock(tile, lock_id=i, init=0) for i in (8, 9, 10)]
    o_cons = aie.lock(tile, lock_id=11, init=0)
    q_dims = [(lq_ct // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)]
    ids = BdIds(MT_EVEN_BD_FIRST["even"], MT_EVEN_BD_FIRST["odd"])

    @aie.memtile_dma(tile)
    def body(block):
        # q in: first half (S2MM 0) and second half (S2MM 1) of each buffer
        aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
        for i, buf in enumerate(("q0", "q1")):
            with block[1 + i]:
                lock_acquire(q_prod_0, 1)
                bd(arena, M[buf], lq_mt * d, bd_id=ids.take(0))
                lock_release(q_cons_0, 1)
                aie.next_bd(block[2 - i])
        with block[3]:
            aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest=block[4], chain=block[6])
        for i, buf in enumerate(("q0", "q1")):
            with block[4 + i]:
                lock_acquire(q_prod_1, 1)
                bd(arena, M[buf] + 2 * lq_mt * d, lq_mt * d, bd_id=ids.take(1))
                lock_release(q_cons_2, 1)
                aie.next_bd(block[5 - i])
        # q out: quarter s of each buffer on MM2S s, chained cons_0 -> cons_1 -> prod_0, cons_2 -> cons_3 -> prod_1
        steps = [(q_cons_0, q_cons_1), (q_cons_1, q_prod_0), (q_cons_2, q_cons_3), (q_cons_3, q_prod_1)]
        for s in range(4):
            head = 6 + 3 * s
            with block[head]:
                aie.dma_start(aie.DMAChannelDir.MM2S, s, dest=block[head + 1], chain=block[head + 3])
            acq, rel = steps[s]
            for i, buf in enumerate(("q0", "q1")):
                with block[head + 1 + i]:
                    lock_acquire(acq, 1)
                    bd(arena, M[buf] + 2 * s * lq_ct * d, lq_ct * d, q_dims, bd_id=ids.take(s))
                    lock_release(rel, 1)
                    aie.next_bd(block[head + 2 - i])
        _o_in_chains(block, 18, arena, M, o_prod, o_cons, 33, ids)
        with block[33]:
            aie.EndOp()


def attn_memtile_odd(arena, tile):
    """k / v in (S2MM 0, 1) -> broadcast to the cores (MM2S 0); o back from 4 cores"""
    M = MT_AT_ODD
    in_prod = aie.lock(tile, lock_id=0, init=4)
    in_cons = aie.lock(tile, lock_id=1, init=0)
    o_prod = [aie.lock(tile, lock_id=2, init=2)] + [aie.lock(tile, lock_id=i, init=0) for i in (3, 4, 5)]
    o_cons = aie.lock(tile, lock_id=6, init=0)
    in_dims = [(lk_mt // lk, lk * d), (lk, 8), (d // 8, lk * 8), (8, 1)]
    ids = BdIds(MT_ODD_BD_FIRST["even"], MT_ODD_BD_FIRST["odd"])

    @aie.memtile_dma(tile)
    def body(block):
        aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
        for i, buf in enumerate(("in0", "in1")):
            with block[1 + i]:
                lock_acquire(in_prod, 1)
                bd(arena, M[buf], lk_mt * d, in_dims, bd_id=ids.take(0))
                lock_release(in_cons, 1)
                aie.next_bd(block[2 - i])
        with block[3]:
            aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest=block[4], chain=block[6])
        for i, buf in enumerate(("in0", "in1")):
            with block[4 + i]:
                lock_acquire(in_prod, 1)
                bd(arena, M[buf] + 2 * lk_mt * d, lk_mt * d, in_dims, bd_id=ids.take(1))
                lock_release(in_cons, 1)
                aie.next_bd(block[5 - i])
        with block[6]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest=block[7], chain=block[9])
        for i, buf in enumerate(("in0", "in1")):
            with block[7 + i]:
                lock_acquire(in_cons, 2)
                bd(arena, M[buf], 2 * lk_mt * d, [(2 * lk_mt // lk, lk * d), (lk, d), (d, 1)], bd_id=ids.take(0))
                lock_release(in_prod, 2)
                aie.next_bd(block[8 - i])
        _o_in_chains(block, 9, arena, M, o_prod, o_cons, 24, ids)
        with block[24]:
            aie.EndOp()
