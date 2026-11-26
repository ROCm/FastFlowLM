# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import numpy as np
import sys

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from typing import List
from utils import mlir_context, tile_with_location, connect_dma, connect_port, lock_acquire, lock_release

# The K / V tile buffers of a core, made for every core first: the rows of a pair read each other's
# (the lower row receives K, the upper row V).
def attn_ctbuf(tile: tile_with_location, mctx: mlir_context):
    return [aie.buffer(tile = tile.tile, datatype = mctx.in_ty, name = f"in_{k}_{tile.row}_{tile.col}") for k in range(2)]

# Core tile: S2MM 0 takes, per round, the query pack (q tiles, rel features, sinks) and then the
# NWIN relative-bias windows; S2MM 1 the K tiles (lower row of a pair) or the V tiles (upper row);
# MM2S 0 the output.
def attn_cttile(tile: tile_with_location, mctx: mlir_context, own_in, pair_in):

    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    attn = mctx.attn
    nwin = mctx.nwin
    L_begin_buffer = mctx.buffers[f"L_begin_attn_core_{ct_row}_{ct_col}"]
    L_end_buffer = mctx.buffers[f"L_end_attn_core_{ct_row}_{ct_col}"]
    window_size_buffer = mctx.buffers[f"window_size_attn_core_{ct_row}_{ct_col}"]

    q_buffer = aie.buffer(tile = ct, datatype = mctx.qp_ty, name = f"q_{ct_row}_{ct_col}")
    in_buffer_0, in_buffer_1 = own_in
    k_in, v_in = (own_in, pair_in) if (ct_row - 2) % 2 == 0 else (pair_in, own_in)
    o_buffer_0 = aie.buffer(tile = ct, datatype = mctx.o_ty, name = f"o_0_{ct_row}_{ct_col}")
    o_buffer_1 = aie.buffer(tile = ct, datatype = mctx.o_ty, name = f"o_1_{ct_row}_{ct_col}")
    s_buffer = aie.buffer(tile = ct, datatype = mctx.s_ty, name = f"s_{ct_row}_{ct_col}")
    m_buffer = aie.buffer(tile = ct, datatype = mctx.m_ty, name = f"m_{ct_row}_{ct_col}")
    y_buffer = aie.buffer(tile = ct, datatype = mctx.y_ty, name = f"y_{ct_row}_{ct_col}")
    w_buffer_0 = aie.buffer(tile = ct, datatype = mctx.w_ty, name = f"w_0_{ct_row}_{ct_col}")
    w_buffer_1 = aie.buffer(tile = ct, datatype = mctx.w_ty, name = f"w_1_{ct_row}_{ct_col}")
    p_buffer = aie.buffer(tile = ct, datatype = mctx.p_ty, name = f"p_{ct_row}_{ct_col}")

    q_prod_lock = aie.lock(ct, lock_id = 0, init = 1)
    q_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    in_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    in_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    o_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    o_cons_lock = aie.lock(ct, lock_id = 5, init = 0)
    w_prod_lock = aie.lock(ct, lock_id = 6, init = 2)
    w_cons_lock = aie.lock(ct, lock_id = 7, init = 0)
    in_fwd_lock = aie.lock(ct, lock_id = 8, init = 0)
    in_done_lock = aie.lock(ct, lock_id = 9, init = 0)

    @aie.core(ct, link_with = "attn.o", stack_size = 1024*4)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            attn(o_buffer_0, o_buffer_1, q_buffer, k_in[0], k_in[1], v_in[0], v_in[1], s_buffer, m_buffer, y_buffer,
                 w_buffer_0, w_buffer_1, p_buffer, L_begin_buffer, L_end_buffer, window_size_buffer, ct_row - 2, ct_col & 0x1)

    # block 1: q pack, blocks 2 .. nwin + 1: the windows, alternating w0 / w1
    @aie.mem(ct)
    def mem_body(block):
        nq = nwin + 1
        aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[nq + 1])

        with block[1]:
            lock_acquire(q_prod_lock, 1)
            aie.dma_bd(q_buffer)
            lock_release(q_cons_lock, 1)
            aie.next_bd(block[2])

        for k in range(nwin):
            with block[2 + k]:
                lock_acquire(w_prod_lock, 1)
                aie.dma_bd(w_buffer_0 if k % 2 == 0 else w_buffer_1)
                lock_release(w_cons_lock, 1)
                aie.next_bd(block[3 + k] if k + 1 < nwin else block[1])

        b = nq + 1
        with block[b]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[b + 1], chain=block[b + 3])

        with block[b + 1]:
            lock_acquire(in_prod_lock, 1)
            aie.dma_bd(in_buffer_0)
            lock_release(in_cons_lock, 1)
            aie.next_bd(block[b + 2])

        with block[b + 2]:
            lock_acquire(in_prod_lock, 1)
            aie.dma_bd(in_buffer_1)
            lock_release(in_cons_lock, 1)
            aie.next_bd(block[b + 1])

        with block[b + 3]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[b + 4], chain=block[b + 6])

        with block[b + 4]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_0)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[b + 5])

        with block[b + 5]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_1)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[b + 4])

        with block[b + 6]:
            aie.EndOp()

# Even memtile: S2MM 0 takes the query packs of a round (4 core rows), S2MM 1 the relative-bias table
# (broadcast to every even memtile); MM2S r sends core row r its pack and then its NWIN windows.
# S2MM 2-5 / MM2S 5 gather the output of core rows 0-1 as before.
def attn_memtile_even(tile: tile_with_location, mctx: mlir_context):
    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    qp = mctx.qp
    nwin = mctx.nwin
    win = mctx.win
    tab_rows = mctx.tab_rows
    nr = mctx.nr
    lq = mctx.lq
    d = mctx.d

    q_buffer = [aie.buffer(tile = mt, datatype = mctx.q_mem_ty, name = f"q_{k}_{mt_row}_{mt_col}") for k in range(2)]
    t_buffer = [aie.buffer(tile = mt, datatype = mctx.t_mem_ty, name = f"t_{k}_{mt_row}_{mt_col}") for k in range(2)]
    o_buffer_0 = aie.buffer(tile = mt, datatype = mctx.o_mem_ty, name = f"o_0_{mt_row}_{mt_col}")
    o_buffer_1 = aie.buffer(tile = mt, datatype = mctx.o_mem_ty, name = f"o_1_{mt_row}_{mt_col}")

    q_free = [aie.lock(mt, lock_id = 0 + k, init = 4) for k in range(2)]
    q_full = [aie.lock(mt, lock_id = 2 + k, init = 0) for k in range(2)]
    t_free = [aie.lock(mt, lock_id = 4 + k, init = 4) for k in range(2)]
    t_full = [aie.lock(mt, lock_id = 6 + k, init = 0) for k in range(2)]
    o_prod_lock_0 = aie.lock(mt, lock_id = 8, init = 2)
    o_prod_lock_1 = aie.lock(mt, lock_id = 9, init = 0)
    o_prod_lock_2 = aie.lock(mt, lock_id = 10, init = 0)
    o_prod_lock_3 = aie.lock(mt, lock_id = 11, init = 0)
    o_cons_lock = aie.lock(mt, lock_id = 12, init = 0)

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        # S2MM 0: query packs
        aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])
        for k in range(2):
            with block[1 + k]:
                lock_acquire(q_free[k], 4)
                aie.dma_bd(q_buffer[k], offset = 0, len = 4 * qp)
                lock_release(q_full[k], 4)
                aie.next_bd(block[2 - k])

        # S2MM 1: the bias table
        with block[3]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])
        for k in range(2):
            with block[4 + k]:
                lock_acquire(t_free[k], 4)
                aie.dma_bd(t_buffer[k], offset = 0, len = 2 * tab_rows * 8)
                lock_release(t_full[k], 4)
                aie.next_bd(block[5 - k])

        # MM2S r: pack r, windows of row r (window n = NWIN-1 .. 0 starts at table row 96 - 32 r + 128 (NWIN-1-n))
        b = 6
        for r in range(4):
            with block[b]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = r, dest=block[b + 1], chain=block[b + 5])
            for k in range(2):
                with block[b + 1 + 2 * k]:
                    lock_acquire(q_full[k], 1)
                    aie.dma_bd(q_buffer[k], offset = r * qp, len = qp)
                    lock_release(q_free[k], 1)
                    aie.next_bd(block[b + 2 + 2 * k])
                with block[b + 2 + 2 * k]:
                    lock_acquire(t_full[k], 1)
                    aie.dma_bd(t_buffer[k], offset = (96 - 32 * r) * 8, len = nwin * 2 * win * 8,
                               dimensions = [(nwin, 128 * 8), (2, tab_rows * 8), (win, 8), (8, 1)])
                    lock_release(t_free[k], 1)
                    aie.next_bd(block[b + 3] if k == 0 else block[b + 1])
            b += 5

        # output of core rows 0-1
        o_prod = [o_prod_lock_0, o_prod_lock_1, o_prod_lock_2, o_prod_lock_3, o_cons_lock]
        for ch in range(4):
            with block[b]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2 + ch, dest=block[b + 1], chain=block[b + 3])
            with block[b + 1]:
                lock_acquire(o_prod[ch], 1)
                aie.dma_bd(o_buffer_0, offset = ch * lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
                lock_release(o_prod[ch + 1], 1)
                aie.next_bd(block[b + 2])
            with block[b + 2]:
                lock_acquire(o_prod[ch], 1)
                aie.dma_bd(o_buffer_1, offset = ch * lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
                lock_release(o_prod[ch + 1], 1)
                aie.next_bd(block[b + 1])
            b += 3

        with block[b]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 5, dest=block[b + 1], chain=block[b + 3])
        with block[b + 1]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_0)
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[b + 2])
        with block[b + 2]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_1)
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[b + 1])
        with block[b + 3]:
            aie.EndOp()

# Odd memtile: S2MM 0 / 1 take the K / V blocks (KV_BUFS 128-row buffers each, already tiled bfp16 from the
# host, so both ways are linear); MM2S 0 sends the K tiles to the lower core row of each pair, MM2S 1 the V
# tiles to the upper row.
KV_BUFS = 2

def attn_memtile_odd(tile: tile_with_location, mctx: mlir_context):
    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    o_mem_ty = mctx.o_mem_ty
    lk_mt = mctx.lk_mt
    lk = mctx.lk
    lq = mctx.lq
    d = mctx.d
    kv_len = lk_mt // lk * mctx.kv_tile
    nb = KV_BUFS

    kv_buffer = [[aie.buffer(tile = mt, datatype = mctx.kv_mem_ty, name = f"{'kv'[h]}_{k}_{mt_row}_{mt_col}") for k in range(nb)]
                 for h in range(2)]
    o_buffer_0 = aie.buffer(tile = mt, datatype = o_mem_ty, name = f"o_0_{mt_row}_{mt_col}")
    o_buffer_1 = aie.buffer(tile = mt, datatype = o_mem_ty, name = f"o_1_{mt_row}_{mt_col}")

    kv_free = [aie.lock(mt, lock_id = 7 + h, init = nb) for h in range(2)]
    kv_full = [aie.lock(mt, lock_id = 9 + h, init = 0) for h in range(2)]

    o_prod_lock_0 = aie.lock(mt, lock_id = 2, init = 2)
    o_prod_lock_1 = aie.lock(mt, lock_id = 3, init = 0)
    o_prod_lock_2 = aie.lock(mt, lock_id = 4, init = 0)
    o_prod_lock_3 = aie.lock(mt, lock_id = 5, init = 0)
    o_cons_lock = aie.lock(mt, lock_id = 6, init = 0)

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        b = 0
        for h in range(2):
            with block[b]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = h, dest=block[b + 1], chain=block[b + 1 + nb])
            for k in range(nb):
                with block[b + 1 + k]:
                    lock_acquire(kv_free[h], 1)
                    aie.dma_bd(kv_buffer[h][k], offset = 0, len = kv_len)
                    lock_release(kv_full[h], 1)
                    aie.next_bd(block[b + 1 + (k + 1) % nb])
            b += 1 + nb
            with block[b]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = h, dest=block[b + 1], chain=block[b + 1 + nb])
            for k in range(nb):
                with block[b + 1 + k]:
                    lock_acquire(kv_full[h], 1)
                    aie.dma_bd(kv_buffer[h][k], offset = 0, len = kv_len)
                    lock_release(kv_free[h], 1)
                    aie.next_bd(block[b + 1 + (k + 1) % nb])
            b += 1 + nb

        o_prod = [o_prod_lock_0, o_prod_lock_1, o_prod_lock_2, o_prod_lock_3, o_cons_lock]
        for ch in range(4):
            with block[b]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2 + ch, dest=block[b + 1], chain=block[b + 3])
            with block[b + 1]:
                lock_acquire(o_prod[ch], 1)
                aie.dma_bd(o_buffer_0, offset = ch * lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
                lock_release(o_prod[ch + 1], 1)
                aie.next_bd(block[b + 2])
            with block[b + 2]:
                lock_acquire(o_prod[ch], 1)
                aie.dma_bd(o_buffer_1, offset = ch * lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
                lock_release(o_prod[ch + 1], 1)
                aie.next_bd(block[b + 1])
            b += 3

        with block[b]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 5, dest=block[b + 1], chain=block[b + 3])
        with block[b + 1]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_0)
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[b + 2])
        with block[b + 2]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_1)
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[b + 1])
        with block[b + 3]:
            aie.EndOp()
