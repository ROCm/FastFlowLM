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

# ---------------------------------------------------------------------------
# Phase 2 layout: all 8 columns, every core owns ONE q/k head (128 channels,
# L2-normed) plus 64 v channels, 192 in all. Per column that is 512 q/k
# channels and 256 v channels, which are not next to each other in memory, so
# they travel as two streams:
#
#   shim MM2S 0 --qk (512/row)--> mem tile S2MM 0 -> qk_in[2]  \  MM2S i: qk part (128)
#   shim MM2S 1 --v  (256/row)--> mem tile S2MM 1 -> v_in[2]   /  then v part (64) -> core i
#   core i --192/row--> mem tile S2MM 2+i: qk part -> qk_out[2], v part -> v_out[2]
#   mem tile MM2S 4 --qk_out--> shim S2MM 0,   mem tile MM2S 5 --v_out--> shim S2MM 1
#
# Each stream starts with the 4 conv-weight rows, then the 3 conv-state rows,
# then the L token rows: the core copies the first 4 rows into its weights.
# That keeps every transfer the same shape, so no channel is spent on weights.
#
# Every buffer has its own lock chain, the same pattern as before: the writer
# takes a "prod" credit and hands the row to core 0's reader, core 0's reader
# hands it to core 1's, ..., core 3's gives the credit back. q/k and v never
# wait on each other.
# ---------------------------------------------------------------------------

def conv_cu(tile: tile_with_location, mctx: mlir_context):
    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    conv = mctx.conv

    o_ty = mctx.o_ty
    x_in_ty = mctx.x_in_ty
    w_ty = mctx.w_ty
    x_ty = mctx.x_ty

    L_buffer = mctx.buffers[f"L_buffer_{ct_row}_{ct_col}"]

    x_buffer_0 = aie.buffer(tile = ct, datatype = x_in_ty, name = f"x_0_{ct_row}_{ct_col}")
    x_buffer_1 = aie.buffer(tile = ct, datatype = x_in_ty, name = f"x_1_{ct_row}_{ct_col}")
    o_buffer_0 = aie.buffer(tile = ct, datatype = o_ty, name = f"o_0_{ct_row}_{ct_col}")
    o_buffer_1 = aie.buffer(tile = ct, datatype = o_ty, name = f"o_1_{ct_row}_{ct_col}")
    w_buffer = aie.buffer(tile = ct, datatype = w_ty, name = f"w_{ct_row}_{ct_col}")    # filled by the core
    x_buffer = aie.buffer(tile = ct, datatype = x_ty, name = f"x_{ct_row}_{ct_col}")    # 4-row ring

    x_prod_lock = aie.lock(ct, lock_id = 0, init = 2)
    x_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    o_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    o_cons_lock = aie.lock(ct, lock_id = 5, init = 0)

    @aie.core(ct, link_with = "conv.o", stack_size = 1024*8)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            conv(o_buffer_0, o_buffer_1, x_buffer_0, x_buffer_1, x_buffer, w_buffer, L_buffer)

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
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

        with block[4]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_0)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[5])

        with block[5]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_1)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[4])

        with block[6]:
            aie.EndOp()


def conv_memtile(tile: tile_with_location, mctx: mlir_context):
    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    qk_mem_ty = mctx.qk_mem_ty
    v_mem_ty = mctx.v_mem_ty
    d_qk = mctx.d_qk        # q/k channels per core (one head)
    d_v = mctx.d_v          # v channels per core
    n_core = mctx.n_core    # cores per column

    def ping_pong(name):
        return [aie.buffer(tile = mt, datatype = qk_mem_ty if name.startswith("qk") else v_mem_ty,
                           name = f"{name}_{i}_{mt_row}_{mt_col}") for i in range(2)]

    qk_in = ping_pong("qk_in")
    v_in = ping_pong("v_in")
    qk_out = ping_pong("qk_out")
    v_out = ping_pong("v_out")

    # input chains: prod (writer credits, 2 = ping + pong), then one hand-off lock per core
    lock_id = [0]
    def new_lock(init):
        l = aie.lock(mt, lock_id = lock_id[0], init = init)
        lock_id[0] += 1
        return l

    qk_in_prod = new_lock(2)
    qk_in_cons = [new_lock(0) for _ in range(n_core)]
    v_in_prod = new_lock(2)
    v_in_cons = [new_lock(0) for _ in range(n_core)]
    # output chains: core 0 writes first (2 credits), hands to core 1, ..., then the reader
    qk_out_prod = [new_lock(2)] + [new_lock(0) for _ in range(n_core - 1)]
    qk_out_cons = new_lock(0)
    v_out_prod = [new_lock(2)] + [new_lock(0) for _ in range(n_core - 1)]
    v_out_cons = new_lock(0)

    # blocks are numbered as they are used. counter = the block the next
    # dma_start goes in; 0 means the entry block, as in the original design
    counter = [0]

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        # one channel: its dma_start, then its BDs in a loop
        def channel(direction, ch, bds):
            first = counter[0] + 1
            end = first + len(bds)
            if counter[0] == 0:
                aie.dma_start(channel_dir = direction, channel_index = ch, dest = block[first], chain = block[end])
            else:
                with block[counter[0]]:
                    aie.dma_start(channel_dir = direction, channel_index = ch, dest = block[first], chain = block[end])
            for k, (acq, buf, off, ln, rel) in enumerate(bds):
                with block[first + k]:
                    lock_acquire(acq[0], acq[1])
                    aie.dma_bd(buf, offset = off, len = ln)
                    lock_release(rel[0], rel[1])
                    aie.next_bd(block[first + (k + 1) % len(bds)])
            counter[0] = end

        S2MM = aie.DMAChannelDir.S2MM
        MM2S = aie.DMAChannelDir.MM2S
        d_qk_col = d_qk * n_core
        d_v_col = d_v * n_core

        # rows in from the shim
        channel(S2MM, 0, [((qk_in_prod, 1), qk_in[p], 0, d_qk_col, (qk_in_cons[0], 1)) for p in range(2)])
        channel(S2MM, 1, [((v_in_prod, 1), v_in[p], 0, d_v_col, (v_in_cons[0], 1)) for p in range(2)])

        # rows out to core i: its q/k head, then its v channels (one 192-value message)
        for i in range(n_core):
            qk_next = qk_in_cons[i + 1] if i + 1 < n_core else qk_in_prod
            v_next = v_in_cons[i + 1] if i + 1 < n_core else v_in_prod
            bds = []
            for p in range(2):
                bds.append(((qk_in_cons[i], 1), qk_in[p], i * d_qk, d_qk, (qk_next, 1)))
                bds.append(((v_in_cons[i], 1), v_in[p], i * d_v, d_v, (v_next, 1)))
            channel(MM2S, i, bds)

        # results back from core i: the q/k part into qk_out, the v part into v_out
        for i in range(n_core):
            qk_next = qk_out_prod[i + 1] if i + 1 < n_core else qk_out_cons
            v_next = v_out_prod[i + 1] if i + 1 < n_core else v_out_cons
            bds = []
            for p in range(2):
                bds.append(((qk_out_prod[i], 1), qk_out[p], i * d_qk, d_qk, (qk_next, 1)))
                bds.append(((v_out_prod[i], 1), v_out[p], i * d_v, d_v, (v_next, 1)))
            channel(S2MM, 2 + i, bds)

        # whole rows out to the shim
        channel(MM2S, 4, [((qk_out_cons, 1), qk_out[p], 0, d_qk_col, (qk_out_prod[0], 1)) for p in range(2)])
        channel(MM2S, 5, [((v_out_cons, 1), v_out[p], 0, d_v_col, (v_out_prod[0], 1)) for p in range(2)])

        with block[counter[0]]:
            aie.EndOp()
