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

def mm_cttile(tile: tile_with_location, mctx: mlir_context):

    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    mm = mctx.mm
    x_ty = mctx.x_ty
    w_ty = mctx.w_ty
    y_ty = mctx.y_ty
    y_acc_ty = mctx.y_acc_ty
    b_ty = mctx.b_ty

    m_slice = mctx.m_slice
    k_slice = mctx.k_slice
    n_slice = mctx.n_slice

    x_buffer_0 = aie.buffer(tile = ct, datatype = x_ty, name = f"x_0_{ct_row}_{ct_col}")
    x_buffer_1 = aie.buffer(tile = ct, datatype = x_ty, name = f"x_1_{ct_row}_{ct_col}")
    w_buffer_0 = aie.buffer(tile = ct, datatype = w_ty, name = f"w_0_{ct_row}_{ct_col}")
    w_buffer_1 = aie.buffer(tile = ct, datatype = w_ty, name = f"w_1_{ct_row}_{ct_col}")
    y_buffer_0 = aie.buffer(tile = ct, datatype = y_ty, name = f"y_0_{ct_row}_{ct_col}")
    y_buffer_1 = aie.buffer(tile = ct, datatype = y_ty, name = f"y_1_{ct_row}_{ct_col}")
    y_acc_buffer = aie.buffer(tile = ct, datatype = y_acc_ty, name = f"y_acc_{ct_row}_{ct_col}")
    b_buffer = aie.buffer(tile = ct, datatype = b_ty, name = f"b_{ct_row}_{ct_col}")
    rtp_buffer = mctx.buffers[f"rtp_{ct_row}_{ct_col}"]

    x_prod_lock = aie.lock(ct, lock_id = 0, init = 2)
    x_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    w_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    w_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    y_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    y_cons_lock = aie.lock(ct, lock_id = 5, init = 0)
    l_prod_lock = aie.lock(ct, lock_id = 6, init = 0)

    @aie.core(ct, link_with = "mm.o", stack_size = 1024 * 3)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            mm(x_buffer_0, x_buffer_1, w_buffer_0, w_buffer_1, y_buffer_0, y_buffer_1, y_acc_buffer, b_buffer, rtp_buffer)

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

def mm_memtile(tile: tile_with_location, mctx: mlir_context):
    """Memtile at physical column j. There is one per column, and each plays two
    unrelated roles, one per operand:

      x: MT[j] holds the m rows of array row j, broadcast along that row to
         CT[j][0..total_cols-1].   (only j < total_rows carry x)
      w: MT[j] holds the n cols of array column j, broadcast down that column to
         CT[0..total_rows-1][j].

    On the way out it collects the total_rows row-partials of its own column into
    one (total_rows*m, n) chunk and streams that to the shim as contiguous rows.
    """
    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    x_mem_ty = mctx.x_mem_ty
    w_mem_ty = mctx.w_mem_ty
    y_mem_ty = mctx.y_mem_ty
    m = mctx.m
    k = mctx.k
    n = mctx.n
    m_slice = mctx.m_slice
    k_slice = mctx.k_slice
    n_slice = mctx.n_slice
    total_rows = mctx.total_rows

    # Packed operands, in int32 words. A memtile x buffer is the column's m rows x
    # one k tile, as the host laid it out: [m/16 groups][k/8][2][72 B], a group's
    # k tile (group_words) contiguous. The core takes it one k_slice chunk at a
    # time, every group's k_slice/8 tile pairs (x_chunk_words) in turn. w is the
    # same with n/16 groups.
    x_chunk_words = (k_slice // 8) * 144 // 4
    group_words = (k // 8) * 144 // 4
    x_words = (m // 16) * group_words
    w_words = (n // 16) * group_words

    x_buffer_0 = aie.buffer(tile = mt, datatype = x_mem_ty, name = f"x_0_{mt_row}_{mt_col}")
    x_buffer_1 = aie.buffer(tile = mt, datatype = x_mem_ty, name = f"x_1_{mt_row}_{mt_col}")
    w_buffer_0 = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_0_{mt_row}_{mt_col}")
    w_buffer_1 = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_1_{mt_row}_{mt_col}")
    y_buffer_0 = aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_0_{mt_row}_{mt_col}")
    y_buffer_1 = aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_1_{mt_row}_{mt_col}")

    x_prod_lock = aie.lock(mt, lock_id = 0, init = 2)
    x_cons_lock = aie.lock(mt, lock_id = 1, init = 0)
    w_prod_lock = aie.lock(mt, lock_id = 2, init = 2)
    w_cons_lock = aie.lock(mt, lock_id = 3, init = 0)
    y_prod_lock = aie.lock(mt, lock_id = 4, init = 2)
    # One chained lock per array row: row r's partial releases row r+1's slot, so
    # the outbound BD only fires once all four rows have landed. A lock count is
    # not an assignment - each row needs its own lock rather than sharing a count.
    y_cons_locks = [aie.lock(mt, lock_id = 5 + 2 * r, init = 0) for r in range(total_rows)]

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        # x in from the shim
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

        # x out to the compute tiles, one k_slice chunk of each 16-row group in turn
        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

        with block[4]:
            lock_acquire(x_cons_lock, 1)
            aie.dma_bd(x_buffer_0, offset = 0, len = x_words, dimensions = [(k // k_slice, x_chunk_words), (m // 16, group_words), (x_chunk_words, 1)])
            lock_release(x_prod_lock, 1)
            aie.next_bd(block[5])

        with block[5]:
            lock_acquire(x_cons_lock, 1)
            aie.dma_bd(x_buffer_1, offset = 0, len = x_words, dimensions = [(k // k_slice, x_chunk_words), (m // 16, group_words), (x_chunk_words, 1)])
            lock_release(x_prod_lock, 1)
            aie.next_bd(block[4])

        # w in from the shim
        with block[6]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[7], chain=block[9])

        with block[7]:
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w_buffer_0)
            lock_release(w_cons_lock, 1)
            aie.next_bd(block[8])

        with block[8]:
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w_buffer_1)
            lock_release(w_cons_lock, 1)
            aie.next_bd(block[7])

        # w out to the compute tiles
        with block[9]:
            s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[10], chain=block[12])

        with block[10]:
            lock_acquire(w_cons_lock, 1)
            aie.dma_bd(w_buffer_0, offset = 0, len = w_words, dimensions = [(k // k_slice, x_chunk_words), (n // 16, group_words), (x_chunk_words, 1)])
            lock_release(w_prod_lock, 1)
            aie.next_bd(block[11])

        with block[11]:
            lock_acquire(w_cons_lock, 1)
            aie.dma_bd(w_buffer_1, offset = 0, len = w_words, dimensions = [(k // k_slice, x_chunk_words), (n // 16, group_words), (x_chunk_words, 1)])
            lock_release(w_prod_lock, 1)
            aie.next_bd(block[10])

        # y in: one S2MM channel per array row, each landing at its own m-offset.
        # The compute tile emits y in mmul C-operand order (8x8 blocks, dim-block
        # major), so the BD un-blocks it into plain (m, n) rows on the way in.
        blk = 12
        for r in range(total_rows):
            s2mm_ch = 2 + r
            with block[blk]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = s2mm_ch,
                              dest = block[blk + 1], chain = block[blk + 3])
            # Row 0 takes a free y slot; later rows wait on the previous row's lock
            # so the four partials of one output tile stay together.
            acq = y_prod_lock if r == 0 else y_cons_locks[r - 1]
            rel = y_cons_locks[r]
            ping, pong = blk + 1, blk + 2
            for this_blk, buf, next_blk in ((ping, y_buffer_0, pong),
                                            (pong, y_buffer_1, ping)):
                with block[this_blk]:
                    lock_acquire(acq, 1)
                    aie.dma_bd(buf, offset = r * n, len = m * n,
                               dimensions = [(m // 8, 8 * total_rows * n), (n // 8, 8), (8, total_rows * n), (8, 1)])
                    lock_release(rel, 1)
                    aie.next_bd(block[next_blk])
            blk += 3

        # y out to the shim: the whole (total_rows*m, n) column chunk, contiguous.
        with block[blk]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2,
                          dest = block[blk + 1], chain = block[blk + 3])

        with block[blk + 1]:
            lock_acquire(y_cons_locks[-1], 1)
            aie.dma_bd(y_buffer_0, offset = 0, len = total_rows * m * n)
            lock_release(y_prod_lock, 1)
            aie.next_bd(block[blk + 2])

        with block[blk + 2]:
            lock_acquire(y_cons_locks[-1], 1)
            aie.dma_bd(y_buffer_1, offset = 0, len = total_rows * m * n)
            lock_release(y_prod_lock, 1)
            aie.next_bd(block[blk + 1])

        with block[blk + 3]:
            aie.EndOp()
