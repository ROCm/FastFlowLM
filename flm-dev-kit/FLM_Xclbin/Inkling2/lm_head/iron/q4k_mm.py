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

# IT0's MM2S 0 carries two things, so both of its flows have to be packet-switched:
# the x broadcast that every core needs, and column 0's second weight stream. Every
# other column gets a plain circuit flow on its own shim tile's MM2S 0. See
# lm_head_column for why column 0 is the odd one out.
X_PKT_ID = 0
W_PKT_ID = 1

# Ping-pong depth of each weight group in the memtile. Two would be enough to keep
# both of a group's output channels busy; four just buys slack against jitter and
# costs 2 * 4 * W_BLOCK_SIZE * 2 B = 70 KB of the memtile's 512 KB.
W_SLOTS = 4

def lm_head_tile(tile: tile_with_location, mctx: mlir_context):

    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    qmm = mctx.qmm
    x_ty = mctx.x_ty
    w_ty = mctx.w_ty
    y_ty = mctx.y_ty

    x_buffer   = aie.buffer(tile = ct, datatype = x_ty, name = f"x_{ct_row}_{ct_col}")
    w_buffer_0 = aie.buffer(tile = ct, datatype = w_ty, name = f"w_0_{ct_row}_{ct_col}")
    w_buffer_1 = aie.buffer(tile = ct, datatype = w_ty, name = f"w_1_{ct_row}_{ct_col}")
    y_buffer_0 = aie.buffer(tile = ct, datatype = y_ty, name = f"y_0_{ct_row}_{ct_col}")
    y_buffer_1 = aie.buffer(tile = ct, datatype = y_ty, name = f"y_1_{ct_row}_{ct_col}")
    
    x_prod_lock = aie.lock(ct, lock_id = 0, init = 1)
    x_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    w_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    w_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    y_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    y_cons_lock = aie.lock(ct, lock_id = 5, init = 0)
    
    @aie.core(ct, link_with = "qmm.o", stack_size = 1024 * 8)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            qmm(y_buffer_0, y_buffer_1, w_buffer_0, w_buffer_1, x_buffer)
    
    @aie.mem(ct)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

        with block[1]:
            lock_acquire(x_prod_lock, 1)
            aie.dma_bd(x_buffer)
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

    ``chains`` is a list of ``(direction, channel, bds)`` where each bd is
    ``(buffer, offset, length, acquire_lock, release_lock)`` and ``length = None``
    means the whole buffer. The bds of a channel are wired into a ring, and the
    channels are chained together the way aie.dma_start's ``chain`` argument wants.
    Writing this out by hand is 60-odd near-identical blocks; the two weight groups
    below are structurally identical, so it is worth generating.
    """
    starts = [0]
    bd_ids = []
    next_id = 1
    for _, _, bds in chains:
        bd_ids.append(list(range(next_id, next_id + len(bds))))
        next_id += len(bds)
        starts.append(next_id)
        next_id += 1
    end_id = starts.pop()

    for ci, (direction, channel, bds) in enumerate(chains):
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
                lock_acquire(acq, 1)
                if length is None:
                    aie.dma_bd(buf)
                else:
                    aie.dma_bd(buf, offset = offset, len = length)
                lock_release(rel, 1)
                aie.next_bd(block[bd_ids[ci][(bi + 1) % len(bds)]])

    with block[end_id]:
        aie.EndOp()


def lm_head_memtile(tile: tile_with_location, mctx: mlir_context):

    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    w_mem_ty = mctx.w_mem_ty
    y_mem_ty = mctx.y_mem_ty

    W_BLOCK_SIZE        = mctx.W_BLOCK_SIZE
    Q4NX_ROW_BLOCK_SIZE = mctx.Q4NX_ROW_BLOCK_SIZE

    # The four cores of a column are fed by two independent weight streams rather
    # than one: a single shim->memtile channel saturates at roughly a quarter of what
    # the four cores can consume, which is what made this kernel purely data-movement
    # bound. Group a serves cores 0/1, group b serves cores 2/3, and each group has
    # its own shim MM2S channel behind it. A slot is therefore two k-blocks wide, not
    # four, and the host lays the weights out in DDR as two contiguous halves per
    # column-round to match (see reorder_cpy in host/host.cpp).
    w_a = [aie.buffer(tile = mt, datatype = w_mem_ty, name = f"wa_{i}_{mt_row}_{mt_col}") for i in range(W_SLOTS)]
    w_b = [aie.buffer(tile = mt, datatype = w_mem_ty, name = f"wb_{i}_{mt_row}_{mt_col}") for i in range(W_SLOTS)]
    y_buffers = [
        aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_0_{mt_row}_{mt_col}"),
        aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_1_{mt_row}_{mt_col}"),
    ]

    # Each group hands a slot down its two output channels in turn and only then
    # returns it to the producer, so a slot is never refilled while either core is
    # still reading it.
    wa_prod_lock   = aie.lock(mt, lock_id = 0, init = W_SLOTS)
    wa_cons_lock_0 = aie.lock(mt, lock_id = 1, init = 0)
    wa_cons_lock_1 = aie.lock(mt, lock_id = 2, init = 0)
    wb_prod_lock   = aie.lock(mt, lock_id = 3, init = W_SLOTS)
    wb_cons_lock_0 = aie.lock(mt, lock_id = 4, init = 0)
    wb_cons_lock_1 = aie.lock(mt, lock_id = 5, init = 0)
    y_prod_lock_0 = aie.lock(mt, lock_id = 6, init = 2)
    y_prod_lock_1 = aie.lock(mt, lock_id = 7, init = 0)
    y_prod_lock_2 = aie.lock(mt, lock_id = 8, init = 0)
    y_prod_lock_3 = aie.lock(mt, lock_id = 9, init = 0)
    y_cons_lock   = aie.lock(mt, lock_id = 10, init = 0)

    Y = Q4NX_ROW_BLOCK_SIZE

    chains = [
        # Weight group a: in from the shim, out to cores 0 and 1.
        (aie.DMAChannelDir.S2MM, 1, [(b, 0, None, wa_prod_lock, wa_cons_lock_0) for b in w_a]),
        (aie.DMAChannelDir.MM2S, 1, [(b, 0, W_BLOCK_SIZE, wa_cons_lock_0, wa_cons_lock_1) for b in w_a]),
        (aie.DMAChannelDir.MM2S, 2, [(b, W_BLOCK_SIZE, W_BLOCK_SIZE, wa_cons_lock_1, wa_prod_lock) for b in w_a]),
        # Weight group b: in from the shim, out to cores 2 and 3.
        (aie.DMAChannelDir.S2MM, 0, [(b, 0, None, wb_prod_lock, wb_cons_lock_0) for b in w_b]),
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


def lm_head_column(CTs: List[tile_with_location], MT: tile_with_location, IT: tile_with_location, XIT: tile_with_location, mctx: mlir_context, w_pkt_id: int = -1):

    lm_head_memtile(MT, mctx)

    # Both of this column's shim MM2S channels carry weights. Column 0 is the
    # exception: its MM2S 0 also has to carry the x broadcast, because x has to come
    # from somewhere and every memtile S2MM channel is now spoken for (two weights,
    # four y). Sharing it costs column 0 an extra 7680 B per token against 67 MB of
    # weights, and requires the two flows off that port to be packet-switched.
    connect_dma(IT, MT, 1, 1)
    connect_dma(IT, MT, 0, 0, pkt_id = w_pkt_id)

    for i, proj_core in enumerate(CTs):
        lm_head_tile(proj_core, mctx)
        # x goes shim -> core directly now, not through the memtile.
        connect_dma(XIT, proj_core, 0, 0, pkt_id = X_PKT_ID)
        connect_dma(MT, proj_core, i + 1, 1)

        connect_dma(proj_core, MT, 0, i + 2)

    connect_dma(MT, IT, 5, 0)
