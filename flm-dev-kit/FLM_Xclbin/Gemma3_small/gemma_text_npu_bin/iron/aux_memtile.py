# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
import aie.dialects.aie as aie
import aie.dialects.aiex as aiex
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils
from aie.utils.trace import MemTilePortEvent, MemTileEvent, PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from aie.dialects import memref

from utils import mlir_context, tile_with_location, lock_acquire, lock_release

def aux_memtile(tile: tile_with_location, ctx: mlir_context):

    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    DH = ctx.parameters["DH"]
    PADDED_D = ctx.parameters["PADDED_D"]
    repeats = PADDED_D // 16 // 16
    q_ty = ctx.data_tys["q_ty"]
    out_ty = ctx.data_tys["out_ty"]

    down_ty = ctx.data_tys["down_ty"]
    down_prod_lock = aie.lock(mt, lock_id = 0, init = repeats)
    down_cons_lock = aie.lock(mt, lock_id = 1, init = 0)
    q_prod_lock = aie.lock(mt, lock_id = 2, init = 1)
    q_cons_lock = aie.lock(mt, lock_id = 3, init = 0)
    o_prod_lock = aie.lock(mt, lock_id = 4, init = repeats)
    o_cons_lock = aie.lock(mt, lock_id = 5, init = 0)

    q_buffer = aie.buffer(tile = mt, datatype = q_ty, name = f"q_buffer_{mt_row}_{mt_col}")
    o_buffer = aie.buffer(tile = mt, datatype = out_ty, name = f"o_buffer_{mt_row}_{mt_col}")

    G = ctx.parameters["kv_groups"]

    down_buffer = aie.buffer(tile = mt, datatype = down_ty, name = f"down_buffer_{mt_row}_{mt_col}")
    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        with block[0]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[1], chain = block[3])

        with block[1]:
            lock_acquire(down_prod_lock, repeats)
            aie.dma_bd(down_buffer)
            lock_release(down_cons_lock, repeats)
            aie.next_bd(block[2])

        with block[2]:
            lock_acquire(down_prod_lock, repeats)
            aie.dma_bd(down_buffer)
            lock_release(down_cons_lock, repeats)
            aie.next_bd(block[1])


        with block[3]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[4], chain = block[6])

        with block[4]:
            lock_acquire(down_cons_lock, 1)
            aie.dma_bd_packet(0, 0)
            aie.dma_bd(down_buffer)
            lock_release(down_prod_lock, 1)
            aie.next_bd(block[4])


        with block[6]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[7], chain = block[8])

        with block[7]:
            lock_acquire(q_prod_lock, 1)
            aie.dma_bd(q_buffer)
            lock_release(q_cons_lock, 1)
            aie.next_bd(block[7])

        with block[8]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest = block[9], chain = block[10])

        with block[9]:
            lock_acquire(q_cons_lock, 1)
            aie.dma_bd(q_buffer, offset = 0, len = G * DH, dimensions = [(DH // 8, 8), (G, DH), (8, 1)])
            lock_release(q_prod_lock, 1)
            aie.next_bd(block[9])

        with block[10]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest = block[11], chain = block[12])

        with block[11]:
            lock_acquire(o_prod_lock, repeats)
            aie.dma_bd(o_buffer, offset = 0, len = G * DH)
            lock_release(o_cons_lock, repeats)
            aie.next_bd(block[11])

        with block[12]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest = block[13], chain = block[14])

        with block[13]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd_packet(0, 2)
            aie.dma_bd(o_buffer)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[13])

        with block[14]:
            aie.EndOp()