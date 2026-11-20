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
from typing import List
from utils import mlir_context, tile_with_location, connect_dma, lock_acquire, lock_release

# phase 1: qkv proj, D -> (D + D // 2) to RoPE

# phase 2: o_proj, D -> D to rms_residual

# phase 3: up/gate proj, D -> D * 8 to MT_GLU

# phase 4: down proj, D * 4 -> D, to rms_residual

# minimal output unit: (m, 1)

def glu_tile(tile: tile_with_location, ctx: mlir_context):

    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    glu = ctx.glu
    D = ctx.D
    m = ctx.m
    k = ctx.k
    ug_ty = ctx.ug_ty
    hid_ty = ctx.hid_ty
    # q_proj tile and mem 
    # double buffer
    x_buffer_0 = aie.buffer(tile = ct, datatype = ug_ty, name = f"x_0_{ct_row}_{ct_col}")
    x_buffer_1 = aie.buffer(tile = ct, datatype = ug_ty, name = f"x_1_{ct_row}_{ct_col}")

    hid_buffer_0 = aie.buffer(tile = ct, datatype = hid_ty, name = f"hid_0_{ct_row}_{ct_col}")
    hid_buffer_1 = aie.buffer(tile = ct, datatype = hid_ty, name = f"hid_1_{ct_row}_{ct_col}")

    x_prod_lock = aie.lock(ct, lock_id = 0, init = 2)
    x_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    hid_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    hid_cons_lock = aie.lock(ct, lock_id = 3, init = 0)

    @aie.core(ct, link_with = "glu.o")
    def core_body():
        for _ in range_(0xFFFFFFFF):
            glu(hid_buffer_0, hid_buffer_1, x_buffer_0, x_buffer_1)
    
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
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[4], chain = block[6])
            
        with block[4]: # x, to mt
            lock_acquire(hid_cons_lock, 1)
            aie.dma_bd(hid_buffer_0)
            lock_release(hid_prod_lock, 1)
            aie.next_bd(block[5])

        with block[5]: # w, to mt
            lock_acquire(hid_cons_lock, 1)
            aie.dma_bd(hid_buffer_1)
            lock_release(hid_prod_lock, 1)
            aie.next_bd(block[4])

        with block[6]:
            aie.EndOp()


# def glu_memtile(tile: tile_with_location, ctx: mlir_context):
#     mt = tile.tile
#     mt_row = tile.row
#     mt_col = tile.col

#     ug_col_ty = ctx.ug_col_ty
#     hid_col_ty = ctx.hid_col_ty
#     hid_buffer_0 = aie.buffer(mt, hid_col_ty, name = f"hid_buffer_0_{mt_row}_{mt_col}")
#     hid_buffer_1 = aie.buffer(mt, hid_col_ty, name = f"hid_buffer_1_{mt_row}_{mt_col}")
#     ug_buffer_0 = aie.buffer(mt, ug_col_ty, name = f"ug_buffer_0_{mt_row}_{mt_col}")
#     ug_buffer_1 = aie.buffer(mt, ug_col_ty, name = f"ug_buffer_1_{mt_row}_{mt_col}")

#     # double buffer
#     # split D to four D / 4
#     ug_prod_lock = aie.lock(mt, lock_id = 0, init = 2)
#     ug_cons_lock = aie.lock(mt, lock_id = 1, init = 0)
#     hid_prod_lock = aie.lock(mt, lock_id = 2, init = 2)
#     hid_cons_lock = aie.lock(mt, lock_id = 3, init = 0)

#     @aie.memtile_dma(mt)
#     def memtile_dma_body(block):
#         s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 4, dest=block[1], chain=block[3])
#         with block[1]:
#             aie.use_lock(ug_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
#             aie.dma_bd(ug_buffer_0)
#             aie.use_lock(ug_cons_lock, aie.LockAction.Release, value = 1)
#             aie.next_bd(block[2])

#         with block[2]:
#             aie.use_lock(ug_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
#             aie.dma_bd(ug_buffer_1)
#             aie.use_lock(ug_cons_lock, aie.LockAction.Release, value = 1)
#             aie.next_bd(block[1])
            
#         with block[3]:
#             s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

#         with block[4]:
#             aie.use_lock(ug_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
#             aie.dma_bd(ug_buffer_0)
#             aie.use_lock(ug_prod_lock, aie.LockAction.Release, value = 1)
#             aie.next_bd(block[5])
        
#         with block[5]:
#             aie.use_lock(ug_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
#             aie.dma_bd(ug_buffer_1)
#             aie.use_lock(ug_prod_lock, aie.LockAction.Release, value = 1)
#             aie.next_bd(block[4])
     

#         with block[6]:
#             aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[7], chain = block[9])
            
#         with block[7]:
#             aie.use_lock(hid_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
#             aie.dma_bd(hid_buffer_0)
#             aie.use_lock(hid_cons_lock, aie.LockAction.Release, value = 1)
#             aie.next_bd(block[8])

#         with block[8]:
#             aie.use_lock(hid_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
#             aie.dma_bd(hid_buffer_1)
#             aie.use_lock(hid_cons_lock, aie.LockAction.Release, value = 1)
#             aie.next_bd(block[7])

#         with block[9]:
#             aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 4, dest = block[10], chain = block[12])
            
#         with block[10]:
#             aie.use_lock(hid_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
#             aie.dma_bd_packet(0, 1)
#             aie.dma_bd(hid_buffer_0)
#             aie.use_lock(hid_prod_lock, aie.LockAction.Release, value = 1)
#             aie.next_bd(block[11])

#         with block[11]:
#             aie.use_lock(hid_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
#             aie.dma_bd_packet(0, 1)
#             aie.dma_bd(hid_buffer_1)
#             aie.use_lock(hid_prod_lock, aie.LockAction.Release, value = 1)
#             aie.next_bd(block[10])

#         with block[12]:
#             aie.EndOp()
            
            

# def glu_column(CT: tile_with_location, MT: tile_with_location, ctx: mlir_context):

#     glu_memtile(MT, ctx)
#     glu_tile(CT, ctx)
#     connect_dma(MT, CT, 0, 0)
#     connect_dma(CT, MT, 0, 0)