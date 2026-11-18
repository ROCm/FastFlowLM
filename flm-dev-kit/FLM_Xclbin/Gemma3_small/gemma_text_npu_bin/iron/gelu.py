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

def gelu_tile(tile: tile_with_location, ctx: mlir_context):

    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    gelu = ctx.gelu
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

    @aie.core(ct, link_with = "gelu.o")
    def core_body():
        for _ in range_(0xFFFFFFFF):
            gelu(hid_buffer_0, hid_buffer_1, x_buffer_0, x_buffer_1)
    
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