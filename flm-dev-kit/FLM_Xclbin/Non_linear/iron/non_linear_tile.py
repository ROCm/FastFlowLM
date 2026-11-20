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

# send s
def non_linear_tile(tile: tile_with_location, mctx: mlir_context):
    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    non_linear = mctx.non_linear
    
    o_ty = mctx.o_ty
    x_ty = mctx.x_ty
    D = mctx.D

    x_buffer_0 = aie.buffer(tile = ct, datatype = x_ty, name = f"x_0_{ct_row}_{ct_col}")
    x_buffer_1 = aie.buffer(tile = ct, datatype = x_ty, name = f"x_1_{ct_row}_{ct_col}")
    o_buffer_0 = aie.buffer(tile = ct, datatype = o_ty, name = f"o_0_{ct_row}_{ct_col}")
    o_buffer_1 = aie.buffer(tile = ct, datatype = o_ty, name = f"o_1_{ct_row}_{ct_col}")
    rtp_activate_func_impl_buffer = mctx.buffers[f"rtp_activate_func_impl_{ct_row}_{ct_col}"]
    
    x_prod_lock = aie.lock(ct, lock_id = 0, init = 2)
    x_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    o_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    o_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    l_cons_lock = aie.lock(ct, lock_id = 4, init = 0)
    
    @aie.core(ct, link_with = "non_linear.o", stack_size = 1024*2)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            non_linear(o_buffer_0, o_buffer_1, x_buffer_0, x_buffer_1, rtp_activate_func_impl_buffer, D)

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

def non_linear_memtile(tile: tile_with_location, mctx: mlir_context):
    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    x_mem_ty = mctx.x_mem_ty
    o_mem_ty = mctx.o_mem_ty
    
    D = mctx.D

    x_buffer_0 = aie.buffer(tile = mt, datatype = x_mem_ty, name = f"x_0_{mt_row}_{mt_col}")
    x_buffer_1 = aie.buffer(tile = mt, datatype = x_mem_ty, name = f"x_1_{mt_row}_{mt_col}")
    o_buffer_0 = aie.buffer(tile = mt, datatype = o_mem_ty, name = f"o_0_{mt_row}_{mt_col}")
    o_buffer_1 = aie.buffer(tile = mt, datatype = o_mem_ty, name = f"o_1_{mt_row}_{mt_col}")

    x_prod_lock = aie.lock(mt, lock_id = 0, init = 2)
    x_cons_lock = aie.lock(mt, lock_id = 1, init = 0)
    o_prod_lock = aie.lock(mt, lock_id = 2, init = 2)
    o_cons_lock = aie.lock(mt, lock_id = 3, init = 0)

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
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
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])
        
        with block[4]:
            lock_acquire(x_cons_lock, 1)
            aie.dma_bd(x_buffer_0, offset = 0, len = D)
            lock_release(x_prod_lock, 1)
            aie.next_bd(block[5])
        
        with block[5]:
            lock_acquire(x_cons_lock, 1)
            aie.dma_bd(x_buffer_1, offset = 0, len = D)
            lock_release(x_prod_lock, 1)
            aie.next_bd(block[4])
        
        with block[6]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[7], chain=block[9])
        
        with block[7]:
            lock_acquire(o_prod_lock, 1)
            aie.dma_bd(o_buffer_0, offset = 0, len = D)
            lock_release(o_cons_lock, 1)
            aie.next_bd(block[8])

        with block[8]:
            lock_acquire(o_prod_lock, 1)
            aie.dma_bd(o_buffer_1, offset = 0, len = D)
            lock_release(o_cons_lock, 1)
            aie.next_bd(block[7])
        
        with block[9]:
            s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[10], chain=block[12])
        
        with block[10]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_0, offset = 0, len = D)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[11])
        
        with block[11]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_1, offset = 0, len = D)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[10])
        
        with block[12]:
            aie.EndOp()
