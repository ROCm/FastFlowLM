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
    w_buffer = aie.buffer(tile = ct, datatype = w_ty, name = f"w_{ct_row}_{ct_col}")
    x_buffer = aie.buffer(tile = ct, datatype = x_ty, name = f"x_{ct_row}_{ct_col}")
    
    x_prod_lock = aie.lock(ct, lock_id = 0, init = 2)
    x_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    w_prod_lock = aie.lock(ct, lock_id = 2, init = 1)
    w_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    o_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    o_cons_lock = aie.lock(ct, lock_id = 5, init = 0)

    @aie.core(ct, link_with = "conv.o", stack_size = 1024*8)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            conv(o_buffer_0, o_buffer_1, x_buffer_0, x_buffer_1, x_buffer, w_buffer, L_buffer, ct_row - 2)

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
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[5])
        
        with block[4]:
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w_buffer)
            lock_release(w_cons_lock, 1)
            aie.next_bd(block[4])
        
        with block[5]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[6], chain=block[8])
        
        with block[6]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_0)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[7])
        
        with block[7]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_1)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[6])
        
        with block[8]:
            aie.EndOp()

def conv_memtile(tile: tile_with_location, mctx: mlir_context):
    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    x_mem_ty = mctx.x_mem_ty
    w_mem_ty = mctx.w_mem_ty
    o_mem_ty = mctx.o_mem_ty
    
    d = mctx.d

    x_buffer_0 = aie.buffer(tile = mt, datatype = x_mem_ty, name = f"x_0_{mt_row}_{mt_col}")
    x_buffer_1 = aie.buffer(tile = mt, datatype = x_mem_ty, name = f"x_1_{mt_row}_{mt_col}")
    w_buffer = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_{mt_row}_{mt_col}")
    o_buffer_0 = aie.buffer(tile = mt, datatype = o_mem_ty, name = f"o_0_{mt_row}_{mt_col}")
    o_buffer_1 = aie.buffer(tile = mt, datatype = o_mem_ty, name = f"o_1_{mt_row}_{mt_col}")

    x_prod_lock = aie.lock(mt, lock_id = 0, init = 2)
    x_cons_lock_0 = aie.lock(mt, lock_id = 1, init = 0)
    x_cons_lock_1 = aie.lock(mt, lock_id = 2, init = 0)
    x_cons_lock_2 = aie.lock(mt, lock_id = 3, init = 0)
    x_cons_lock_3 = aie.lock(mt, lock_id = 4, init = 0)
    w_prod_lock = aie.lock(mt, lock_id = 5, init = 1)
    w_cons_lock = aie.lock(mt, lock_id = 6, init = 0)
    o_prod_lock_0 = aie.lock(mt, lock_id = 7, init = 2)
    o_prod_lock_1 = aie.lock(mt, lock_id = 8, init = 0)
    o_prod_lock_2 = aie.lock(mt, lock_id = 9, init = 0)
    o_prod_lock_3 = aie.lock(mt, lock_id = 10, init = 0)
    o_cons_lock = aie.lock(mt, lock_id = 11, init = 0)

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])
        
        with block[1]:
            lock_acquire(x_prod_lock, 1)
            aie.dma_bd(x_buffer_0)
            lock_release(x_cons_lock_0, 1)
            aie.next_bd(block[2])
        
        with block[2]:
            lock_acquire(x_prod_lock, 1)
            aie.dma_bd(x_buffer_1)
            lock_release(x_cons_lock_0, 1)
            aie.next_bd(block[1])
        
        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])
        
        with block[4]:
            lock_acquire(x_cons_lock_0, 1)
            aie.dma_bd(x_buffer_0, offset = 0, len = d)
            lock_release(x_cons_lock_1, 1)
            aie.next_bd(block[5])
        
        with block[5]:
            lock_acquire(x_cons_lock_0, 1)
            aie.dma_bd(x_buffer_1, offset = 0, len = d)
            lock_release(x_cons_lock_1, 1)
            aie.next_bd(block[4])
        
        with block[6]:
            s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[7], chain=block[9])
        
        with block[7]:
            lock_acquire(x_cons_lock_1, 1)
            aie.dma_bd(x_buffer_0, offset = d, len = d)
            lock_release(x_cons_lock_2, 1)
            aie.next_bd(block[8])
            
        with block[8]:
            lock_acquire(x_cons_lock_1, 1)
            aie.dma_bd(x_buffer_1, offset = d, len = d)
            lock_release(x_cons_lock_2, 1)
            aie.next_bd(block[7])
        
        with block[9]:
            s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[10], chain=block[12])
        
        with block[10]:
            lock_acquire(x_cons_lock_2, 1)
            aie.dma_bd(x_buffer_0, offset = 2 * d, len = d)
            lock_release(x_cons_lock_3, 1)
            aie.next_bd(block[11])
        
        with block[11]:
            lock_acquire(x_cons_lock_2, 1)
            aie.dma_bd(x_buffer_1, offset = 2 * d, len = d)
            lock_release(x_cons_lock_3, 1)
            aie.next_bd(block[10])
        
        with block[12]:
            s5 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 3, dest=block[13], chain=block[15])
        
        with block[13]:
            lock_acquire(x_cons_lock_3, 1)
            aie.dma_bd(x_buffer_0, offset = 3 * d, len = d)
            lock_release(x_prod_lock, 1)
            aie.next_bd(block[14])
        
        with block[14]:
            lock_acquire(x_cons_lock_3, 1)
            aie.dma_bd(x_buffer_1, offset = 3 * d, len = d)
            lock_release(x_prod_lock, 1)
            aie.next_bd(block[13])
        
        with block[15]:
            s6 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[16], chain=block[17])
        
        with block[16]:
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w_buffer)
            lock_release(w_cons_lock, 1)
            aie.next_bd(block[16])
            
        with block[17]:
            s7 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 4, dest=block[18], chain=block[19])
        
        with block[18]:
            lock_acquire(w_cons_lock, 1)
            aie.dma_bd(w_buffer)
            lock_release(w_prod_lock, 1)
            aie.next_bd(block[18])
        
        with block[19]:
            s8 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[20], chain=block[22])
        
        with block[20]:
            lock_acquire(o_prod_lock_0, 1)
            aie.dma_bd(o_buffer_0, offset = 0, len = d)
            lock_release(o_prod_lock_1, 1)
            aie.next_bd(block[21])

        with block[21]:
            lock_acquire(o_prod_lock_0, 1)
            aie.dma_bd(o_buffer_1, offset = 0, len = d)
            lock_release(o_prod_lock_1, 1)
            aie.next_bd(block[20])
        
        with block[22]:
            s9 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest=block[23], chain=block[25])
        
        with block[23]:
            lock_acquire(o_prod_lock_1, 1)
            aie.dma_bd(o_buffer_0, offset = d, len = d)
            lock_release(o_prod_lock_2, 1)
            aie.next_bd(block[24])
        
        with block[24]:
            lock_acquire(o_prod_lock_1, 1)
            aie.dma_bd(o_buffer_1, offset = d, len = d)
            lock_release(o_prod_lock_2, 1)
            aie.next_bd(block[23])
        
        with block[25]:
            s10 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 4, dest=block[26], chain=block[28])
        
        with block[26]:
            lock_acquire(o_prod_lock_2, 1)
            aie.dma_bd(o_buffer_0, offset = 2 * d, len = d)
            lock_release(o_prod_lock_3, 1)
            aie.next_bd(block[27])
        
        with block[27]:
            lock_acquire(o_prod_lock_2, 1)
            aie.dma_bd(o_buffer_1, offset = 2 * d, len = d)
            lock_release(o_prod_lock_3, 1)
            aie.next_bd(block[26])
        
        with block[28]:
            s11 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 5, dest=block[29], chain=block[31])
        
        with block[29]:
            lock_acquire(o_prod_lock_3, 1)
            aie.dma_bd(o_buffer_0, offset = 3 * d, len = d)
            lock_release(o_cons_lock, 1)
            aie.next_bd(block[30])
        
        with block[30]:
            lock_acquire(o_prod_lock_3, 1)
            aie.dma_bd(o_buffer_1, offset = 3 * d, len = d)
            lock_release(o_cons_lock, 1)
            aie.next_bd(block[29])
        
        with block[31]:
            s12 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 5, dest=block[32], chain=block[34])
        
        with block[32]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_0, offset = 0, len = d * 4)
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[33])
        
        with block[33]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_1, offset = 0, len = d * 4)
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[32])
        
        with block[34]:
            aie.EndOp()
