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

def lm_head_tile(tile: tile_with_location, mctx: mlir_context):

    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    qmm = mctx.qmm
    x_ty = mctx.x_ty
    w_ty = mctx.w_ty
    y_ty = mctx.y_ty
    rtp_ty = mctx.rtp_ty

    x_buffer   = aie.buffer(tile = ct, datatype = x_ty, name = f"x_{ct_row}_{ct_col}")
    w_buffer_0 = aie.buffer(tile = ct, datatype = w_ty, name = f"w_0_{ct_row}_{ct_col}")
    w_buffer_1 = aie.buffer(tile = ct, datatype = w_ty, name = f"w_1_{ct_row}_{ct_col}")
    y_buffer_0 = aie.buffer(tile = ct, datatype = y_ty, name = f"y_0_{ct_row}_{ct_col}")
    y_buffer_1 = aie.buffer(tile = ct, datatype = y_ty, name = f"y_1_{ct_row}_{ct_col}")
    rtp_buffer = aie.buffer(tile = ct, datatype = rtp_ty, name = f"rtp_{ct_row}_{ct_col}")
    
    x_prod_lock = aie.lock(ct, lock_id = 0, init = 1)
    x_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    w_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    w_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    y_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    y_cons_lock = aie.lock(ct, lock_id = 5, init = 0)
    
    @aie.core(ct, link_with = "qmm.o", stack_size = 1024 * 10)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            qmm(y_buffer_0, y_buffer_1, w_buffer_0, w_buffer_1, x_buffer, rtp_buffer)
    
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

def lm_head_memtile(tile: tile_with_location, mctx: mlir_context):

    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    x_ty     = mctx.x_ty    
    w_mem_ty = mctx.w_mem_ty
    y_mem_ty = mctx.y_mem_ty

    W_BLOCK_SIZE        = mctx.W_BLOCK_SIZE
    Q4NX_ROW_BLOCK_SIZE = mctx.Q4NX_ROW_BLOCK_SIZE

    x_buffer   = aie.buffer(tile = mt, datatype = x_ty, name = f"x_{mt_row}_{mt_col}")
    w_buffer_0 = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_0_{mt_row}_{mt_col}")
    w_buffer_1 = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_1_{mt_row}_{mt_col}")
    y_buffer_0 = aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_0_{mt_row}_{mt_col}")
    y_buffer_1 = aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_1_{mt_row}_{mt_col}")

    x_prod_lock = aie.lock(mt, lock_id = 0, init = 1)
    x_cons_lock = aie.lock(mt, lock_id = 1, init = 0)
    w_prod_lock = aie.lock(mt, lock_id = 2, init = 2)
    w_cons_lock_0 = aie.lock(mt, lock_id = 3, init = 0)
    w_cons_lock_1 = aie.lock(mt, lock_id = 4, init = 0)
    w_cons_lock_2 = aie.lock(mt, lock_id = 5, init = 0)
    w_cons_lock_3 = aie.lock(mt, lock_id = 6, init = 0)
    y_prod_lock_0 = aie.lock(mt, lock_id = 7, init = 2)
    y_prod_lock_1 = aie.lock(mt, lock_id = 8, init = 0)
    y_prod_lock_2 = aie.lock(mt, lock_id = 9, init = 0)
    y_prod_lock_3 = aie.lock(mt, lock_id = 10, init = 0)
    y_cons_lock = aie.lock(mt, lock_id = 11, init = 0)
    
    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

        with block[1]:
            lock_acquire(x_prod_lock, 1)
            aie.dma_bd(x_buffer)
            lock_release(x_cons_lock, 1)
            aie.next_bd(block[1])
        
        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

        with block[4]:
            lock_acquire(x_cons_lock, 1)
            aie.dma_bd(x_buffer)
            lock_release(x_prod_lock, 1)
            aie.next_bd(block[4])
        
        with block[6]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[7], chain=block[9])
        
        with block[7]:
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w_buffer_0)
            lock_release(w_cons_lock_0, 1)
            aie.next_bd(block[8])
        
        with block[8]:
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w_buffer_1)
            lock_release(w_cons_lock_0, 1)
            aie.next_bd(block[7])
        
        with block[9]:
            s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[10], chain=block[12])
        
        with block[10]:
            lock_acquire(w_cons_lock_0, 1)
            aie.dma_bd(w_buffer_0, offset = 0, len = W_BLOCK_SIZE)
            lock_release(w_cons_lock_1, 1)
            aie.next_bd(block[11])
        
        with block[11]:
            lock_acquire(w_cons_lock_0, 1)
            aie.dma_bd(w_buffer_1, offset = 0, len = W_BLOCK_SIZE)
            lock_release(w_cons_lock_1, 1)
            aie.next_bd(block[10])

        with block[12]:
            s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[13], chain=block[15])
        
        with block[13]:
            lock_acquire(w_cons_lock_1, 1)
            aie.dma_bd(w_buffer_0, offset = W_BLOCK_SIZE, len = W_BLOCK_SIZE)
            lock_release(w_cons_lock_2, 1)
            aie.next_bd(block[14])
        
        with block[14]:
            lock_acquire(w_cons_lock_1, 1)
            aie.dma_bd(w_buffer_1, offset = W_BLOCK_SIZE, len = W_BLOCK_SIZE)
            lock_release(w_cons_lock_2, 1)
            aie.next_bd(block[13])
        
        with block[15]:
            s5 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 3, dest=block[16], chain=block[18])
        
        with block[16]:
            lock_acquire(w_cons_lock_2, 1)
            aie.dma_bd(w_buffer_0, offset = W_BLOCK_SIZE * 2, len = W_BLOCK_SIZE)
            lock_release(w_cons_lock_3, 1)
            aie.next_bd(block[17])
        
        with block[17]:
            lock_acquire(w_cons_lock_2, 1)
            aie.dma_bd(w_buffer_1, offset = W_BLOCK_SIZE * 2, len = W_BLOCK_SIZE)
            lock_release(w_cons_lock_3, 1)
            aie.next_bd(block[16])
        
        with block[18]:
            s6 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 4, dest=block[19], chain=block[21])

        with block[19]:
            lock_acquire(w_cons_lock_3, 1)
            aie.dma_bd(w_buffer_0, offset = W_BLOCK_SIZE * 3, len = W_BLOCK_SIZE)
            lock_release(w_prod_lock, 1)
            aie.next_bd(block[20])
        
        with block[20]:
            lock_acquire(w_cons_lock_3, 1)
            aie.dma_bd(w_buffer_1, offset = W_BLOCK_SIZE * 3, len = W_BLOCK_SIZE)
            lock_release(w_prod_lock, 1)
            aie.next_bd(block[19])
            
        with block[21]:
            s7 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[22], chain=block[24])
        
        with block[22]:
            lock_acquire(y_prod_lock_0, 1)
            aie.dma_bd(y_buffer_0, offset = 0, len = Q4NX_ROW_BLOCK_SIZE)
            lock_release(y_prod_lock_1, 1)
            aie.next_bd(block[23])
            
        with block[23]:
            lock_acquire(y_prod_lock_0, 1)
            aie.dma_bd(y_buffer_1, offset = 0, len = Q4NX_ROW_BLOCK_SIZE)
            lock_release(y_prod_lock_1, 1)
            aie.next_bd(block[22])
        
        with block[24]:
            s8 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest=block[25], chain=block[27])
        
        with block[25]:
            lock_acquire(y_prod_lock_1, 1)
            aie.dma_bd(y_buffer_0, offset = Q4NX_ROW_BLOCK_SIZE, len = Q4NX_ROW_BLOCK_SIZE)
            lock_release(y_prod_lock_2, 1)
            aie.next_bd(block[26])
        
        with block[26]:
            lock_acquire(y_prod_lock_1, 1)
            aie.dma_bd(y_buffer_1, offset = Q4NX_ROW_BLOCK_SIZE, len = Q4NX_ROW_BLOCK_SIZE)
            lock_release(y_prod_lock_2, 1)
            aie.next_bd(block[25])
        
        with block[27]:
            s9 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 4, dest=block[28], chain=block[30])
        
        with block[28]:
            lock_acquire(y_prod_lock_2, 1)
            aie.dma_bd(y_buffer_0, offset = Q4NX_ROW_BLOCK_SIZE * 2, len = Q4NX_ROW_BLOCK_SIZE)
            lock_release(y_prod_lock_3, 1)
            aie.next_bd(block[29])
        
        with block[29]:
            lock_acquire(y_prod_lock_2, 1)
            aie.dma_bd(y_buffer_1, offset = Q4NX_ROW_BLOCK_SIZE * 2, len = Q4NX_ROW_BLOCK_SIZE)
            lock_release(y_prod_lock_3, 1)
            aie.next_bd(block[28])
        
        with block[30]:
            s10 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 5, dest=block[31], chain=block[33])
        
        with block[31]:
            lock_acquire(y_prod_lock_3, 1)
            aie.dma_bd(y_buffer_0, offset = Q4NX_ROW_BLOCK_SIZE * 3, len = Q4NX_ROW_BLOCK_SIZE)
            lock_release(y_cons_lock, 1)
            aie.next_bd(block[32])
        
        with block[32]:
            lock_acquire(y_prod_lock_3, 1)
            aie.dma_bd(y_buffer_1, offset = Q4NX_ROW_BLOCK_SIZE * 3, len = Q4NX_ROW_BLOCK_SIZE)
            lock_release(y_cons_lock, 1)
            aie.next_bd(block[31])
        
        with block[33]:
            s11 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 5, dest=block[34], chain=block[36])
        
        with block[34]:
            lock_acquire(y_cons_lock, 1)
            aie.dma_bd(y_buffer_0)
            lock_release(y_prod_lock_0, 1)
            aie.next_bd(block[35])
        
        with block[35]:
            lock_acquire(y_cons_lock, 1)
            aie.dma_bd(y_buffer_1)  
            lock_release(y_prod_lock_0, 1)
            aie.next_bd(block[34])

        with block[36]:
            aie.EndOp()


def lm_head_column(CTs: List[tile_with_location], MT: tile_with_location, IT: tile_with_location, XIT: tile_with_location, mctx: mlir_context):

    lm_head_memtile(MT, mctx)

    connect_dma(XIT, MT, 0, 0)
    connect_dma(IT, MT, 1, 1)

    for i, proj_core in enumerate(CTs):
        lm_head_tile(proj_core, mctx) 
        connect_dma(MT, proj_core, 0, 0)
        connect_dma(MT, proj_core, i + 1, 1)

        connect_dma(proj_core, MT, 0, i + 2)
    
    connect_dma(MT, IT, 5, 0)