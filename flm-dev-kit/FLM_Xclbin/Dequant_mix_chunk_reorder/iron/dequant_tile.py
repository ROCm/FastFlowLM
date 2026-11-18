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

def dequant_tile(tile: tile_with_location, mctx: mlir_context):

    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    dequant = mctx.dequant
    w_ty = mctx.w_ty
    w8_ty = mctx.w8_ty
    w16_ty = mctx.w16_ty

    w_buffer_0 = aie.buffer(tile = ct, datatype = w_ty, name = f"w_0_{ct_row}_{ct_col}")
    w_buffer_1 = aie.buffer(tile = ct, datatype = w_ty, name = f"w_1_{ct_row}_{ct_col}")
    w8_buffer = aie.buffer(tile = ct, datatype = w8_ty, name = f"w8_{ct_row}_{ct_col}")
    w16_buffer_0 = aie.buffer(tile = ct, datatype = w16_ty, name = f"y_0_{ct_row}_{ct_col}")
    w16_buffer_1 = aie.buffer(tile = ct, datatype = w16_ty, name = f"y_1_{ct_row}_{ct_col}")
    scale_buffer = aie.buffer(tile = ct, datatype = w_ty, name = f"scale_buffer_{ct_row}_{ct_col}")
    zp_buffer = aie.buffer(tile = ct, datatype = w_ty, name = f"zp_buffer_{ct_row}_{ct_col}")
    rtp_buffer = mctx.buffers[f"rtp_{ct_row}_{ct_col}"]
    rtp_transpose_buffer = mctx.buffers[f"rtp_transpose_{ct_row}_{ct_col}"]
    Din_buffer = mctx.buffers[f"Din_{ct_row}_{ct_col}"]
    Dout_buffer = mctx.buffers[f"Dout_{ct_row}_{ct_col}"]
    
    w_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    w_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    y_prod_ping_lock = aie.lock(ct, lock_id = 4, init = 1)
    y_cons_ping_lock = aie.lock(ct, lock_id = 7, init = 0)
    y_prod_pong_lock = aie.lock(ct, lock_id = 5, init = 1)
    y_cons_pong_lock = aie.lock(ct, lock_id = 8, init = 0)
    
    @aie.core(ct, link_with = "dequant.o", stack_size = 1024 * 8)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            dequant(w16_buffer_0, w_buffer_0, w16_buffer_1, w_buffer_1, scale_buffer, zp_buffer, w8_buffer, Din_buffer, Dout_buffer, rtp_buffer, rtp_transpose_buffer)
    
    @aie.mem(ct)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

        with block[1]:
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w_buffer_0)
            lock_release(w_cons_lock, 1)
            aie.next_bd(block[2])
        
        with block[2]:
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w_buffer_1)
            lock_release(w_cons_lock, 1)
            aie.next_bd(block[1])
            
        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

        with block[4]:
            lock_acquire(y_cons_ping_lock, 1)
            aie.dma_bd(w16_buffer_0)
            lock_release(y_prod_ping_lock, 1)
            aie.next_bd(block[5])

        with block[5]:
            lock_acquire(y_cons_pong_lock, 1)
            aie.dma_bd(w16_buffer_1)
            lock_release(y_prod_pong_lock, 1)
            aie.next_bd(block[4])
        
        with block[6]:
            aie.EndOp()

def dequant_memtile(tile: tile_with_location, mctx: mlir_context):

    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col
 
    w_mem_ty = mctx.w_mem_ty
    w16_mem_ty = mctx.w16_mem_ty

    W_BLOCK_SIZE        = mctx.W_BLOCK_SIZE
    W_SIZE        = mctx.W_SIZE
    Q4NX_ROW_BLOCK_SIZE = mctx.Q4NX_ROW_BLOCK_SIZE
    Q4NX_COL_BLOCK_SIZE = mctx.Q4NX_COL_BLOCK_SIZE

    # channel 0
    w_buffer_0 = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_0_{mt_row}_{mt_col}")
    w_buffer_1 = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_1_{mt_row}_{mt_col}")
    w_buffer_2 = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_2_{mt_row}_{mt_col}")
    w_buffer_3 = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_3_{mt_row}_{mt_col}")

    
    # channel 1
    w_buffer_4 = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_4_{mt_row}_{mt_col}")
    w_buffer_5 = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_5_{mt_row}_{mt_col}")
    w_buffer_6 = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_6_{mt_row}_{mt_col}")
    w_buffer_7 = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_7_{mt_row}_{mt_col}")


    w16_buffer_0 = aie.buffer(tile = mt, datatype = w16_mem_ty, name = f"y_0_{mt_row}_{mt_col}")
    w16_buffer_1 = aie.buffer(tile = mt, datatype = w16_mem_ty, name = f"y_1_{mt_row}_{mt_col}")
    w16_buffer_2 = aie.buffer(tile = mt, datatype = w16_mem_ty, name = f"y_2_{mt_row}_{mt_col}")
    w16_buffer_3 = aie.buffer(tile = mt, datatype = w16_mem_ty, name = f"y_3_{mt_row}_{mt_col}")

    w_prod_lock_0 = aie.lock(mt, lock_id = 0, init = 4)
    w_cons_lock_0 = aie.lock(mt, lock_id = 1, init = 0)
    w_cons_lock_1 = aie.lock(mt, lock_id = 2, init = 0)

    w_prod_lock_1 = aie.lock(mt, lock_id = 5, init = 4)
    w_cons_lock_4 = aie.lock(mt, lock_id = 6, init = 0)
    w_cons_lock_5 = aie.lock(mt, lock_id = 7, init = 0)

    y_prod_lock_0 = aie.lock(mt, lock_id = 10, init = 2)
    y_prod_lock_1 = aie.lock(mt, lock_id = 11, init = 0)
    y_prod_lock_2 = aie.lock(mt, lock_id = 12, init = 2)
    y_prod_lock_3 = aie.lock(mt, lock_id = 13, init = 0)
    y_cons_lock_0 = aie.lock(mt, lock_id = 14, init = 0)
    y_cons_lock_1 = aie.lock(mt, lock_id = 15, init = 0)
    
    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])
        
        with block[1]:
            lock_acquire(w_prod_lock_0, 1)
            aie.dma_bd(w_buffer_0)
            lock_release(w_cons_lock_0, 1)
            aie.next_bd(block[2])
        
        with block[2]:
            lock_acquire(w_prod_lock_0, 1)
            aie.dma_bd(w_buffer_1)
            lock_release(w_cons_lock_0, 1)
            aie.next_bd(block[117])

        with block[117]:
            lock_acquire(w_prod_lock_0, 1)
            aie.dma_bd(w_buffer_2)
            lock_release(w_cons_lock_0, 1)
            aie.next_bd(block[118])
        
        with block[118]:
            lock_acquire(w_prod_lock_0, 1)
            aie.dma_bd(w_buffer_3)
            lock_release(w_cons_lock_0, 1)
            aie.next_bd(block[1])

        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])
        
        with block[4]:
            lock_acquire(w_prod_lock_1, 1)
            aie.dma_bd(w_buffer_4)
            lock_release(w_cons_lock_4, 1)
            aie.next_bd(block[5])
        
        with block[5]:
            lock_acquire(w_prod_lock_1, 1)
            aie.dma_bd(w_buffer_5)
            lock_release(w_cons_lock_4, 1)
            aie.next_bd(block[127])

        with block[127]:
            lock_acquire(w_prod_lock_1, 1)
            aie.dma_bd(w_buffer_6)
            lock_release(w_cons_lock_4, 1)
            aie.next_bd(block[128])
        
        with block[128]:
            lock_acquire(w_prod_lock_1, 1)
            aie.dma_bd(w_buffer_7)
            lock_release(w_cons_lock_4, 1)
            aie.next_bd(block[4])

        with block[6]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[7], chain=block[9])
        
        with block[7]:
            lock_acquire(w_cons_lock_0, 1)
            aie.dma_bd(w_buffer_0, offset = 0, len = W_BLOCK_SIZE)
            lock_release(w_cons_lock_1, 1)
            aie.next_bd(block[8])
        
        with block[8]:
            lock_acquire(w_cons_lock_0, 1)
            aie.dma_bd(w_buffer_1, offset = 0, len = W_BLOCK_SIZE)
            lock_release(w_cons_lock_1, 1)
            aie.next_bd(block[210])

        with block[210]:
            lock_acquire(w_cons_lock_0, 1)
            aie.dma_bd(w_buffer_2, offset = 0, len = W_BLOCK_SIZE)
            lock_release(w_cons_lock_1, 1)
            aie.next_bd(block[211])
        
        with block[211]:
            lock_acquire(w_cons_lock_0, 1)
            aie.dma_bd(w_buffer_3, offset = 0, len = W_BLOCK_SIZE)
            lock_release(w_cons_lock_1, 1)
            aie.next_bd(block[7])

        with block[9]:
            s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[10], chain=block[12])
        
        with block[10]:
            lock_acquire(w_cons_lock_1, 1)
            aie.dma_bd(w_buffer_0, offset = W_BLOCK_SIZE, len = W_BLOCK_SIZE)
            lock_release(w_prod_lock_0, 1)
            aie.next_bd(block[11])
        
        with block[11]:
            lock_acquire(w_cons_lock_1, 1)
            aie.dma_bd(w_buffer_1, offset = W_BLOCK_SIZE, len = W_BLOCK_SIZE)
            lock_release(w_prod_lock_0, 1)
            aie.next_bd(block[313])
        
        with block[313]:
            lock_acquire(w_cons_lock_1, 1)
            aie.dma_bd(w_buffer_2, offset = W_BLOCK_SIZE, len = W_BLOCK_SIZE)
            lock_release(w_prod_lock_0, 1)
            aie.next_bd(block[314])
        
        with block[314]:
            lock_acquire(w_cons_lock_1, 1)
            aie.dma_bd(w_buffer_3, offset = W_BLOCK_SIZE, len = W_BLOCK_SIZE)
            lock_release(w_prod_lock_0, 1)
            aie.next_bd(block[10])
        
        with block[12]:
            s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[13], chain=block[15])
        
        with block[13]:
            lock_acquire(w_cons_lock_4, 1)
            aie.dma_bd(w_buffer_4, offset = 0, len = W_BLOCK_SIZE)
            lock_release(w_cons_lock_5, 1)
            aie.next_bd(block[14])
        
        with block[14]:
            lock_acquire(w_cons_lock_4, 1)
            aie.dma_bd(w_buffer_5, offset = 0, len = W_BLOCK_SIZE)
            lock_release(w_cons_lock_5, 1)
            aie.next_bd(block[416])
        
        with block[416]:
            lock_acquire(w_cons_lock_4, 1)
            aie.dma_bd(w_buffer_6, offset = 0, len = W_BLOCK_SIZE)
            lock_release(w_cons_lock_5, 1)
            aie.next_bd(block[417])
        
        with block[417]:
            lock_acquire(w_cons_lock_4, 1)
            aie.dma_bd(w_buffer_7, offset = 0, len = W_BLOCK_SIZE)
            lock_release(w_cons_lock_5, 1)
            aie.next_bd(block[13])
        
        with block[15]:
            s5 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 3, dest=block[16], chain=block[18])

        with block[16]:
            lock_acquire(w_cons_lock_5, 1)
            aie.dma_bd(w_buffer_4, offset = W_BLOCK_SIZE, len = W_BLOCK_SIZE)
            lock_release(w_prod_lock_1, 1)
            aie.next_bd(block[17])
        
        with block[17]:
            lock_acquire(w_cons_lock_5, 1)
            aie.dma_bd(w_buffer_5, offset = W_BLOCK_SIZE, len = W_BLOCK_SIZE)
            lock_release(w_prod_lock_1, 1)
            aie.next_bd(block[519])
            
        with block[519]:
            lock_acquire(w_cons_lock_5, 1)
            aie.dma_bd(w_buffer_6, offset = W_BLOCK_SIZE, len = W_BLOCK_SIZE)
            lock_release(w_prod_lock_1, 1)
            aie.next_bd(block[520])
        
        with block[520]:
            lock_acquire(w_cons_lock_5, 1)
            aie.dma_bd(w_buffer_7, offset = W_BLOCK_SIZE, len = W_BLOCK_SIZE)
            lock_release(w_prod_lock_1, 1)
            aie.next_bd(block[16])
            
        with block[18]:
            s7 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[22], chain=block[24])
        
        with block[22]:
            lock_acquire(y_prod_lock_0, 1)
            aie.dma_bd(w16_buffer_0, offset = 0, len = W_SIZE, dimensions = [(Q4NX_ROW_BLOCK_SIZE // 8, 8 * Q4NX_COL_BLOCK_SIZE), (Q4NX_COL_BLOCK_SIZE // 8, 8), (8, Q4NX_COL_BLOCK_SIZE), (8, 1)])
            lock_release(y_prod_lock_1, 1)
            aie.next_bd(block[23])
            
        with block[23]:
            lock_acquire(y_prod_lock_0, 1)
            aie.dma_bd(w16_buffer_1, offset = 0, len = W_SIZE, dimensions = [(Q4NX_ROW_BLOCK_SIZE // 8, 8 * Q4NX_COL_BLOCK_SIZE), (Q4NX_COL_BLOCK_SIZE // 8, 8), (8, Q4NX_COL_BLOCK_SIZE), (8, 1)])
            lock_release(y_prod_lock_1, 1)
            aie.next_bd(block[22])
        
        with block[24]:
            s8 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest=block[25], chain=block[27])
        
        with block[25]:
            lock_acquire(y_prod_lock_1, 1)
            aie.dma_bd(w16_buffer_0, offset = W_SIZE, len = W_SIZE, dimensions = [(Q4NX_ROW_BLOCK_SIZE // 8, 8 * Q4NX_COL_BLOCK_SIZE), (Q4NX_COL_BLOCK_SIZE // 8, 8), (8, Q4NX_COL_BLOCK_SIZE), (8, 1)])
            lock_release(y_cons_lock_0, 1)
            aie.next_bd(block[26])
        
        with block[26]:
            lock_acquire(y_prod_lock_1, 1)
            aie.dma_bd(w16_buffer_1, offset = W_SIZE, len = W_SIZE, dimensions = [(Q4NX_ROW_BLOCK_SIZE // 8, 8 * Q4NX_COL_BLOCK_SIZE), (Q4NX_COL_BLOCK_SIZE // 8, 8), (8, Q4NX_COL_BLOCK_SIZE), (8, 1)])
            lock_release(y_cons_lock_0, 1)
            aie.next_bd(block[25])
        
        with block[27]:
            s9 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 4, dest=block[28], chain=block[30])
        
        with block[28]:
            lock_acquire(y_prod_lock_2, 1)
            aie.dma_bd(w16_buffer_2, offset = 0, len = W_SIZE, dimensions = [(Q4NX_ROW_BLOCK_SIZE // 8, 8 * Q4NX_COL_BLOCK_SIZE), (Q4NX_COL_BLOCK_SIZE // 8, 8), (8, Q4NX_COL_BLOCK_SIZE), (8, 1)])
            lock_release(y_prod_lock_3, 1)
            aie.next_bd(block[29])
        
        with block[29]:
            lock_acquire(y_prod_lock_2, 1)
            aie.dma_bd(w16_buffer_3, offset = 0, len = W_SIZE, dimensions = [(Q4NX_ROW_BLOCK_SIZE // 8, 8 * Q4NX_COL_BLOCK_SIZE), (Q4NX_COL_BLOCK_SIZE // 8, 8), (8, Q4NX_COL_BLOCK_SIZE), (8, 1)])
            lock_release(y_prod_lock_3, 1)
            aie.next_bd(block[28])
        
        with block[30]:
            s10 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 5, dest=block[31], chain=block[33])
        
        with block[31]:
            lock_acquire(y_prod_lock_3, 1)
            aie.dma_bd(w16_buffer_2, offset = W_SIZE, len = W_SIZE, dimensions = [(Q4NX_ROW_BLOCK_SIZE // 8, 8 * Q4NX_COL_BLOCK_SIZE), (Q4NX_COL_BLOCK_SIZE // 8, 8), (8, Q4NX_COL_BLOCK_SIZE), (8, 1)])
            lock_release(y_cons_lock_1, 1)
            aie.next_bd(block[32])
        
        with block[32]:
            lock_acquire(y_prod_lock_3, 1)
            aie.dma_bd(w16_buffer_3, offset = W_SIZE, len = W_SIZE, dimensions = [(Q4NX_ROW_BLOCK_SIZE // 8, 8 * Q4NX_COL_BLOCK_SIZE), (Q4NX_COL_BLOCK_SIZE // 8, 8), (8, Q4NX_COL_BLOCK_SIZE), (8, 1)])
            lock_release(y_cons_lock_1, 1)
            aie.next_bd(block[31])
        
        with block[33]:
            s11 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 4, dest=block[34], chain=block[36])
        
        with block[34]:
            lock_acquire(y_cons_lock_0, 1)
            aie.dma_bd(w16_buffer_0, offset = 0, len = W_SIZE * 2)
            lock_release(y_prod_lock_0, 1)
            aie.next_bd(block[35])
        
        with block[35]:
            lock_acquire(y_cons_lock_0, 1)
            aie.dma_bd(w16_buffer_1, offset = 0, len = W_SIZE * 2)
            lock_release(y_prod_lock_0, 1)
            aie.next_bd(block[34])

        with block[36]:
            s12 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 5, dest=block[37], chain=block[39])
        
        with block[37]:
            lock_acquire(y_cons_lock_1, 1)
            aie.dma_bd(w16_buffer_2, offset = 0, len = W_SIZE * 2)
            lock_release(y_prod_lock_2, 1)
            aie.next_bd(block[38])
        
        with block[38]:
            lock_acquire(y_cons_lock_1, 1)
            aie.dma_bd(w16_buffer_3, offset = 0, len = W_SIZE * 2)
            lock_release(y_prod_lock_2, 1)
            aie.next_bd(block[37])

        with block[39]:
            aie.EndOp()
