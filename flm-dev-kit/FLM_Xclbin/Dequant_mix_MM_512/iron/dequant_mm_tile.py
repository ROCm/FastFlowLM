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

def dequant_mm_tile(tile: tile_with_location, mctx: mlir_context):

    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    if ct_row % 2 == 0:
        is_even = True
        link_with = "dequant_mm_e.o"
        dequant_mm = mctx.dequant_mm_e
    else:
        is_even = False
        link_with = "dequant_mm_o.o"
        dequant_mm = mctx.dequant_mm_o


    x_ty = mctx.x_ty
    qw_ty = mctx.qw_ty
    w8_ty = mctx.w8_ty
    w_ty = mctx.w_ty
    y_ty = mctx.y_ty
    y_acc_ty = mctx.y_acc_ty

    m_slice = mctx.m_slice
    k_slice = mctx.k_slice

    x_buffer_0 = aie.buffer(tile = ct, datatype = x_ty, name = f"x_0_{ct_row}_{ct_col}")
    x_buffer_1 = aie.buffer(tile = ct, datatype = x_ty, name = f"x_1_{ct_row}_{ct_col}")
    qw_buffer_0 = aie.buffer(tile = ct, datatype = qw_ty, name = f"w_0_{ct_row}_{ct_col}")
    qw_buffer_1 = aie.buffer(tile = ct, datatype = qw_ty, name = f"w_1_{ct_row}_{ct_col}")
    w8_buffer = aie.buffer(tile = ct, datatype = w8_ty, name = f"w8_{ct_row}_{ct_col}")
    scale_buffer = aie.buffer(tile = ct, datatype = qw_ty, name = f"scale_buffer_{ct_row}_{ct_col}")
    zp_buffer = aie.buffer(tile = ct, datatype = qw_ty, name = f"zp_buffer_{ct_row}_{ct_col}")
    y_buffer_0 = aie.buffer(tile = ct, datatype = y_ty, name = f"y_0_{ct_row}_{ct_col}")
    y_buffer_1 = aie.buffer(tile = ct, datatype = y_ty, name = f"y_1_{ct_row}_{ct_col}")
    y_acc_buffer = aie.buffer(tile = ct, datatype = y_acc_ty, name = f"y_acc_{ct_row}_{ct_col}")
    # Each tile of the (even, odd) row pair owns half of the dequantized weight
    # tile, and both cores are handed both halves -- the neighbour's one is
    # reached as its local memory.  Only the odd tile's half is used for now.
    w_buffer_e = mctx.buffers[f"w_{ct_row & ~1}_{ct_col}"]
    w_buffer_o = mctx.buffers[f"w_{ct_row | 1}_{ct_col}"]
    K_div_k_buffer = mctx.buffers[f"K_div_k_{ct_row}_{ct_col}"]
    rtp_activate = mctx.buffers[f"rtp_act_{ct_row}_{ct_col}"]
    rtp_q4 = mctx.buffers[f"rtp_q4_{ct_row}_{ct_col}"]
    rtp_zp = mctx.buffers[f"rtp_zp_{ct_row}_{ct_col}"]
    
    x_prod_lock = aie.lock(ct, lock_id = 0, init = 2)
    x_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    qw_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    qw_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    y_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    y_cons_lock = aie.lock(ct, lock_id = 5, init = 0)
    sync_lock = aie.lock(ct, lock_id = 6, init = 0)

    @aie.core(ct, link_with = link_with, stack_size = 1024 * 8)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            dequant_mm(y_buffer_0, y_buffer_1, x_buffer_0, x_buffer_1, qw_buffer_0, qw_buffer_1, w8_buffer, w_buffer_e, w_buffer_o, scale_buffer, zp_buffer, y_acc_buffer, K_div_k_buffer, rtp_activate, rtp_q4, rtp_zp)
    
    @aie.mem(ct)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

        with block[1]:
            lock_acquire(x_prod_lock, 1)
            aie.dma_bd(x_buffer_0, offset = 0, len = m_slice * k_slice, dimensions = [(m_slice, 8), (k_slice // 8, m_slice * 8), (8, 1)])
            lock_release(x_cons_lock, 1)
            aie.next_bd(block[2])
        
        with block[2]:
            lock_acquire(x_prod_lock, 1)
            aie.dma_bd(x_buffer_1, offset = 0, len = m_slice * k_slice, dimensions = [(m_slice, 8), (k_slice // 8, m_slice * 8), (8, 1)])
            lock_release(x_cons_lock, 1)
            aie.next_bd(block[1])
        
        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])

        with block[4]:
            lock_acquire(qw_prod_lock, 1)
            aie.dma_bd(qw_buffer_0)
            lock_release(qw_cons_lock, 1)
            aie.next_bd(block[5])
            
        with block[5]:
            lock_acquire(qw_prod_lock, 1)
            aie.dma_bd(qw_buffer_1)
            lock_release(qw_cons_lock, 1)
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

def dequant_mm_memtile(tile: tile_with_location, mctx: mlir_context):
    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    x_mem_ty = mctx.x_mem_ty
    qw_mem_ty = mctx.qw_mem_ty
    y_mem_ty = mctx.y_mem_ty

    m = mctx.m
    k = mctx.k
    n = mctx.n
    k_slice = mctx.k_slice
    W_BLOCK_SIZE = mctx.W_BLOCK_SIZE

    x_buffer_0 = aie.buffer(tile = mt, datatype = x_mem_ty, name = f"x_0_{mt_row}_{mt_col}")
    x_buffer_1 = aie.buffer(tile = mt, datatype = x_mem_ty, name = f"x_1_{mt_row}_{mt_col}")
    y_buffer_0 = aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_0_{mt_row}_{mt_col}")
    y_buffer_1 = aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_1_{mt_row}_{mt_col}")

    qw_buffer_0 = aie.buffer(tile = mt, datatype = qw_mem_ty, name = f"qw_0_{mt_row}_{mt_col}")
    qw_buffer_1 = aie.buffer(tile = mt, datatype = qw_mem_ty, name = f"qw_1_{mt_row}_{mt_col}")
    qw_buffer_2 = aie.buffer(tile = mt, datatype = qw_mem_ty, name = f"qw_2_{mt_row}_{mt_col}")
    qw_buffer_3 = aie.buffer(tile = mt, datatype = qw_mem_ty, name = f"qw_3_{mt_row}_{mt_col}")

    x_prod_lock = aie.lock(mt, lock_id = 0, init = 2)
    x_cons_lock = aie.lock(mt, lock_id = 1, init = 0)
    
    y_prod_lock_0 = aie.lock(mt, lock_id = 5, init = 2)
    y_prod_lock_1 = aie.lock(mt, lock_id = 6, init = 0)
    y_prod_lock_2 = aie.lock(mt, lock_id = 7, init = 0)
    y_prod_lock_3 = aie.lock(mt, lock_id = 8, init = 0)
    y_cons_lock = aie.lock(mt, lock_id = 9, init = 0)

    qw_prod_lock = aie.lock(mt, lock_id = 10, init = 4)
    qw_cons_lock_0 = aie.lock(mt, lock_id = 11, init = 0)
    qw_cons_lock_1 = aie.lock(mt, lock_id = 12, init = 0)

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
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[12])

        with block[4]:
            lock_acquire(x_cons_lock, 1)
            aie.dma_bd(x_buffer_0, offset = 0, len = m * k, dimensions = [(k // k_slice, k_slice), (m, k), (k_slice, 1)])
            lock_release(x_prod_lock, 1)
            aie.next_bd(block[5])

        with block[5]:
            lock_acquire(x_cons_lock, 1)
            aie.dma_bd(x_buffer_1, offset = 0, len = m * k, dimensions = [(k // k_slice, k_slice), (m, k), (k_slice, 1)])
            lock_release(x_prod_lock, 1)
            aie.next_bd(block[4])

        with block[12]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[13], chain=block[15])
        
        with block[13]:
            lock_acquire(y_prod_lock_0, 1) #, dimensions = [(m // 8, 512), (8, 8), (8, 64), (8, 1)]
            aie.dma_bd(y_buffer_0, offset = 0, len = m * n, dimensions = [(16, 256), (4, 8), (8, 32), (8, 1)])
            lock_release(y_prod_lock_1, 1)
            aie.next_bd(block[14])
        
        with block[14]:
            lock_acquire(y_prod_lock_0, 1)
            aie.dma_bd(y_buffer_1, offset = 0, len = m * n, dimensions = [(16, 256), (4, 8), (8, 32), (8, 1)])
            lock_release(y_prod_lock_1, 1)
            aie.next_bd(block[13])
        
        with block[15]:
            s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[16], chain=block[18])
        
        with block[16]:
            lock_acquire(y_prod_lock_1, 1)
            aie.dma_bd(y_buffer_0, offset = m * n, len = m * n, dimensions = [(16, 256), (4, 8), (8, 32), (8, 1)])
            lock_release(y_prod_lock_2, 1)
            aie.next_bd(block[17])
            
        with block[17]:
            lock_acquire(y_prod_lock_1, 1)
            aie.dma_bd(y_buffer_1, offset = m * n, len = m * n, dimensions = [(16, 256), (4, 8), (8, 32), (8, 1)])
            lock_release(y_prod_lock_2, 1)
            aie.next_bd(block[16])
        
        with block[18]:
            s6 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest=block[19], chain=block[21])
        
        with block[19]:
            lock_acquire(y_prod_lock_2, 1)
            aie.dma_bd(y_buffer_0, offset = 2 * m * n, len = m * n , dimensions = [(16, 256), (4, 8), (8, 32), (8, 1)])
            lock_release(y_prod_lock_3, 1)
            aie.next_bd(block[20])
        
        with block[20]:
            lock_acquire(y_prod_lock_2, 1)
            aie.dma_bd(y_buffer_1, offset = 2 * m * n, len = m * n, dimensions = [(16, 256), (4, 8), (8, 32), (8, 1)])
            lock_release(y_prod_lock_3, 1)
            aie.next_bd(block[19])
        
        with block[21]:
            s7 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 4, dest=block[22], chain=block[24])

        with block[22]:
            lock_acquire(y_prod_lock_3, 1)
            aie.dma_bd(y_buffer_0, offset = 3 * m * n, len = m * n, dimensions = [(16, 256), (4, 8), (8, 32), (8, 1)])
            lock_release(y_cons_lock, 1)
            aie.next_bd(block[23])
        
        with block[23]:
            lock_acquire(y_prod_lock_3, 1)
            aie.dma_bd(y_buffer_1, offset = 3 * m * n, len = m * n, dimensions = [(16, 256), (4, 8), (8, 32), (8, 1)])
            lock_release(y_cons_lock, 1)
            aie.next_bd(block[22])
        
        with block[24]:
            s8 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[25], chain=block[27])
        
        with block[25]:
            lock_acquire(y_cons_lock, 1)
            aie.dma_bd(y_buffer_0, offset = 0, len = 4 * m * n, dimensions = [(m, n), (4, m * n), (n, 1)])
            lock_release(y_prod_lock_0, 1)
            aie.next_bd(block[26])
            
        with block[26]:
            lock_acquire(y_cons_lock, 1)
            aie.dma_bd(y_buffer_1, offset = 0, len = 4 * m * n, dimensions = [(m, n), (4, m * n), (n, 1)])
            lock_release(y_prod_lock_0, 1)
            aie.next_bd(block[25])
        
        with block[27]:
            s9 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 5, dest=block[28], chain=block[30])
        
        with block[28]:
            lock_acquire(qw_prod_lock, 1)
            aie.dma_bd(qw_buffer_0)
            lock_release(qw_cons_lock_0, 1)
            aie.next_bd(block[29])
        
        with block[29]:
            lock_acquire(qw_prod_lock, 1)
            aie.dma_bd(qw_buffer_1)
            lock_release(qw_cons_lock_0, 1)
            aie.next_bd(block[117])

        with block[117]:
            lock_acquire(qw_prod_lock, 1)
            aie.dma_bd(qw_buffer_2)
            lock_release(qw_cons_lock_0, 1)
            aie.next_bd(block[118])
        
        with block[118]:
            lock_acquire(qw_prod_lock, 1)
            aie.dma_bd(qw_buffer_3)
            lock_release(qw_cons_lock_0, 1)
            aie.next_bd(block[28])

        with block[30]:
            s10 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[31], chain=block[33])
        
        with block[31]:
            lock_acquire(qw_cons_lock_0, 1)
            aie.dma_bd(qw_buffer_0, offset = 0, len = W_BLOCK_SIZE)
            lock_release(qw_cons_lock_1, 1)
            aie.next_bd(block[32])
        
        with block[32]:
            lock_acquire(qw_cons_lock_0, 1)
            aie.dma_bd(qw_buffer_1, offset = 0, len = W_BLOCK_SIZE)
            lock_release(qw_cons_lock_1, 1)
            aie.next_bd(block[210])

        with block[210]:
            lock_acquire(qw_cons_lock_0, 1)
            aie.dma_bd(qw_buffer_2, offset = 0, len = W_BLOCK_SIZE)
            lock_release(qw_cons_lock_1, 1)
            aie.next_bd(block[211])
        
        with block[211]:
            lock_acquire(qw_cons_lock_0, 1)
            aie.dma_bd(qw_buffer_3, offset = 0, len = W_BLOCK_SIZE)
            lock_release(qw_cons_lock_1, 1)
            aie.next_bd(block[31])

        with block[33]:
            s13 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 3, dest=block[34], chain=block[36])
        
        with block[34]:
            lock_acquire(qw_cons_lock_1, 1)
            aie.dma_bd(qw_buffer_0, offset = W_BLOCK_SIZE, len = W_BLOCK_SIZE)
            lock_release(qw_prod_lock, 1)
            aie.next_bd(block[35])
        
        with block[35]:
            lock_acquire(qw_cons_lock_1, 1)
            aie.dma_bd(qw_buffer_1, offset = W_BLOCK_SIZE, len = W_BLOCK_SIZE)
            lock_release(qw_prod_lock, 1)
            aie.next_bd(block[313])
        
        with block[313]:
            lock_acquire(qw_cons_lock_1, 1)
            aie.dma_bd(qw_buffer_2, offset = W_BLOCK_SIZE, len = W_BLOCK_SIZE)
            lock_release(qw_prod_lock, 1)
            aie.next_bd(block[314])
        
        with block[314]:
            lock_acquire(qw_cons_lock_1, 1)
            aie.dma_bd(qw_buffer_3, offset = W_BLOCK_SIZE, len = W_BLOCK_SIZE)
            lock_release(qw_prod_lock, 1)
            aie.next_bd(block[34])
        
        with block[36]:
            aie.EndOp()