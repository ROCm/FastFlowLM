# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

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

def dequant_mm_tile(tile_e: tile_with_location, tile_o: tile_with_location, mctx: mlir_context):

    ct_e = tile_e.tile
    ct_e_row = tile_e.row
    ct_e_col = tile_e.col
    ct_o = tile_o.tile
    ct_o_row = tile_o.row
    ct_o_col = tile_o.col


    x_ty = mctx.x_ty
    qw_ty = mctx.qw_ty
    w_ty = mctx.w_ty
    y_ty = mctx.y_ty
    y_acc_ty = mctx.y_acc_ty

    dequant_mm_e = mctx.dequant_mm_e
    dequant_mm_o = mctx.dequant_mm_o

    m_slice = mctx.m_slice
    k_slice = mctx.k_slice

    x_buffer_e_0 = aie.buffer(tile = ct_e, datatype = x_ty, name = f"x_0_{ct_e_row}_{ct_e_col}")
    x_buffer_e_1 = aie.buffer(tile = ct_e, datatype = x_ty, name = f"x_1_{ct_e_row}_{ct_e_col}")
    qw_buffer_0 = aie.buffer(tile = ct_e, datatype = qw_ty, name = f"w_0_{ct_e_row}_{ct_e_col}")
    qw_buffer_1 = aie.buffer(tile = ct_e, datatype = qw_ty, name = f"w_1_{ct_e_row}_{ct_e_col}")
    y_buffer_e_0 = aie.buffer(tile = ct_e, datatype = y_ty, name = f"y_0_{ct_e_row}_{ct_e_col}")
    y_buffer_e_1 = aie.buffer(tile = ct_e, datatype = y_ty, name = f"y_1_{ct_e_row}_{ct_e_col}")
    y_acc_buffer_e = aie.buffer(tile = ct_e, datatype = y_acc_ty, name = f"y_acc_{ct_e_row}_{ct_e_col}")
    K_div_k_buffer_e = mctx.buffers[f"K_div_k_{ct_e_row}_{ct_e_col}"]
    rtp_activate_e = mctx.buffers[f"rtp_act_{ct_e_row}_{ct_e_col}"]

    w_buffer = aie.buffer(tile = ct_o, datatype = w_ty, name = f"w_{ct_o_row}_{ct_o_col}")

    x_buffer_o_0 = aie.buffer(tile = ct_o, datatype = x_ty, name = f"x_0_{ct_o_row}_{ct_o_col}")
    x_buffer_o_1 = aie.buffer(tile = ct_o, datatype = x_ty, name = f"x_1_{ct_o_row}_{ct_o_col}")
    y_buffer_o_0 = aie.buffer(tile = ct_o, datatype = y_ty, name = f"y_0_{ct_o_row}_{ct_o_col}")
    y_buffer_o_1 = aie.buffer(tile = ct_o, datatype = y_ty, name = f"y_1_{ct_o_row}_{ct_o_col}")
    y_acc_buffer_o = aie.buffer(tile = ct_o, datatype = y_acc_ty, name = f"y_acc_{ct_o_row}_{ct_o_col}")
    K_div_k_buffer_o = mctx.buffers[f"K_div_k_{ct_o_row}_{ct_o_col}"]
    rtp_activate_o = mctx.buffers[f"rtp_act_{ct_o_row}_{ct_o_col}"]
    
    x_prod_lock = aie.lock(ct_e, lock_id = 0, init = 2)
    x_cons_lock = aie.lock(ct_e, lock_id = 1, init = 0)
    qw_prod_lock = aie.lock(ct_e, lock_id = 2, init = 4)
    qw_cons_lock = aie.lock(ct_e, lock_id = 3, init = 0)
    y_prod_lock = aie.lock(ct_e, lock_id = 4, init = 2)
    y_cons_lock = aie.lock(ct_e, lock_id = 5, init = 0)
    sync_lock = aie.lock(ct_e, lock_id = 6, init = 0)
    x_prod_lock_o = aie.lock(ct_o, lock_id = 0, init = 2)
    x_cons_lock_o = aie.lock(ct_o, lock_id = 1, init = 0)
    y_prod_lock_o = aie.lock(ct_o, lock_id = 4, init = 2)
    y_cons_lock_o = aie.lock(ct_o, lock_id = 5, init = 0)
    # Second half of the w_buffer rendezvous.  One lock only gives a one-way
    # signal; the shared buffer needs a real barrier in both directions, both
    # after the two dequant halves and after both cores have read the tile.
    sync_lock_o = aie.lock(ct_o, lock_id = 6, init = 0)
    
    @aie.core(ct_e, link_with = "dequant_mm_e.o", stack_size = 1024 * 2)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            dequant_mm_e(y_buffer_e_0, y_buffer_e_1, x_buffer_e_0, x_buffer_e_1, qw_buffer_0, qw_buffer_1, w_buffer, y_acc_buffer_e, K_div_k_buffer_e, rtp_activate_e)
    
    @aie.mem(ct_e)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

        with block[1]:
            lock_acquire(x_prod_lock, 1)
            aie.dma_bd(x_buffer_e_0)
            lock_release(x_cons_lock, 1)
            aie.next_bd(block[2])
        
        with block[2]:
            lock_acquire(x_prod_lock, 1)
            aie.dma_bd(x_buffer_e_1)
            lock_release(x_cons_lock, 1)
            aie.next_bd(block[1])
        
        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])

        with block[4]:
            lock_acquire(qw_prod_lock, 2)
            aie.dma_bd(qw_buffer_0)
            lock_release(qw_cons_lock, 2)
            aie.next_bd(block[5])
            
        with block[5]:
            lock_acquire(qw_prod_lock, 2)
            aie.dma_bd(qw_buffer_1)
            lock_release(qw_cons_lock, 2)
            aie.next_bd(block[4])
            
        with block[6]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[7], chain=block[9])

        with block[7]:
            lock_acquire(y_cons_lock, 1)
            aie.dma_bd(y_buffer_e_0)
            lock_release(y_prod_lock, 1)
            aie.next_bd(block[8])

        with block[8]:
            lock_acquire(y_cons_lock, 1)
            aie.dma_bd(y_buffer_e_1)
            lock_release(y_prod_lock, 1)
            aie.next_bd(block[7])

        with block[9]:
            aie.EndOp()

    @aie.core(ct_o, link_with = "dequant_mm_o.o", stack_size = 1024 * 2)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            dequant_mm_o(y_buffer_o_0, y_buffer_o_1, x_buffer_o_0, x_buffer_o_1, qw_buffer_0, qw_buffer_1, w_buffer, y_acc_buffer_o, K_div_k_buffer_o, rtp_activate_o)
    
    @aie.mem(ct_o)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

        with block[1]:
            lock_acquire(x_prod_lock_o, 1)
            aie.dma_bd(x_buffer_o_0)
            lock_release(x_cons_lock_o, 1)
            aie.next_bd(block[2])
        
        with block[2]:
            lock_acquire(x_prod_lock_o, 1)
            aie.dma_bd(x_buffer_o_1)
            lock_release(x_cons_lock_o, 1)
            aie.next_bd(block[1])
        
        with block[3]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

        with block[4]:
            lock_acquire(y_cons_lock_o, 1)
            aie.dma_bd(y_buffer_o_0)
            lock_release(y_prod_lock_o, 1)
            aie.next_bd(block[5])

        with block[5]:
            lock_acquire(y_cons_lock_o, 1)
            aie.dma_bd(y_buffer_o_1)
            lock_release(y_prod_lock_o, 1)
            aie.next_bd(block[4])

        with block[6]:
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
    qw_size = mctx.qw_size
    # one 256-k half of the mem tile x buffer: 8 row groups x 64 bfp16 tiles (in bf16 elements)
    x_half_len = (m // 16) * 16 * 256 * 9 // 8 // 2

    x_buffer_0 = aie.buffer(tile = mt, datatype = x_mem_ty, name = f"x_0_{mt_row}_{mt_col}")
    x_buffer_1 = aie.buffer(tile = mt, datatype = x_mem_ty, name = f"x_1_{mt_row}_{mt_col}")
    y_buffer_0 = aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_0_{mt_row}_{mt_col}")
    y_buffer_1 = aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_1_{mt_row}_{mt_col}")
    qw_buffer_0 = aie.buffer(tile = mt, datatype = qw_mem_ty, name = f"qw_0_{mt_row}_{mt_col}")
    qw_buffer_1 = aie.buffer(tile = mt, datatype = qw_mem_ty, name = f"qw_1_{mt_row}_{mt_col}")

    x_prod_lock = aie.lock(mt, lock_id = 0, init = 2)
    x_cons_lock_0 = aie.lock(mt, lock_id = 1, init = 0)
    x_cons_lock_1 = aie.lock(mt, lock_id = 2, init = 0)
    
    # one free / full lock per (core, y buffer), so the 4 cores write their tiles independently
    y_free = [[aie.lock(mt, lock_id = 12 + 2 * c + b, init = 1) for b in range(2)] for c in range(4)]
    y_full = [[aie.lock(mt, lock_id = 20 + 2 * c + b, init = 0) for b in range(2)] for c in range(4)]

    qw_prod_lock = aie.lock(mt, lock_id = 10, init = 2)
    qw_cons_lock = aie.lock(mt, lock_id = 11, init = 0)

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
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[8])

        with block[4]:
            lock_acquire(x_cons_lock_0, 1)
            aie.dma_bd(x_buffer_0, offset = 0, len = x_half_len)
            lock_release(x_cons_lock_1, 1)
            aie.next_bd(block[5])

        with block[5]:
            lock_acquire(x_cons_lock_1, 1)
            aie.dma_bd(x_buffer_0, offset = x_half_len, len = x_half_len)
            lock_release(x_prod_lock, 1)
            aie.next_bd(block[6])

        with block[6]:
            lock_acquire(x_cons_lock_0, 1)
            aie.dma_bd(x_buffer_1, offset = 0, len = x_half_len)
            lock_release(x_cons_lock_1, 1)
            aie.next_bd(block[7])

        with block[7]:
            lock_acquire(x_cons_lock_1, 1)
            aie.dma_bd(x_buffer_1, offset = x_half_len, len = x_half_len)
            lock_release(x_prod_lock, 1)
            aie.next_bd(block[4])

        with block[8]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[9], chain=block[11])

        with block[9]:
            lock_acquire(y_free[0][0], 1)
            aie.dma_bd(y_buffer_0, offset = 0 * m * n, len = m * n, dimensions = [(m // 8, 8 * n), (n // 8, 8), (8, n), (8, 1)])
            lock_release(y_full[0][0], 1)
            aie.next_bd(block[10])

        with block[10]:
            lock_acquire(y_free[0][1], 1)
            aie.dma_bd(y_buffer_1, offset = 0 * m * n, len = m * n, dimensions = [(m // 8, 8 * n), (n // 8, 8), (8, n), (8, 1)])
            lock_release(y_full[0][1], 1)
            aie.next_bd(block[9])

        with block[11]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[12], chain=block[14])

        with block[12]:
            lock_acquire(y_free[1][0], 1)
            aie.dma_bd(y_buffer_0, offset = 1 * m * n, len = m * n, dimensions = [(m // 8, 8 * n), (n // 8, 8), (8, n), (8, 1)])
            lock_release(y_full[1][0], 1)
            aie.next_bd(block[13])

        with block[13]:
            lock_acquire(y_free[1][1], 1)
            aie.dma_bd(y_buffer_1, offset = 1 * m * n, len = m * n, dimensions = [(m // 8, 8 * n), (n // 8, 8), (8, n), (8, 1)])
            lock_release(y_full[1][1], 1)
            aie.next_bd(block[12])

        with block[14]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest=block[15], chain=block[17])

        with block[15]:
            lock_acquire(y_free[2][0], 1)
            aie.dma_bd(y_buffer_0, offset = 2 * m * n, len = m * n, dimensions = [(m // 8, 8 * n), (n // 8, 8), (8, n), (8, 1)])
            lock_release(y_full[2][0], 1)
            aie.next_bd(block[16])

        with block[16]:
            lock_acquire(y_free[2][1], 1)
            aie.dma_bd(y_buffer_1, offset = 2 * m * n, len = m * n, dimensions = [(m // 8, 8 * n), (n // 8, 8), (8, n), (8, 1)])
            lock_release(y_full[2][1], 1)
            aie.next_bd(block[15])

        with block[17]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 4, dest=block[18], chain=block[20])

        with block[18]:
            lock_acquire(y_free[3][0], 1)
            aie.dma_bd(y_buffer_0, offset = 3 * m * n, len = m * n, dimensions = [(m // 8, 8 * n), (n // 8, 8), (8, n), (8, 1)])
            lock_release(y_full[3][0], 1)
            aie.next_bd(block[19])

        with block[19]:
            lock_acquire(y_free[3][1], 1)
            aie.dma_bd(y_buffer_1, offset = 3 * m * n, len = m * n, dimensions = [(m // 8, 8 * n), (n // 8, 8), (8, n), (8, 1)])
            lock_release(y_full[3][1], 1)
            aie.next_bd(block[18])

        with block[20]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[21], chain=block[29])

        with block[21]:
            lock_acquire(y_full[0][0], 1)
            aie.dma_bd(y_buffer_0, offset = 0 * m * n, len = m * n)
            lock_release(y_free[0][0], 1)
            aie.next_bd(block[22])

        with block[22]:
            lock_acquire(y_full[1][0], 1)
            aie.dma_bd(y_buffer_0, offset = 1 * m * n, len = m * n)
            lock_release(y_free[1][0], 1)
            aie.next_bd(block[23])

        with block[23]:
            lock_acquire(y_full[2][0], 1)
            aie.dma_bd(y_buffer_0, offset = 2 * m * n, len = m * n)
            lock_release(y_free[2][0], 1)
            aie.next_bd(block[24])

        with block[24]:
            lock_acquire(y_full[3][0], 1)
            aie.dma_bd(y_buffer_0, offset = 3 * m * n, len = m * n)
            lock_release(y_free[3][0], 1)
            aie.next_bd(block[25])

        with block[25]:
            lock_acquire(y_full[0][1], 1)
            aie.dma_bd(y_buffer_1, offset = 0 * m * n, len = m * n)
            lock_release(y_free[0][1], 1)
            aie.next_bd(block[26])

        with block[26]:
            lock_acquire(y_full[1][1], 1)
            aie.dma_bd(y_buffer_1, offset = 1 * m * n, len = m * n)
            lock_release(y_free[1][1], 1)
            aie.next_bd(block[27])

        with block[27]:
            lock_acquire(y_full[2][1], 1)
            aie.dma_bd(y_buffer_1, offset = 2 * m * n, len = m * n)
            lock_release(y_free[2][1], 1)
            aie.next_bd(block[28])

        with block[28]:
            lock_acquire(y_full[3][1], 1)
            aie.dma_bd(y_buffer_1, offset = 3 * m * n, len = m * n)
            lock_release(y_free[3][1], 1)
            aie.next_bd(block[21])

        with block[29]:
            s9 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 5, dest=block[30], chain=block[32])
        
        with block[30]:
            lock_acquire(qw_prod_lock, 1)
            aie.dma_bd(qw_buffer_0)
            lock_release(qw_cons_lock, 1)
            aie.next_bd(block[31])
        
        with block[31]:
            lock_acquire(qw_prod_lock, 1)
            aie.dma_bd(qw_buffer_1)
            lock_release(qw_cons_lock, 1)
            aie.next_bd(block[30])

        with block[32]:
            s10 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[33], chain=block[35])
        
        with block[33]:
            lock_acquire(qw_cons_lock, 1)
            aie.dma_bd(qw_buffer_0)
            lock_release(qw_prod_lock, 1)
            aie.next_bd(block[34])
        
        with block[34]:
            lock_acquire(qw_cons_lock, 1)
            aie.dma_bd(qw_buffer_1)
            lock_release(qw_prod_lock, 1)
            aie.next_bd(block[33])

        with block[35]:
            aie.EndOp()