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

def mm_cttile(tile: tile_with_location, mctx: mlir_context):

    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    mm = mctx.mm
    x_ty = mctx.x_ty
    w_ty = mctx.w_ty
    y_ty = mctx.y_ty
    y_acc_ty = mctx.y_acc_ty
    b_ty = mctx.b_ty

    m_slice = mctx.m_slice
    k_slice = mctx.k_slice
    n_slice = mctx.n_slice

    x_buffer_0 = aie.buffer(tile = ct, datatype = x_ty, name = f"x_0_{ct_row}_{ct_col}")
    x_buffer_1 = aie.buffer(tile = ct, datatype = x_ty, name = f"x_1_{ct_row}_{ct_col}")
    w_buffer_0 = aie.buffer(tile = ct, datatype = w_ty, name = f"w_0_{ct_row}_{ct_col}")
    w_buffer_1 = aie.buffer(tile = ct, datatype = w_ty, name = f"w_1_{ct_row}_{ct_col}")
    y_buffer_0 = aie.buffer(tile = ct, datatype = y_ty, name = f"y_0_{ct_row}_{ct_col}")
    y_buffer_1 = aie.buffer(tile = ct, datatype = y_ty, name = f"y_1_{ct_row}_{ct_col}")
    y_acc_buffer = aie.buffer(tile = ct, datatype = y_acc_ty, name = f"y_acc_{ct_row}_{ct_col}")
    b_buffer = aie.buffer(tile = ct, datatype = b_ty, name = f"b_{ct_row}_{ct_col}")
    rtp_buffer = mctx.buffers[f"rtp_{ct_row}_{ct_col}"]
    
    x_prod_lock = aie.lock(ct, lock_id = 0, init = 2)
    x_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    w_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    w_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    y_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    y_cons_lock = aie.lock(ct, lock_id = 5, init = 0)
    l_prod_lock = aie.lock(ct, lock_id = 6, init = 0)

    @aie.core(ct, link_with = "mm.o", stack_size = 1024 * 3)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            mm(x_buffer_0, x_buffer_1, w_buffer_0, w_buffer_1, y_buffer_0, y_buffer_1, y_acc_buffer, b_buffer, rtp_buffer)
    
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
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w_buffer_0, offset = 0, len = n_slice * k_slice, dimensions = [(n_slice, 8), (k_slice // 8, n_slice * 8), (8, 1)])
            lock_release(w_cons_lock, 1)
            aie.next_bd(block[5])
        
        with block[5]:
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w_buffer_1, offset = 0, len = n_slice * k_slice, dimensions = [(n_slice, 8), (k_slice // 8, n_slice * 8), (8, 1)])
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

def mm_memtile(tile: tile_with_location, mctx: mlir_context):
    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    x_mem_ty = mctx.x_mem_ty
    w_mem_ty = mctx.w_mem_ty
    y_mem_ty = mctx.y_mem_ty
    m = mctx.m
    k = mctx.k
    n = mctx.n
    m_slice = mctx.m_slice
    k_slice = mctx.k_slice
    n_slice = mctx.n_slice

    x_buffer_0 = aie.buffer(tile = mt, datatype = x_mem_ty, name = f"x_0_{mt_row}_{mt_col}")
    x_buffer_1 = aie.buffer(tile = mt, datatype = x_mem_ty, name = f"x_1_{mt_row}_{mt_col}")
    w_buffer_0 = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_0_{mt_row}_{mt_col}")
    w_buffer_1 = aie.buffer(tile = mt, datatype = w_mem_ty, name = f"w_1_{mt_row}_{mt_col}")
    y_buffer_0 = aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_0_{mt_row}_{mt_col}")
    y_buffer_1 = aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_1_{mt_row}_{mt_col}")

    x_prod_lock = aie.lock(mt, lock_id = 0, init = 2)
    x_cons_lock = aie.lock(mt, lock_id = 1, init = 0)
    w_prod_lock = aie.lock(mt, lock_id = 2, init = 2)
    w_cons_lock = aie.lock(mt, lock_id = 3, init = 0)
    y_prod_lock = aie.lock(mt, lock_id = 4, init = 2)
    y_cons_lock_0 = aie.lock(mt, lock_id = 5, init = 0)
    y_cons_lock_1 = aie.lock(mt, lock_id = 7, init = 0)
    y_cons_lock_2 = aie.lock(mt, lock_id = 9, init = 0)
    y_cons_lock_3 = aie.lock(mt, lock_id = 11, init = 0)

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        # 0 is A
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
            aie.dma_bd(x_buffer_0, offset = 0, len = m * k, dimensions = [(k // k_slice, k_slice), (m, k), (k_slice, 1)])
            lock_release(x_prod_lock, 1)
            aie.next_bd(block[5])

        with block[5]:
            lock_acquire(x_cons_lock, 1)
            aie.dma_bd(x_buffer_1, offset = 0, len = m * k, dimensions = [(k // k_slice, k_slice), (m, k), (k_slice, 1)])
            lock_release(x_prod_lock, 1)
            aie.next_bd(block[4])
        
        with block[6]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[7], chain=block[9])
        
        with block[7]:
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w_buffer_0)
            lock_release(w_cons_lock, 1)
            aie.next_bd(block[8])
        
        with block[8]:
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w_buffer_1)
            lock_release(w_cons_lock, 1)
            aie.next_bd(block[7])
        
        with block[9]:
            s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[10], chain=block[12])
        
        with block[10]:
            lock_acquire(w_cons_lock, 1)
            aie.dma_bd(w_buffer_0, offset = 0, len = n * k, dimensions = [(k // k_slice, k_slice), (n, k), (k_slice, 1)])
            lock_release(w_prod_lock, 1)
            aie.next_bd(block[11])

        with block[11]:
            lock_acquire(w_cons_lock, 1)
            aie.dma_bd(w_buffer_1, offset = 0, len = n * k, dimensions = [(k // k_slice, k_slice), (n, k), (k_slice, 1)])
            lock_release(w_prod_lock, 1)
            aie.next_bd(block[10])
        
        with block[12]:
            s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[13], chain=block[15])
        
        with block[13]:
            lock_acquire(y_prod_lock, 1)
            aie.dma_bd(y_buffer_0, offset = 0, len = m * n, dimensions = [(m // 8, 512), (8, 8), (8, 64), (8, 1)])
            lock_release(y_cons_lock_0, 1)
            aie.next_bd(block[14])
        
        with block[14]:
            lock_acquire(y_prod_lock, 1)
            aie.dma_bd(y_buffer_1, offset = 0, len = m * n, dimensions = [(m // 8, 512), (8, 8), (8, 64), (8, 1)])
            lock_release(y_cons_lock_0, 1)
            aie.next_bd(block[13])
        
        with block[15]:
            s5 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest=block[16], chain=block[18])
        
        with block[16]:
            lock_acquire(y_cons_lock_0, 1)
            aie.dma_bd(y_buffer_0, offset = m * n, len = m * n, dimensions = [(m // 8, 512), (8, 8), (8, 64), (8, 1)])
            lock_release(y_cons_lock_1, 1)
            aie.next_bd(block[17])
        
        with block[17]:
            lock_acquire(y_cons_lock_0, 1)
            aie.dma_bd(y_buffer_1, offset = m * n, len = m * n, dimensions = [(m // 8, 512), (8, 8), (8, 64), (8, 1)])
            lock_release(y_cons_lock_1, 1)
            aie.next_bd(block[16])
        
        with block[18]:
            s6 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 4, dest=block[19], chain=block[21])
        
        with block[19]:
            lock_acquire(y_cons_lock_1, 1)
            aie.dma_bd(y_buffer_0, offset = 2 * m * n, len = m * n, dimensions = [(m // 8, 512), (8, 8), (8, 64), (8, 1)])
            lock_release(y_cons_lock_2, 1)
            aie.next_bd(block[20])
        
        with block[20]:
            lock_acquire(y_cons_lock_1, 1)
            aie.dma_bd(y_buffer_1, offset = 2 * m * n, len = m * n, dimensions = [(m // 8, 512), (8, 8), (8, 64), (8, 1)])
            lock_release(y_cons_lock_2, 1)
            aie.next_bd(block[19])
        
        with block[21]:
            s7 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 5, dest=block[22], chain=block[24])
        
        with block[22]:
            lock_acquire(y_cons_lock_2, 1)
            aie.dma_bd(y_buffer_0, offset = 3 * m * n, len = m * n, dimensions = [(m // 8, 512), (8, 8), (8, 64), (8, 1)])
            lock_release(y_cons_lock_3, 1)
            aie.next_bd(block[23])
        
        with block[23]:
            lock_acquire(y_cons_lock_2, 1)
            aie.dma_bd(y_buffer_1, offset = 3 * m * n, len = m * n, dimensions = [(m // 8, 512), (8, 8), (8, 64), (8, 1)])
            lock_release(y_cons_lock_3, 1)
            aie.next_bd(block[22])
        
        with block[24]:
            s8 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[25], chain=block[27])
        
        with block[25]:
            lock_acquire(y_cons_lock_3, 1)
            aie.dma_bd(y_buffer_0, offset = 0, len = 4 * m * n, dimensions = [(m, n), (4, m * n), (n, 1)])
            lock_release(y_prod_lock, 1)
            aie.next_bd(block[26])
        
        with block[26]:
            lock_acquire(y_cons_lock_3, 1)
            aie.dma_bd(y_buffer_1, offset = 0, len = 4 * m * n, dimensions = [(m, n), (4, m * n), (n, 1)])
            lock_release(y_prod_lock, 1)
            aie.next_bd(block[25])
        
        with block[27]:
            aie.EndOp()