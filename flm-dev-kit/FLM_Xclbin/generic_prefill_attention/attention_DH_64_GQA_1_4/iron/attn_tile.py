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

def attn_cttile(tile: tile_with_location, mctx: mlir_context):

    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    attn = mctx.attn
    o_ty = mctx.o_ty
    q_ty = mctx.q_ty
    in_ty = mctx.in_ty
    s_ty = mctx.s_ty
    m_ty = mctx.m_ty
    y_ty = mctx.y_ty
    L_begin_buffer = mctx.buffers[f"L_begin_attn_core_{ct_row}_{ct_col}"]
    L_end_buffer = mctx.buffers[f"L_end_attn_core_{ct_row}_{ct_col}"]
    window_size_buffer = mctx.buffers[f"window_size_attn_core_{ct_row}_{ct_col}"]

    q_buffer = aie.buffer(tile = ct, datatype = q_ty, name = f"q_{ct_row}_{ct_col}")
    in_buffer_0 = aie.buffer(tile = ct, datatype = in_ty, name = f"in_0_{ct_row}_{ct_col}")
    in_buffer_1 = aie.buffer(tile = ct, datatype = in_ty, name = f"in_1_{ct_row}_{ct_col}")
    o_buffer_0 = aie.buffer(tile = ct, datatype = o_ty, name = f"o_0_{ct_row}_{ct_col}")
    o_buffer_1 = aie.buffer(tile = ct, datatype = o_ty, name = f"o_1_{ct_row}_{ct_col}")
    s_buffer = aie.buffer(tile = ct, datatype = s_ty, name = f"s_{ct_row}_{ct_col}")
    y_buffer = aie.buffer(tile = ct, datatype = y_ty, name = f"y_{ct_row}_{ct_col}")
    m_buffer = aie.buffer(tile = ct, datatype = m_ty, name = f"m_{ct_row}_{ct_col}")
    
    q_prod_lock = aie.lock(ct, lock_id = 0, init = 1)
    q_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    in_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    in_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    o_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    o_cons_lock = aie.lock(ct, lock_id = 5, init = 0)

    @aie.core(ct, link_with = "attn.o", stack_size = 1024*8)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            attn(o_buffer_0, o_buffer_1, q_buffer, in_buffer_0, in_buffer_1, s_buffer, m_buffer, y_buffer, L_begin_buffer, L_end_buffer, window_size_buffer, ct_row - 2, ct_col & 0x1)
    
    @aie.mem(ct)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[2])

        with block[1]:
            lock_acquire(q_prod_lock, 1)
            aie.dma_bd(q_buffer)
            lock_release(q_cons_lock, 1)
            aie.next_bd(block[1]) 

        with block[2]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[3], chain=block[5])
        
        with block[3]:
            lock_acquire(in_prod_lock, 1)
            aie.dma_bd(in_buffer_0)
            lock_release(in_cons_lock, 1)
            aie.next_bd(block[4])
        
        with block[4]:
            lock_acquire(in_prod_lock, 1)
            aie.dma_bd(in_buffer_1)
            lock_release(in_cons_lock, 1)
            aie.next_bd(block[3])
        
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

def attn_memtile_even(tile: tile_with_location, mctx: mlir_context):
    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    q_mem_ty = mctx.q_mem_ty
    o_mem_ty = mctx.o_mem_ty

    lq_mt = mctx.lq_mt
    lq_ct = mctx.lq_ct
    lq = mctx.lq
    d = mctx.d

    q_buffer_0 = aie.buffer(tile = mt, datatype = q_mem_ty, name = f"q_0_{mt_row}_{mt_col}")
    q_buffer_1 = aie.buffer(tile = mt, datatype = q_mem_ty, name = f"q_1_{mt_row}_{mt_col}")
    o_buffer_0 = aie.buffer(tile = mt, datatype = o_mem_ty, name = f"o_0_{mt_row}_{mt_col}")
    o_buffer_1 = aie.buffer(tile = mt, datatype = o_mem_ty, name = f"o_1_{mt_row}_{mt_col}")

    q_prod_lock_0 = aie.lock(mt, lock_id = 0, init = 2)
    q_cons_lock_0 = aie.lock(mt, lock_id = 1, init = 0)
    q_cons_lock_1 = aie.lock(mt, lock_id = 3, init = 0)
    q_prod_lock_1 = aie.lock(mt, lock_id = 4, init = 2)
    q_cons_lock_2 = aie.lock(mt, lock_id = 5, init = 0)
    q_cons_lock_3 = aie.lock(mt, lock_id = 6, init = 0)
    o_prod_lock_0 = aie.lock(mt, lock_id = 7, init = 2)
    o_prod_lock_1 = aie.lock(mt, lock_id = 8, init = 0)
    o_prod_lock_2 = aie.lock(mt, lock_id = 9, init = 0)
    o_prod_lock_3 = aie.lock(mt, lock_id = 10, init = 0)
    o_cons_lock = aie.lock(mt, lock_id = 11, init = 0)

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])
        
        with block[1]:
            lock_acquire(q_prod_lock_0, 1)
            aie.dma_bd(q_buffer_0, offset = 0, len = lq_mt * d)
            lock_release(q_cons_lock_0, 1)
            aie.next_bd(block[2])
        
        with block[2]:
            lock_acquire(q_prod_lock_0, 1)
            aie.dma_bd(q_buffer_1, offset = 0, len = lq_mt * d)
            lock_release(q_cons_lock_0, 1)
            aie.next_bd(block[1])
        
        with block[3]:
            s6 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])
        
        with block[4]:
            lock_acquire(q_prod_lock_1, 1)
            aie.dma_bd(q_buffer_0, offset = lq_mt * d, len = lq_mt * d)
            lock_release(q_cons_lock_2, 1)
            aie.next_bd(block[5])
        
        with block[5]:
            lock_acquire(q_prod_lock_1, 1)
            aie.dma_bd(q_buffer_1, offset = lq_mt * d, len = lq_mt * d)
            lock_release(q_cons_lock_2, 1)
            aie.next_bd(block[4])
        
        with block[6]:
            s7 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[7], chain=block[9])
        
        with block[7]:
            lock_acquire(q_cons_lock_0, 1)
            aie.dma_bd(q_buffer_0, offset = 0, len = lq_ct * d, dimensions = [(lq_ct // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(q_cons_lock_1, 1)
            aie.next_bd(block[8])
            
        with block[8]:
            lock_acquire(q_cons_lock_0, 1)
            aie.dma_bd(q_buffer_1, offset = 0, len = lq_ct * d, dimensions = [(lq_ct // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(q_cons_lock_1, 1)
            aie.next_bd(block[7])
        
        with block[9]:
            s8 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[10], chain=block[12])
        
        with block[10]:
            lock_acquire(q_cons_lock_1, 1)
            aie.dma_bd(q_buffer_0, offset = lq_ct * d, len = lq_ct * d, dimensions = [(lq_ct // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(q_prod_lock_0, 1)
            aie.next_bd(block[11])

        with block[11]:
            lock_acquire(q_cons_lock_1, 1)
            aie.dma_bd(q_buffer_1, offset = lq_ct * d, len = lq_ct * d, dimensions = [(lq_ct // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(q_prod_lock_0, 1)
            aie.next_bd(block[10])
        
        with block[12]:
            s9 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[13], chain=block[15])
        
        with block[13]:
            lock_acquire(q_cons_lock_2, 1)
            aie.dma_bd(q_buffer_0, offset = 2 * lq_ct * d, len = lq_ct * d, dimensions = [(lq_ct // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(q_cons_lock_3, 1)
            aie.next_bd(block[14])
        
        with block[14]:
            lock_acquire(q_cons_lock_2, 1)
            aie.dma_bd(q_buffer_1, offset = 2 * lq_ct * d, len = lq_ct * d, dimensions = [(lq_ct // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(q_cons_lock_3, 1)
            aie.next_bd(block[13])
        
        with block[15]:
            s10 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 3, dest=block[16], chain=block[18])

        with block[16]:
            lock_acquire(q_cons_lock_3, 1)
            aie.dma_bd(q_buffer_0, offset = 3 * lq_ct * d, len = lq_ct * d, dimensions = [(lq_ct // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(q_prod_lock_1, 1)
            aie.next_bd(block[17])
        
        with block[17]:
            lock_acquire(q_cons_lock_3, 1)
            aie.dma_bd(q_buffer_1, offset = 3 * lq_ct * d, len = lq_ct * d, dimensions = [(lq_ct // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(q_prod_lock_1, 1)
            aie.next_bd(block[16])
    
        with block[18]:
            s9 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[19], chain=block[21])
        
        with block[19]:
            lock_acquire(o_prod_lock_0, 1)
            aie.dma_bd(o_buffer_0, offset = 0, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_prod_lock_1, 1)
            aie.next_bd(block[20])

        with block[20]:
            lock_acquire(o_prod_lock_0, 1)
            aie.dma_bd(o_buffer_1, offset = 0, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_prod_lock_1, 1)
            aie.next_bd(block[19])
        
        with block[21]:
            s10 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest=block[22], chain=block[24])
        
        with block[22]:
            lock_acquire(o_prod_lock_1, 1)
            aie.dma_bd(o_buffer_0, offset = lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_prod_lock_2, 1)
            aie.next_bd(block[23])
        
        with block[23]:
            lock_acquire(o_prod_lock_1, 1)
            aie.dma_bd(o_buffer_1, offset = lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_prod_lock_2, 1)
            aie.next_bd(block[22])
        
        with block[24]:
            s11 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 4, dest=block[25], chain=block[27])
        
        with block[25]:
            lock_acquire(o_prod_lock_2, 1)
            aie.dma_bd(o_buffer_0, offset = 2 * lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_prod_lock_3, 1)
            aie.next_bd(block[26])
        
        with block[26]:
            lock_acquire(o_prod_lock_2, 1)
            aie.dma_bd(o_buffer_1, offset = 2 * lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_prod_lock_3, 1)
            aie.next_bd(block[25])
        
        with block[27]:
            s12 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 5, dest=block[28], chain=block[30])

        with block[28]:
            lock_acquire(o_prod_lock_3, 1)
            aie.dma_bd(o_buffer_0, offset = 3 * lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_cons_lock, 1)
            aie.next_bd(block[29])
        
        with block[29]:
            lock_acquire(o_prod_lock_3, 1)
            aie.dma_bd(o_buffer_1, offset = 3 * lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_cons_lock, 1)
            aie.next_bd(block[28])

        with block[30]:
            s12 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 5, dest=block[31], chain=block[33])
        
        with block[31]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_0)
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[32])
        
        with block[32]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_1)
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[31])
        
        with block[33]:
            aie.EndOp()

def attn_memtile_odd(tile: tile_with_location, mctx: mlir_context):
    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    in_mem_ty = mctx.in_mem_ty
    o_mem_ty = mctx.o_mem_ty
    lk_mt = mctx.lk_mt
    lk = mctx.lk
    lq = mctx.lq
    d = mctx.d

    in_buffer_0 = aie.buffer(tile = mt, datatype = in_mem_ty, name = f"in_0_{mt_row}_{mt_col}")
    in_buffer_1 = aie.buffer(tile = mt, datatype = in_mem_ty, name = f"in_1_{mt_row}_{mt_col}")
    o_buffer_0 = aie.buffer(tile = mt, datatype = o_mem_ty, name = f"o_0_{mt_row}_{mt_col}")
    o_buffer_1 = aie.buffer(tile = mt, datatype = o_mem_ty, name = f"o_1_{mt_row}_{mt_col}")

    in_prod_lock = aie.lock(mt, lock_id = 0, init = 4)
    in_cons_lock = aie.lock(mt, lock_id = 1, init = 0)

    o_prod_lock_0 = aie.lock(mt, lock_id = 2, init = 2)
    o_prod_lock_1 = aie.lock(mt, lock_id = 3, init = 0)
    o_prod_lock_2 = aie.lock(mt, lock_id = 4, init = 0)
    o_prod_lock_3 = aie.lock(mt, lock_id = 5, init = 0)
    o_cons_lock = aie.lock(mt, lock_id = 6, init = 0)

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])
        
        with block[1]:
            lock_acquire(in_prod_lock, 1)
            aie.dma_bd(in_buffer_0, offset = 0, len = lk_mt * d, dimensions = [(lk_mt // lk, lk * d), (lk, 8), (8, 256), (8, 1)])
            lock_release(in_cons_lock, 1)
            aie.next_bd(block[2])
        
        with block[2]:
            lock_acquire(in_prod_lock, 1)
            aie.dma_bd(in_buffer_1, offset = 0, len = lk_mt * d, dimensions = [(lk_mt // lk, lk * d), (lk, 8), (8, 256), (8, 1)])
            lock_release(in_cons_lock, 1)
            aie.next_bd(block[1])
        
        with block[3]:
            s6 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])
        
        with block[4]:
            lock_acquire(in_prod_lock, 1)
            aie.dma_bd(in_buffer_0, offset = lk_mt * d, len = lk_mt * d, dimensions = [(lk_mt // lk, lk * d), (lk, 8), (8, 256), (8, 1)])
            lock_release(in_cons_lock, 1)
            aie.next_bd(block[5])
        
        with block[5]:
            lock_acquire(in_prod_lock, 1)
            aie.dma_bd(in_buffer_1, offset = lk_mt * d, len = lk_mt * d, dimensions = [(lk_mt // lk, lk * d), (lk, 8), (8, 256), (8, 1)])
            lock_release(in_cons_lock, 1)
            aie.next_bd(block[4])
        
        with block[6]:
            s7 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[7], chain=block[9])
        
        with block[7]:
            lock_acquire(in_cons_lock, 2)
            aie.dma_bd(in_buffer_0)
            lock_release(in_prod_lock, 2)
            aie.next_bd(block[8])
            
        with block[8]:
            lock_acquire(in_cons_lock, 2)
            aie.dma_bd(in_buffer_1)
            lock_release(in_prod_lock, 2)
            aie.next_bd(block[7])
        
        with block[9]:
            s8 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[10], chain=block[12])
        
        with block[10]:
            lock_acquire(o_prod_lock_0, 1)
            aie.dma_bd(o_buffer_0, offset = 0, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_prod_lock_1, 1)
            aie.next_bd(block[11])
        
        with block[11]:
            lock_acquire(o_prod_lock_0, 1)
            aie.dma_bd(o_buffer_1, offset = 0, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_prod_lock_1, 1)
            aie.next_bd(block[10])
        
        with block[12]:
            s9 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest=block[13], chain=block[15])
        
        with block[13]:
            lock_acquire(o_prod_lock_1, 1)
            aie.dma_bd(o_buffer_0, offset = lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_prod_lock_2, 1)
            aie.next_bd(block[14])

        with block[14]:
            lock_acquire(o_prod_lock_1, 1)
            aie.dma_bd(o_buffer_1, offset = lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_prod_lock_2, 1)
            aie.next_bd(block[13])
        
        with block[15]:
            s10 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 4, dest=block[16], chain=block[18])
        
        with block[16]:
            lock_acquire(o_prod_lock_2, 1)
            aie.dma_bd(o_buffer_0, offset = 2 * lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_prod_lock_3, 1)
            aie.next_bd(block[17])
        
        with block[17]:
            lock_acquire(o_prod_lock_2, 1)
            aie.dma_bd(o_buffer_1, offset = 2 * lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_prod_lock_3, 1)
            aie.next_bd(block[16])
        
        with block[18]:
            s11 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 5, dest=block[19], chain=block[21])
        
        with block[19]:
            lock_acquire(o_prod_lock_3, 1)
            aie.dma_bd(o_buffer_0, offset = 3 * lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_cons_lock, 1)
            aie.next_bd(block[20])
        
        with block[20]:
            lock_acquire(o_prod_lock_3, 1)
            aie.dma_bd(o_buffer_1, offset = 3 * lq * d, len = lq * d, dimensions = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)])
            lock_release(o_cons_lock, 1)
            aie.next_bd(block[19])
        
        with block[21]:
            s12 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 5, dest=block[22], chain=block[24])
        
        with block[22]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_0)
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[23])
        
        with block[23]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_1)
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[22])
        
        with block[24]:
            aie.EndOp()