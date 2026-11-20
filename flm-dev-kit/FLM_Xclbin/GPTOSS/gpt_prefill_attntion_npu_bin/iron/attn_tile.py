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
    LQ = mctx.LQ
    LK = mctx.LK
    d = mctx.d
    o_ty = mctx.o_ty
    q_ty = mctx.q_ty
    q_slice_ty = mctx.q_slice_ty
    v_ty = mctx.v_ty
    s_ty = mctx.s_ty
    y_ty = mctx.y_ty
    k_ty = mctx.k_ty
    L_ty = mctx.L_ty
    L_begin_buffer = mctx.buffers[f"L_begin_attn_core_{ct_row}_{ct_col}"]
    L_end_buffer = mctx.buffers[f"L_end_attn_core_{ct_row}_{ct_col}"]
    sink_buffer = mctx.buffers[f"sink_attn_core_{ct_row}_{ct_col}"]
    window_size_buffer = mctx.buffers[f"window_size_attn_core_{ct_row}_{ct_col}"]

    q_buffer = aie.buffer(tile = ct, datatype = q_ty, name = f"q_{ct_row}_{ct_col}")
    q_buffer_0 = aie.buffer(tile = ct, datatype = q_slice_ty, name = f"q_0_{ct_row}_{ct_col}")
    q_buffer_1 = aie.buffer(tile = ct, datatype = q_slice_ty, name = f"q_1_{ct_row}_{ct_col}")
    in_buffer_0 = aie.buffer(tile = ct, datatype = v_ty, name = f"in_0_{ct_row}_{ct_col}")
    in_buffer_1 = aie.buffer(tile = ct, datatype = v_ty, name = f"in_1_{ct_row}_{ct_col}")
    o_buffer_0 = aie.buffer(tile = ct, datatype = o_ty, name = f"o_0_{ct_row}_{ct_col}")
    o_buffer_1 = aie.buffer(tile = ct, datatype = o_ty, name = f"o_1_{ct_row}_{ct_col}")
    s_buffer = aie.buffer(tile = ct, datatype = s_ty, name = f"s_{ct_row}_{ct_col}")
    y_buffer = aie.buffer(tile = ct, datatype = y_ty, name = f"y_{ct_row}_{ct_col}")
    
    q_prod_lock = aie.lock(ct, lock_id = 0, init = 2)
    q_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    in_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    in_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    o_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    o_cons_lock = aie.lock(ct, lock_id = 5, init = 0)

    @aie.core(ct, link_with = "attn.o", stack_size = 1024*8)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            attn(o_buffer_0, o_buffer_1, q_buffer_0, q_buffer_1, in_buffer_0, in_buffer_1, q_buffer, s_buffer, y_buffer, L_begin_buffer, L_end_buffer, sink_buffer, window_size_buffer, ct_row - 2)
    
    @aie.mem(ct)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

        with block[1]:
            lock_acquire(q_prod_lock, 1)
            aie.dma_bd(q_buffer_0)
            lock_release(q_cons_lock, 1)
            aie.next_bd(block[2]) 
        
        with block[2]:
            lock_acquire(q_prod_lock, 1)
            aie.dma_bd(q_buffer_1)
            lock_release(q_cons_lock, 1)
            aie.next_bd(block[1])
        
        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])
        
        with block[4]:
            lock_acquire(in_prod_lock, 1)
            aie.dma_bd(in_buffer_0)
            lock_release(in_cons_lock, 1)
            aie.next_bd(block[5])
        
        with block[5]:
            lock_acquire(in_prod_lock, 1)
            aie.dma_bd(in_buffer_1)
            lock_release(in_cons_lock, 1)
            aie.next_bd(block[4])
        
        with block[6]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[7], chain=block[9])
        
        with block[7]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_0)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[8])
        
        with block[8]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_1)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[7])
        
        with block[9]:
            aie.EndOp()
        
        

def attn_memtile(tile: tile_with_location, mctx: mlir_context, mt_id: int, total_cols: int):

    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    k_mem_ty = mctx.k_mem_ty
    v_mem_ty = mctx.v_mem_ty
    o_mem_ty = mctx.o_mem_ty
    LQ = mctx.LQ
    LK = mctx.LK
    d = mctx.d
    HEAD_PER_COL = mctx.HEAD_PER_COL

   
    k_buffer_0 = aie.buffer(tile = mt, datatype = k_mem_ty, name = f"k_0_{mt_row}_{mt_col}")
    k_buffer_1 = aie.buffer(tile = mt, datatype = k_mem_ty, name = f"k_1_{mt_row}_{mt_col}")
    v_buffer_0 = aie.buffer(tile = mt, datatype = v_mem_ty, name = f"v_0_{mt_row}_{mt_col}")
    v_buffer_1 = aie.buffer(tile = mt, datatype = v_mem_ty, name = f"v_1_{mt_row}_{mt_col}")
    o_buffer_0 = aie.buffer(tile = mt, datatype = o_mem_ty, name = f"o_0_{mt_row}_{mt_col}")
    o_buffer_1 = aie.buffer(tile = mt, datatype = o_mem_ty, name = f"o_1_{mt_row}_{mt_col}")

    k_prod_lock = aie.lock(mt, lock_id = 5, init = 2)
    k_cons_lock = aie.lock(mt, lock_id = 6, init = 0)
    v_prod_lock = aie.lock(mt, lock_id = 7, init = 2)
    v_cons_lock = aie.lock(mt, lock_id = 8, init = 0)
    o_prod_lock_0 = aie.lock(mt, lock_id = 9, init = 2)
    o_prod_lock_1 = aie.lock(mt, lock_id = 10, init = 0)
    o_prod_lock_2 = aie.lock(mt, lock_id = 11, init = 0)
    o_prod_lock_3 = aie.lock(mt, lock_id = 12, init = 0)
    o_cons_lock = aie.lock(mt, lock_id = 13, init = 0)

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        # 0 is A
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])
        
        with block[1]:
            lock_acquire(k_prod_lock, 1)
            aie.dma_bd(k_buffer_0)
            lock_release(k_cons_lock, 1)
            aie.next_bd(block[2])
        
        with block[2]:
            lock_acquire(k_prod_lock, 1)
            aie.dma_bd(k_buffer_1)
            lock_release(k_cons_lock, 1)
            aie.next_bd(block[1])
        
        with block[3]:
            s6 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])
        
        with block[4]:
            lock_acquire(v_prod_lock, 1)
            aie.dma_bd(v_buffer_0)
            lock_release(v_cons_lock, 1)
            aie.next_bd(block[5])
        
        with block[5]:
            lock_acquire(v_prod_lock, 1)
            aie.dma_bd(v_buffer_1)
            lock_release(v_cons_lock, 1)
            aie.next_bd(block[4])
        
        with block[6]:
            s7 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[7], chain=block[11])
        
        with block[7]:
            lock_acquire(k_cons_lock, 1)
            aie.dma_bd(k_buffer_0, offset = d * mt_id, len = LK * d, dimensions = [(d // 8, 8), (LK, d * total_cols), (8, 1)])
            lock_release(k_prod_lock, 1)
            aie.next_bd(block[8])
            
        with block[8]:
            lock_acquire(v_cons_lock, 1)
            aie.dma_bd(v_buffer_0, offset = d * mt_id, len = LK * d, dimensions = [(d // 4, 4), (LK, d * total_cols), (4, 1)])
            lock_release(v_prod_lock, 1)
            aie.next_bd(block[9])
        
        with block[9]:
            lock_acquire(k_cons_lock, 1)
            aie.dma_bd(k_buffer_1, offset = d * mt_id, len = LK * d, dimensions = [(d // 8, 8), (LK, d * total_cols), (8, 1)])
            lock_release(k_prod_lock, 1)
            aie.next_bd(block[10])
        
        with block[10]:
            lock_acquire(v_cons_lock, 1)
            aie.dma_bd(v_buffer_1, offset = d * mt_id, len = LK * d, dimensions = [(d // 4, 4), (LK, d * total_cols), (4, 1)])
            lock_release(v_prod_lock, 1)
            aie.next_bd(block[7])
        
        with block[11]:
            s8 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[12], chain=block[14])
        
        with block[12]:
            lock_acquire(o_prod_lock_0, 1)
            aie.dma_bd(o_buffer_0, offset = 0, len = LQ * d, dimensions = [(LQ // 8, 8 * d), (d // 4, 4), (8, d), (4, 1)])
            lock_release(o_prod_lock_1, 1)
            aie.next_bd(block[13])
        
        with block[13]:
            lock_acquire(o_prod_lock_0, 1)
            aie.dma_bd(o_buffer_1, offset = 0, len = LQ * d, dimensions = [(LQ // 8, 8 * d), (d // 4, 4), (8, d), (4, 1)])
            lock_release(o_prod_lock_1, 1)
            aie.next_bd(block[12])
        
        with block[14]:
            s9 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest=block[15], chain=block[17])
        
        with block[15]:
            lock_acquire(o_prod_lock_1, 1)
            aie.dma_bd(o_buffer_0, offset = LQ * d, len = LQ * d, dimensions = [(LQ // 8, 8 * d), (d // 4, 4), (8, d), (4, 1)])
            lock_release(o_prod_lock_2, 1)
            aie.next_bd(block[16])

        with block[16]:
            lock_acquire(o_prod_lock_1, 1)
            aie.dma_bd(o_buffer_1, offset = LQ * d, len = LQ * d, dimensions = [(LQ // 8, 8 * d), (d // 4, 4), (8, d), (4, 1)])
            lock_release(o_prod_lock_2, 1)
            aie.next_bd(block[15])
        
        with block[17]:
            s10 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 4, dest=block[18], chain=block[20])
        
        with block[18]:
            lock_acquire(o_prod_lock_2, 1)
            aie.dma_bd(o_buffer_0, offset = 2 * LQ * d, len = LQ * d, dimensions = [(LQ // 8, 8 * d), (d // 4, 4), (8, d), (4, 1)])
            lock_release(o_prod_lock_3, 1)
            aie.next_bd(block[19])
        
        with block[19]:
            lock_acquire(o_prod_lock_2, 1)
            aie.dma_bd(o_buffer_1, offset = 2 * LQ * d, len = LQ * d, dimensions = [(LQ // 8, 8 * d), (d // 4, 4), (8, d), (4, 1)])
            lock_release(o_prod_lock_3, 1)
            aie.next_bd(block[18])
        
        with block[20]:
            s11 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 5, dest=block[21], chain=block[23])
        
        with block[21]:
            lock_acquire(o_prod_lock_3, 1)
            aie.dma_bd(o_buffer_0, offset = 3 * LQ * d, len = LQ * d, dimensions = [(LQ // 8, 8 * d), (d // 4, 4), (8, d), (4, 1)])
            lock_release(o_cons_lock, 1)
            aie.next_bd(block[22])
        
        with block[22]:
            lock_acquire(o_prod_lock_3, 1)
            aie.dma_bd(o_buffer_1, offset = 3 * LQ * d, len = LQ * d, dimensions = [(LQ // 8, 8 * d), (d // 4, 4), (8, d), (4, 1)])
            lock_release(o_cons_lock, 1)
            aie.next_bd(block[21])
        
        with block[23]:
            s12 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[24], chain=block[26])
        
        with block[24]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_0, offset = 0, len = LQ * d * HEAD_PER_COL, dimensions = [(LQ, d), (HEAD_PER_COL, LQ * d), (d, 1)])
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[25])
        
        with block[25]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_1, offset = 0, len = LQ * d * HEAD_PER_COL, dimensions = [(LQ, d), (HEAD_PER_COL, LQ * d), (d, 1)])
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[24])
        
        with block[26]:
            aie.EndOp()
