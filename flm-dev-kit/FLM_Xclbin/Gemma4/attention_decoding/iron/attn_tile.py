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
def attn_cu(tile_qk: tile_with_location, tile_kv: tile_with_location, mctx: mlir_context):

    ct_qk = tile_qk.tile
    ct_qk_row = tile_qk.row
    ct_qk_col = tile_qk.col
    
    ct_kv = tile_kv.tile
    ct_kv_row = tile_kv.row
    ct_kv_col = tile_kv.col

    attn_qk = mctx.attn_qk
    attn_kv = mctx.attn_kv
    
    o_ty = mctx.o_ty
    s_ty = mctx.s_ty
    q_ty = mctx.q_ty
    k_ty = mctx.k_ty
    v_ty = mctx.v_ty
    y_ty = mctx.y_ty
    
    Q_HEADS_PER_ROUND = mctx.Q_HEADS_PER_ROUND
    DH = mctx.DH
    
    L_qk = mctx.buffers[f"RTP_L_qk_{ct_qk_row}_{ct_qk_col}"]
    L_kv = mctx.buffers[f"RTP_L_kv_{ct_kv_row}_{ct_kv_col}"]

    q_buffer_0 = aie.buffer(tile = ct_qk, datatype = q_ty, name = f"q_0_{ct_qk_row}_{ct_qk_col}")
    q_buffer_1 = aie.buffer(tile = ct_qk, datatype = q_ty, name = f"q_1_{ct_qk_row}_{ct_qk_col}")
    k_buffer_0 = aie.buffer(tile = ct_qk, datatype = k_ty, name = f"k_0_{ct_qk_row}_{ct_qk_col}")
    k_buffer_1 = aie.buffer(tile = ct_qk, datatype = k_ty, name = f"k_1_{ct_qk_row}_{ct_qk_col}")
    s_buffer_0 = aie.buffer(tile = ct_qk, datatype = s_ty, name = f"s_0_{ct_qk_row}_{ct_qk_col}")
    s_buffer_1 = aie.buffer(tile = ct_qk, datatype = s_ty, name = f"s_1_{ct_qk_row}_{ct_qk_col}")
    
    v_buffer_0 = aie.buffer(tile = ct_kv, datatype = v_ty, name = f"v_0_{ct_kv_row}_{ct_kv_col}")
    v_buffer_1 = aie.buffer(tile = ct_kv, datatype = v_ty, name = f"v_1_{ct_kv_row}_{ct_kv_col}")
    y_buffer = aie.buffer(tile = ct_kv, datatype = y_ty, name = f"y_{ct_kv_row}_{ct_kv_col}")
    o_buffer = aie.buffer(tile = ct_kv, datatype = o_ty, name = f"o_{ct_kv_row}_{ct_kv_col}")
    
    # qk ct
    q_prod_lock = aie.lock(ct_qk, lock_id = 0, init = 2)
    q_cons_lock = aie.lock(ct_qk, lock_id = 1, init = 0)
    k_prod_lock = aie.lock(ct_qk, lock_id = 2, init = 2)
    k_cons_lock = aie.lock(ct_qk, lock_id = 3, init = 0)
    s_prod_lock_0 = aie.lock(ct_qk, lock_id = 4, init = 1)
    s_cons_lock_0 = aie.lock(ct_qk, lock_id = 5, init = 0)
    s_prod_lock_1 = aie.lock(ct_qk, lock_id = 6, init = 1)
    s_cons_lock_1 = aie.lock(ct_qk, lock_id = 7, init = 0)
    l_prod_lock = aie.lock(ct_qk, lock_id = 8, init = 0)
    
    # kv ct
    v_prod_lock = aie.lock(ct_kv, lock_id = 2, init = 2)
    v_cons_lock = aie.lock(ct_kv, lock_id = 3, init = 0)
    o_prod_lock = aie.lock(ct_kv, lock_id = 0, init = 1)
    o_cons_lock = aie.lock(ct_kv, lock_id = 1, init = 0)
    l_prod_lock = aie.lock(ct_kv, lock_id = 8, init = 0)

    @aie.core(ct_qk, link_with = "attn_qk.o", stack_size = 1024*4)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            attn_qk(q_buffer_0, q_buffer_1, k_buffer_0, k_buffer_1, s_buffer_0, s_buffer_1, L_qk)
    
    @aie.core(ct_kv, link_with = "attn_kv.o", stack_size = 1024*6)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            attn_kv(s_buffer_0, s_buffer_1, v_buffer_0, v_buffer_1, y_buffer, o_buffer, L_kv)
    
    @aie.mem(ct_qk)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

        with block[1]:
            lock_acquire(q_prod_lock, 1)
            aie.dma_bd(q_buffer_0, offset=0, len=Q_HEADS_PER_ROUND * DH, dimensions = [(Q_HEADS_PER_ROUND, 8), (DH // 8, Q_HEADS_PER_ROUND * 8), (8, 1)])
            lock_release(q_cons_lock, 1)
            aie.next_bd(block[2]) 
        
        with block[2]:
            lock_acquire(q_prod_lock, 1)
            aie.dma_bd(q_buffer_1, offset=0, len=Q_HEADS_PER_ROUND * DH, dimensions = [(Q_HEADS_PER_ROUND, 8), (DH // 8, Q_HEADS_PER_ROUND * 8), (8, 1)])
            lock_release(q_cons_lock, 1)
            aie.next_bd(block[1])

        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])
        
        with block[4]:
            lock_acquire(k_prod_lock, 1)
            aie.dma_bd(k_buffer_0)
            lock_release(k_cons_lock, 1)
            aie.next_bd(block[5])
        
        with block[5]:
            lock_acquire(k_prod_lock, 1)
            aie.dma_bd(k_buffer_1)
            lock_release(k_cons_lock, 1)
            aie.next_bd(block[4])
        
        with block[6]:
            aie.EndOp()
    
    @aie.mem(ct_kv)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[1], chain=block[3])

        with block[1]:
            lock_acquire(v_prod_lock, 1)
            aie.dma_bd(v_buffer_0)
            lock_release(v_cons_lock, 1)
            aie.next_bd(block[2]) 
        
        with block[2]:
            lock_acquire(v_prod_lock, 1)
            aie.dma_bd(v_buffer_1)
            lock_release(v_cons_lock, 1)
            aie.next_bd(block[1])

        with block[3]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[5])
        
        with block[4]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer, offset=0, len=Q_HEADS_PER_ROUND * DH, dimensions = [(Q_HEADS_PER_ROUND, 8), (DH // 8, Q_HEADS_PER_ROUND * 8), (8, 1)])
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[4])
        
        with block[5]:
            aie.EndOp()

def attn_memtile(tile: tile_with_location, mctx: mlir_context):
    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    k_mem_ty = mctx.k_mem_ty
    v_mem_ty = mctx.v_mem_ty
    
    DH = mctx.DH
    lc = mctx.lc
    
    k_buffer_0 = aie.buffer(tile = mt, datatype = k_mem_ty, name = f"k_0_{mt_row}_{mt_col}")
    k_buffer_1 = aie.buffer(tile = mt, datatype = k_mem_ty, name = f"k_1_{mt_row}_{mt_col}")
    v_buffer_0 = aie.buffer(tile = mt, datatype = v_mem_ty, name = f"v_0_{mt_row}_{mt_col}")
    v_buffer_1 = aie.buffer(tile = mt, datatype = v_mem_ty, name = f"v_1_{mt_row}_{mt_col}")
    
    k_prod_lock = aie.lock(mt, lock_id = 0, init = 2)
    k_cons_lock = aie.lock(mt, lock_id = 1, init = 0)
    v_prod_lock = aie.lock(mt, lock_id = 2, init = 2)
    v_cons_lock = aie.lock(mt, lock_id = 3, init = 0)

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
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
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])
        
        with block[4]:
            lock_acquire(k_cons_lock, 1)
            aie.dma_bd(k_buffer_0, offset = 0, len = lc * DH * 2, dimensions = [(2 * DH // 8, 8), (lc, DH * 2), (8, 1)])
            lock_release(k_prod_lock, 1)
            aie.next_bd(block[5])
        
        with block[5]:
            lock_acquire(k_cons_lock, 1)
            aie.dma_bd(k_buffer_1, offset = 0, len = lc * DH * 2, dimensions = [(2 * DH // 8, 8), (lc, DH * 2), (8, 1)])
            lock_release(k_prod_lock, 1)
            aie.next_bd(block[4])
        
        with block[6]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[7], chain=block[9])
        
        with block[7]:
            lock_acquire(v_prod_lock, 1)
            aie.dma_bd(v_buffer_0)
            lock_release(v_cons_lock, 1)
            aie.next_bd(block[8])
        
        with block[8]:
            lock_acquire(v_prod_lock, 1)
            aie.dma_bd(v_buffer_1)
            lock_release(v_cons_lock, 1)
            aie.next_bd(block[7])
            
        with block[9]:
            s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[10], chain=block[12])
        
        with block[10]:
            lock_acquire(v_cons_lock, 1)
            aie.dma_bd(v_buffer_0, offset = 0, len = lc * DH * 2, dimensions = [(2 * DH // 8, 8), (lc, DH * 2), (8, 1)])
            lock_release(v_prod_lock, 1)
            aie.next_bd(block[11])
        
        with block[11]:
            lock_acquire(v_cons_lock, 1)
            aie.dma_bd(v_buffer_1, offset = 0, len = lc * DH * 2, dimensions = [(2 * DH // 8, 8), (lc, DH * 2), (8, 1)])
            lock_release(v_prod_lock, 1)
            aie.next_bd(block[10])
        
        with block[12]:
            aie.EndOp()

def aux_memtile(tile: tile_with_location, mctx: mlir_context):
    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    q_mem_ty = mctx.q_mem_ty
    o_mem_ty = mctx.o_mem_ty
    
    q_buffer_0 = aie.buffer(tile = mt, datatype = q_mem_ty, name = f"q_0_{mt_row}_{mt_col}")
    q_buffer_1 = aie.buffer(tile = mt, datatype = q_mem_ty, name = f"q_1_{mt_row}_{mt_col}")
    o_buffer_0 = aie.buffer(tile = mt, datatype = o_mem_ty, name = f"o_0_{mt_row}_{mt_col}")
    o_buffer_1 = aie.buffer(tile = mt, datatype = o_mem_ty, name = f"o_1_{mt_row}_{mt_col}")
    
    q_prod_lock = aie.lock(mt, lock_id = 0, init = 2)
    q_cons_lock = aie.lock(mt, lock_id = 1, init = 0)
    o_prod_lock = aie.lock(mt, lock_id = 2, init = 2)
    o_cons_lock = aie.lock(mt, lock_id = 3, init = 0)

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
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
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])
        
        with block[4]:
            lock_acquire(q_cons_lock, 1)
            aie.dma_bd(q_buffer_0)
            lock_release(q_prod_lock, 1)
            aie.next_bd(block[5])
        
        with block[5]:
            lock_acquire(q_cons_lock, 1)
            aie.dma_bd(q_buffer_1)
            lock_release(q_prod_lock, 1)
            aie.next_bd(block[4])
        
        with block[6]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[7], chain=block[9])
        
        with block[7]:
            lock_acquire(o_prod_lock, 1)
            aie.dma_bd(o_buffer_0)
            lock_release(o_cons_lock, 1)
            aie.next_bd(block[8])
        
        with block[8]:
            lock_acquire(o_prod_lock, 1)
            aie.dma_bd(o_buffer_1)
            lock_release(o_cons_lock, 1)
            aie.next_bd(block[7])
        
        with block[9]:
            s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[10], chain=block[12])
        
        with block[10]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_0)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[11])
        
        with block[11]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_1)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[10])
        
        with block[12]:
            aie.EndOp()