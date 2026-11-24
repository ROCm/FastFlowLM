# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import argparse

from ml_dtypes import bfloat16
import numpy as np
import sys
from aie.extras.dialects.ext import arith

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.utils.trace_events_enum import CoreEvent
from aie.helpers.dialects.ext.scf import _for as range_
from typing import List
from utils import mlir_context, tile_with_location, connect_stream

def attn_tile(tile_qk: tile_with_location, tile_kv: tile_with_location, mctx: mlir_context):
    ct = tile_qk.tile
    ct_row = tile_qk.row
    ct_col = tile_qk.col
    attn_q_ty = mctx.attn_q_ty
    attn_k_ty = mctx.attn_k_ty
    attn_s_ty = mctx.attn_s_ty
    attn_v_ty = mctx.attn_v_ty
    attn_o_ty = mctx.attn_o_ty
    sink_ty = mctx.sink_ty
    s_buffer_0 = mctx.buffer(f"s_shared_buffer_0_{ct_row + 1}_{ct_col}", tile = tile_kv, dtype = attn_s_ty)
    s_buffer_1 = mctx.buffer(f"s_shared_buffer_1_{ct_row + 1}_{ct_col}", tile = tile_kv, dtype = attn_s_ty)

    attn_qk = mctx.attn_qk
    L_qk_buffer = mctx.buffers[f"L_attn_qk_core_{ct_row}_{ct_col}"]
    #sink_qk_buffer = mctx.buffers[f"sink_qk_core_{ct_row}_{ct_col}"]
    sink_qk_buffer = aie.buffer(tile=ct, datatype=sink_ty, name=f"sink_{ct_row}_{ct_col}")
    k_buffer_0 = aie.buffer(tile = ct, datatype = attn_k_ty, name = f"k_0_{ct_row}_{ct_col}")
    k_buffer_1 = aie.buffer(tile = ct, datatype = attn_k_ty, name = f"k_1_{ct_row}_{ct_col}")
    q_buffer_0 = aie.buffer(tile = ct, datatype = attn_q_ty, name = f"q_0_{ct_row}_{ct_col}")
    q_buffer_1 = aie.buffer(tile = ct, datatype = attn_q_ty, name = f"q_1_{ct_row}_{ct_col}")

    q_prod_lock = aie.lock(ct, lock_id = 0, init = 2)
    q_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    k_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    k_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    s_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    s_cons_lock = aie.lock(ct, lock_id = 5, init = 0)
    l_prod_lock = aie.lock(ct, lock_id = 6, init = 2)
    l_cons_lock = aie.lock(ct, lock_id = 7, init = 0)

    @aie.core(ct, link_with = "attn_qk.o", stack_size = 1024 * 8)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            attn_qk(q_buffer_0, q_buffer_1, k_buffer_0, k_buffer_1, s_buffer_0, s_buffer_1, sink_qk_buffer, L_qk_buffer)

    @aie.mem(ct)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

        with block[1]:
            aie.use_lock(q_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(q_buffer_0)
            aie.use_lock(q_cons_lock, aie.LockAction.Release, value = 1)
            aie.next_bd(block[2])

        with block[2]:
            aie.use_lock(q_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(q_buffer_1)
            aie.use_lock(q_cons_lock, aie.LockAction.Release, value = 1)
            aie.next_bd(block[1])

        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])
        
        with block[4]:
            aie.use_lock(k_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(k_buffer_0)
            aie.use_lock(k_cons_lock, aie.LockAction.Release, value = 1)
            aie.next_bd(block[5])

        with block[5]:
            aie.use_lock(k_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(k_buffer_1)
            aie.use_lock(k_cons_lock, aie.LockAction.Release, value = 1)
            aie.next_bd(block[4])

        with block[6]:
            aie.EndOp()

    ct = tile_kv.tile
    ct_row = tile_kv.row
    ct_col = tile_kv.col
    # kv
    attn_kv = mctx.attn_kv
    D = mctx.DH
    attn_v_ty = mctx.attn_v_ty
    attn_o_ty = mctx.attn_o_ty
    attn_y_ty = mctx.attn_y_ty


    L_kv_buffer = mctx.buffers[f"L_attn_kv_core_{ct_row}_{ct_col}"]
    # sink_kv_buffer = mctx.buffers[f"sink_kv_core_{ct_row}_{ct_col}"] #NOTEL this will use the sink buffer from qk instead 

    v_buffer_0 = aie.buffer(tile = ct, datatype = attn_v_ty, name = f"v_0_{ct_row}_{ct_col}")
    v_buffer_1 = aie.buffer(tile = ct, datatype = attn_v_ty, name = f"v_1_{ct_row}_{ct_col}")
    o_buffer_0 = aie.buffer(tile = ct, datatype = attn_o_ty, name = f"o_0_{ct_row}_{ct_col}")
    o_buffer_1 = aie.buffer(tile = ct, datatype = attn_o_ty, name = f"o_1_{ct_row}_{ct_col}")
    y_buffer = aie.buffer(tile = ct, datatype = attn_y_ty, name = f"y_{ct_row}_{ct_col}")

    o_prod_lock = aie.lock(ct, lock_id = 0, init = 2)
    o_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    v_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    v_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    s_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    s_cons_lock = aie.lock(ct, lock_id = 5, init = 0)

    @aie.core(ct, link_with = "attn_kv.o", stack_size = 1024 * 8)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            attn_kv(s_buffer_0, s_buffer_1, v_buffer_0, v_buffer_1, o_buffer_0, o_buffer_1, y_buffer, sink_qk_buffer, L_kv_buffer)

    @aie.mem(ct)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[1], chain=block[3])
        
        with block[1]:
            aie.use_lock(v_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(v_buffer_0)
            aie.use_lock(v_cons_lock, aie.LockAction.Release, value = 1)
            aie.next_bd(block[2])
            
        with block[2]:
            aie.use_lock(v_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(v_buffer_1)
            aie.use_lock(v_cons_lock, aie.LockAction.Release, value = 1)
            aie.next_bd(block[1])

        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

        with block[4]:
            aie.use_lock(o_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(o_buffer_0)
            aie.use_lock(o_prod_lock, aie.LockAction.Release, value = 1)
            aie.next_bd(block[5])
        
        with block[5]:
            aie.use_lock(o_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(o_buffer_1)
            aie.use_lock(o_prod_lock, aie.LockAction.Release, value = 1)
            aie.next_bd(block[4])
            
        with block[6]:
            aie.EndOp()

def attn_memtile(tile: tile_with_location, mctx: mlir_context):
    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    DH = mctx.DH

    attn_k_mem_ty = mctx.attn_k_mem_ty
    attn_v_mem_ty = mctx.attn_v_mem_ty

    k_buffer_0 = aie.buffer(mt, attn_k_mem_ty, name = f"k_buffer_0_{mt_row}_{mt_col}")
    k_buffer_1 = aie.buffer(mt, attn_k_mem_ty, name = f"k_buffer_1_{mt_row}_{mt_col}")
    v_buffer_0 = aie.buffer(mt, attn_v_mem_ty, f"v_buffer_0_{mt_row}_{mt_col}")
    v_buffer_1 = aie.buffer(mt, attn_v_mem_ty, f"v_buffer_1_{mt_row}_{mt_col}")

    k_prod_lock = aie.lock(mt, lock_id = 0, init = 2)
    k_cons_lock_0 = aie.lock(mt, lock_id = 1, init = 0)
    k_cons_lock_1 = aie.lock(mt, lock_id = 2, init = 0)
    v_prod_lock = aie.lock(mt, lock_id = 3, init = 2)
    v_cons_lock_0 = aie.lock(mt, lock_id = 4, init = 0)
    v_cons_lock_1 = aie.lock(mt, lock_id = 5, init = 0)

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

        with block[1]:
            aie.use_lock(k_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
            aie.dma_bd(k_buffer_0)
            aie.use_lock(k_cons_lock_0, aie.LockAction.Release, value=1)
            aie.next_bd(block[2])

        with block[2]:
            aie.use_lock(k_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
            aie.dma_bd(k_buffer_1)
            aie.use_lock(k_cons_lock_0, aie.LockAction.Release, value=1)
            aie.next_bd(block[1])

        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

        with block[4]:
            aie.use_lock(k_cons_lock_0, aie.LockAction.AcquireGreaterEqual, value=1)
            aie.dma_bd(k_buffer_0, offset = 0, len = 16 * DH * 2, dimensions = [(DH * 2 // 8, 8), (16, DH * 4), (8, 1)])
            aie.use_lock(k_cons_lock_1, aie.LockAction.Release, value=1)
            aie.next_bd(block[5])

        with block[5]:
            aie.use_lock(k_cons_lock_0, aie.LockAction.AcquireGreaterEqual, value=1)
            aie.dma_bd(k_buffer_1, offset = 0, len = 16 * DH * 2, dimensions = [(DH * 2 // 8, 8), (16, DH * 4), (8, 1)])
            aie.use_lock(k_cons_lock_1, aie.LockAction.Release, value=1)
            aie.next_bd(block[4])
        
        with block[6]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[7], chain=block[9])
        
        with block[7]:
            aie.use_lock(k_cons_lock_1, aie.LockAction.AcquireGreaterEqual, value=1)
            aie.dma_bd(k_buffer_0, offset = DH * 2, len = 16 * DH * 2, dimensions = [(DH * 2 // 8, 8), (16, DH * 4), (8, 1)])
            aie.use_lock(k_prod_lock, aie.LockAction.Release, value=1)
            aie.next_bd(block[8])
        
        with block[8]:
            aie.use_lock(k_cons_lock_1, aie.LockAction.AcquireGreaterEqual, value=1)
            aie.dma_bd(k_buffer_1, offset = DH * 2, len = 16 * DH * 2, dimensions = [(DH * 2 // 8, 8), (16, DH * 4), (8, 1)])
            aie.use_lock(k_prod_lock, aie.LockAction.Release, value=1)
            aie.next_bd(block[7])

        with block[9]:
            s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[10], chain=block[12])

        with block[10]:
            aie.use_lock(v_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
            aie.dma_bd(v_buffer_0)
            aie.use_lock(v_cons_lock_0, aie.LockAction.Release, value=1)
            aie.next_bd(block[11])

        with block[11]:
            aie.use_lock(v_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
            aie.dma_bd(v_buffer_1)
            aie.use_lock(v_cons_lock_0, aie.LockAction.Release, value=1)
            aie.next_bd(block[10])

        with block[12]:
            s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[13], chain=block[15])

        with block[13]:
            aie.use_lock(v_cons_lock_0, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(v_buffer_0, offset = 0, len = 16 * DH * 2, dimensions = [(2, 8 * DH * 4), (DH * 2 // 8, 8), (8, DH * 4), (8, 1)])
            aie.use_lock(v_cons_lock_1, aie.LockAction.Release, value = 1)
            aie.next_bd(block[14])

        with block[14]:
            aie.use_lock(v_cons_lock_0, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(v_buffer_1, offset = 0, len = 16 * DH * 2, dimensions = [(2, 8 * DH * 4), (DH * 2 // 8, 8), (8, DH * 4), (8, 1)])
            aie.use_lock(v_cons_lock_1, aie.LockAction.Release, value = 1)
            aie.next_bd(block[13])
        
        with block[15]:
            s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 3, dest=block[16], chain=block[18])
        
        with block[16]:
            aie.use_lock(v_cons_lock_1, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(v_buffer_0, offset = DH * 2, len = 16 * DH * 2, dimensions = [(2, 8 * DH * 4), (DH * 2 // 8, 8), (8, DH * 4), (8, 1)])
            aie.use_lock(v_prod_lock, aie.LockAction.Release, value = 1)
            aie.next_bd(block[17])
        
        with block[17]:
            aie.use_lock(v_cons_lock_1, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(v_buffer_1, offset = DH * 2, len = 16 * DH * 2, dimensions = [(2, 8 * DH * 4), (DH * 2 // 8, 8), (8, DH * 4), (8, 1)])
            aie.use_lock(v_prod_lock, aie.LockAction.Release, value = 1)
            aie.next_bd(block[16])
        
        with block[18]:
            aie.EndOp()

