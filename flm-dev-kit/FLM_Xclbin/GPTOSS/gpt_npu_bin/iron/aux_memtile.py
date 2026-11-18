# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
import aie.dialects.aie as aie
import aie.dialects.aiex as aiex
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils
from aie.utils.trace import MemTilePortEvent, MemTileEvent, PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from aie.dialects import memref

from utils import mlir_context, tile_with_location, lock_acquire, lock_release

def aux_memtile(tile: tile_with_location, ctx: mlir_context):

    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    DH = ctx.parameters["DH"]
    D = ctx.parameters["D"]
    D_PADDED_Q4NX_COL = ctx.parameters["D_PADDED_Q4NX_COL"]
    D_PADDED_FOUR_Q4NX_ROW = ctx.parameters["D_PADDED_FOUR_Q4NX_ROW"]
    Q4NX_ROW = ctx.parameters["Q4NX_ROW"]
    
    down_repeats = (D_PADDED_FOUR_Q4NX_ROW)//Q4NX_ROW// 4  # each column process separate expert
    out_repeats = 6
    
    q_ty = ctx.data_tys["q_ty"]
    out_ty = ctx.data_tys["out_ty"]

    down_ty = ctx.data_tys["down_ty"]
    down_prod_lock = aie.lock(mt, lock_id = 0, init = down_repeats)
    down_cons_lock = aie.lock(mt, lock_id = 1, init = 0)
    q_prod_lock = aie.lock(mt, lock_id = 2, init = 1)
    q_cons_lock_0 = aie.lock(mt, lock_id = 3, init = 0)
    q_cons_lock_1 = aie.lock(mt, lock_id = 4, init = 0)
    q_cons_lock_2 = aie.lock(mt, lock_id = 5, init = 0)
    q_cons_lock_3 = aie.lock(mt, lock_id = 6, init = 0)    
    o_prod_lock_0 = aie.lock(mt, lock_id = 7, init = out_repeats)
    o_prod_lock_1 = aie.lock(mt, lock_id = 8, init = 0)
    o_prod_lock_2 = aie.lock(mt, lock_id = 9, init = 0)
    o_prod_lock_3 = aie.lock(mt, lock_id = 10, init = 0)
    o_cons_lock = aie.lock(mt, lock_id = 11, init = 0)
    
    sink_cons_lock_0 = aie.lock(mt, lock_id=12, init=0)
    sink_cons_lock_1 = aie.lock(mt, lock_id=13, init=0)
    sink_cons_lock_2 = aie.lock(mt, lock_id=14, init=0)
    sink_cons_lock_3 = aie.lock(mt, lock_id=15, init=0)

    attn_q_mem_ty = ctx.attn_q_mem_ty
    attn_o_mem_ty = ctx.attn_o_mem_ty

    q_buffer = aie.buffer(mt, attn_q_mem_ty, f"q_buffer_{mt_row}_{mt_col}")
    o_buffer = aie.buffer(mt, attn_o_mem_ty, f"o_buffer_{mt_row}_{mt_col}")

    G = ctx.parameters["kv_groups"]
    num_attn_heads = ctx.parameters["num_attn_heads"]

    down_buffer = aie.buffer(tile = mt, datatype = down_ty, name = f"down_buffer_{mt_row}_{mt_col}")
    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        with block[0]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[1], chain = block[3])

        with block[1]:
            lock_acquire(down_prod_lock, down_repeats)
            aie.dma_bd(down_buffer)
            lock_release(down_cons_lock, down_repeats)
            aie.next_bd(block[2])

        with block[2]:
            lock_acquire(down_prod_lock,down_repeats)
            aie.dma_bd(down_buffer)
            lock_release(down_cons_lock, down_repeats)
            aie.next_bd(block[1])


        with block[3]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[4], chain = block[6])

        with block[4]:
            lock_acquire(down_cons_lock, 1)
            aie.dma_bd_packet(0, 0)
            aie.dma_bd(down_buffer)
            lock_release(down_prod_lock, 1)
            aie.next_bd(block[4])


        with block[6]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[7], chain = block[8])

        with block[7]:
            lock_acquire(q_prod_lock, 1)
            aie.dma_bd(q_buffer)
            lock_release(q_cons_lock_0, 1)
            aie.next_bd(block[7])

        with block[8]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest = block[9], chain = block[11])

        with block[9]:
            lock_acquire(q_cons_lock_0, 1)
            aie.dma_bd(q_buffer, offset = 0, len = 16 * DH, dimensions =  [(2, 8 * DH), (DH // 8, 8), (8, DH), (8, 1)])
            lock_release(sink_cons_lock_0, 1)
            aie.next_bd(block[10])
        with block[10]:
            assert num_attn_heads//4 == 16 # sanity checks
            assert num_attn_heads == 64
            lock_acquire(sink_cons_lock_0, 1)
            aie.dma_bd(q_buffer, offset = DH*num_attn_heads, len=16)  # div 4 because 4 pair of q-k-v core
            lock_release(q_cons_lock_1, 1 )
            aie.next_bd(block[9])
            
        with block[11]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest = block[12], chain = block[14])

        with block[12]:
            lock_acquire(q_cons_lock_1, 1)
            aie.dma_bd(q_buffer, offset = 16 * DH, len = 16 * DH, dimensions = [(2, 8 * DH), (DH // 8, 8), (8, DH), (8, 1)])
            lock_release(sink_cons_lock_1, 1)
            aie.next_bd(block[13])
        with block[13]:
            lock_acquire(sink_cons_lock_1, 1 )
            aie.dma_bd(q_buffer, offset = DH*num_attn_heads + 16  , len= 16) 
            lock_release( q_cons_lock_2, 1)
            aie.next_bd(block[12])
        with block[14]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 3, dest = block[15], chain = block[17])

        with block[15]:
            lock_acquire(q_cons_lock_2, 1)
            aie.dma_bd(q_buffer, offset = 32 * DH, len = 16 * DH, dimensions = [(2, 8 * DH), (DH // 8, 8), (8, DH), (8, 1)])
            lock_release(sink_cons_lock_2, 1)
            aie.next_bd(block[16])
        with block[16]:
            lock_acquire(sink_cons_lock_2, 1)
            aie.dma_bd(q_buffer, offset = DH*num_attn_heads + 2*16  , len= 16) 
            lock_release(q_cons_lock_3, 1)
            aie.next_bd(block[15])
        
        with block[17]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 4, dest = block[18], chain = block[20])

        with block[18]:
            lock_acquire(q_cons_lock_3, 1)
            aie.dma_bd(q_buffer, offset = 48 * DH, len = 16 * DH, dimensions = [(2, 8 * DH), (DH // 8, 8), (8, DH), (8, 1)])
            lock_release(sink_cons_lock_3, 1)
            aie.next_bd(block[19])
        with block[19]:
            lock_acquire(sink_cons_lock_3, 1)
            aie.dma_bd(q_buffer, offset = DH*num_attn_heads + 3*16  , len= 16) 
            lock_release(q_prod_lock, 1)
            aie.next_bd(block[18])
            
        with block[20]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest = block[21], chain = block[22])

        with block[21]:
            lock_acquire(o_prod_lock_0, out_repeats)
            aie.dma_bd(o_buffer, offset = 0, len = 16 * DH, dimensions = [(2, 8 * DH), (DH // 8, 8), (8, DH), (8, 1)])
            lock_release(o_prod_lock_1, 1)
            aie.next_bd(block[21])

        with block[22]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest = block[23], chain = block[24])

        with block[23]:
            lock_acquire(o_prod_lock_1, 1)
            aie.dma_bd(o_buffer, offset = 16 * DH, len = 16 * DH, dimensions = [(2, 8 * DH), (DH // 8, 8), (8, DH), (8, 1)])
            lock_release(o_prod_lock_2, 1)
            aie.next_bd(block[23])

        with block[24]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 4, dest = block[25], chain = block[26])

        with block[25]:
            lock_acquire(o_prod_lock_2, 1)
            aie.dma_bd(o_buffer, offset = 32 * DH, len = 16 * DH, dimensions = [(2, 8 * DH), (DH // 8, 8), (8, DH), (8, 1)])
            lock_release(o_prod_lock_3, 1)
            aie.next_bd(block[25])

        with block[26]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 5, dest = block[27], chain = block[28])

        with block[27]:
            lock_acquire(o_prod_lock_3, 1)
            aie.dma_bd(o_buffer, offset = 48 * DH, len = 16 * DH, dimensions = [(2, 8 * DH), (DH // 8, 8), (8, DH), (8, 1)])
            lock_release(o_cons_lock, out_repeats)
            aie.next_bd(block[27])

        with block[28]:
            aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 5, dest = block[29], chain = block[30])

        with block[29]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd_packet(0, 2)
            aie.dma_bd(o_buffer)
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[29])

        with block[30]:
            aie.EndOp()
            