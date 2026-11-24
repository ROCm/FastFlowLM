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

from utils import mlir_context, tile_with_location, connect_dma, dma_port, connect_port, lock_acquire, lock_release
from typing import List
from dataclasses import dataclass
# phase 1: qkv proj, D -> (D + D // 2) to RoPE

# phase 2: o_proj, D -> D to rms_residual

# phase 3: up/gate proj, D -> D * 8 to MT_GLU

# phase 4: down proj, D * 4 -> D, to rms_residual

# minimal output unit: (m, 1)

def proj_tile(tile: tile_with_location, ctx: mlir_context):

    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    proj = ctx.funcs["proj"]
    D = ctx.parameters["D"]
    m = ctx.parameters["m"]
    x_slice_ty = ctx.data_tys["x_slice_ty"]
    m_pkt_ty = ctx.data_tys["m_pkt_ty"]
    linear_w_ty = ctx.data_tys["linear_w_ty"]

    # q_proj tile and mem 
    # double buffer
    x_buffer_0 = aie.buffer(tile = ct, datatype = x_slice_ty, name = f"x_0_{ct_row}_{ct_col}")
    w_buffer_0 = aie.buffer(tile = ct, datatype = linear_w_ty, name = f"w_0_{ct_row}_{ct_col}")
    y_buffer_0 = aie.buffer(tile = ct, datatype = m_pkt_ty, name = f"y_0_{ct_row}_{ct_col}")
    x_buffer_1 = aie.buffer(tile = ct, datatype = x_slice_ty, name = f"x_1_{ct_row}_{ct_col}")
    w_buffer_1 = aie.buffer(tile = ct, datatype = linear_w_ty, name = f"w_1_{ct_row}_{ct_col}")
    y_buffer_1 = aie.buffer(tile = ct, datatype = m_pkt_ty, name = f"y_1_{ct_row}_{ct_col}")

    x_prod_lock = aie.lock(ct, lock_id = 0, init = 2)
    x_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    w_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    w_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    y_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    y_cons_lock = aie.lock(ct, lock_id = 5, init = 0)

    @aie.core(ct, link_with = "proj.o", stack_size = 1024 * 2)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            proj(y_buffer_0, w_buffer_0, x_buffer_0, y_buffer_1, w_buffer_1, x_buffer_1)
    
    @aie.mem(ct)
    def mem_body(block):
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
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[7], chain = block[9])
            
        with block[7]: # x, to mt
            lock_acquire(y_cons_lock, 1)
            aie.dma_bd(y_buffer_0, offset = 14, len = m + 2)
            lock_release(y_prod_lock, 1)
            aie.next_bd(block[8])

        with block[8]: # w, to mt
            lock_acquire(y_cons_lock, 1)
            aie.dma_bd(y_buffer_1, offset = 14, len = m + 2)
            lock_release(y_prod_lock, 1)
            aie.next_bd(block[7])

        with block[9]:
            aie.EndOp()


def memtile_assemble(tile: tile_with_location, ctx: mlir_context):

    mt = tile.tile
    row = tile.row
    col = tile.col

    m = ctx.parameters["m"]
    k = ctx.parameters["k"]
    w_block_size = ctx.parameters["w_block_size"]
    m_ty = ctx.data_tys["m_ty"]
    m_col_ty = ctx.data_tys["m_col_ty"]

    linear_4w_ty = ctx.data_tys["linear_4w_ty"]

    y_buffer_0 = aie.buffer(mt, m_col_ty, name = f"y_buffer_0_{row}_{col}")
    y_buffer_1 = aie.buffer(mt, m_col_ty, name = f"y_buffer_1_{row}_{col}")

    w_buffer_0 = aie.buffer(mt, linear_4w_ty, name = f"w_buffer_0_{row}_{col}")
    w_buffer_1 = aie.buffer(mt, linear_4w_ty, name = f"w_buffer_1_{row}_{col}")

    y_prod_lock_0 = aie.lock(mt, lock_id = 0, init = 2)
    y_prod_lock_1 = aie.lock(mt, lock_id = 1, init = 0)
    y_prod_lock_2 = aie.lock(mt, lock_id = 2, init = 0)
    y_prod_lock_3 = aie.lock(mt, lock_id = 3, init = 0)
    y_cons_lock = aie.lock(mt, lock_id = 4, init = 0)
    w_p0_prod_lock = aie.lock(mt, lock_id = 5, init = 2)
    w_p0_cons_lock_0 = aie.lock(mt, lock_id = 6, init = 0)
    w_p0_cons_lock_1 = aie.lock(mt, lock_id = 7, init = 0)
    w_p1_prod_lock = aie.lock(mt, lock_id = 8, init = 2)
    w_p1_cons_lock_0 = aie.lock(mt, lock_id = 9, init = 0)
    w_p1_cons_lock_1 = aie.lock(mt, lock_id = 10, init = 0)

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[3])
        with block[1]:
            lock_acquire(y_prod_lock_0, 1)
            aie.dma_bd(y_buffer_0, offset = 0, len = m + 2)
            lock_release(y_prod_lock_1, 1)
            aie.next_bd(block[2])

        with block[2]:
            lock_acquire(y_prod_lock_0, 1)
            aie.dma_bd(y_buffer_1, offset = 0, len = m + 2)
            lock_release(y_prod_lock_1, 1)
            aie.next_bd(block[1])
            
        with block[3]:
            s1 = aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[4], chain = block[6])
        
        with block[4]:
            lock_acquire(y_prod_lock_1, 1)
            aie.dma_bd(y_buffer_0, offset = m + 2, len = m)
            lock_release(y_prod_lock_2, 1)
            aie.next_bd(block[5])
            
        with block[5]:
            lock_acquire(y_prod_lock_1, 1)
            aie.dma_bd(y_buffer_1, offset = m + 2, len = m)
            lock_release(y_prod_lock_2, 1)
            aie.next_bd(block[4])
            
        with block[6]:
            s2 = aie.dma_start(aie.DMAChannelDir.S2MM, 2, dest = block[7], chain = block[9])
        
        with block[7]:
            lock_acquire(y_prod_lock_2, 1)
            aie.dma_bd(y_buffer_0, offset = 2 * m + 2, len = m)
            lock_release(y_prod_lock_3, 1)
            aie.next_bd(block[8])

        with block[8]:
            lock_acquire(y_prod_lock_2, 1)
            aie.dma_bd(y_buffer_1, offset = 2 * m + 2, len = m)
            lock_release(y_prod_lock_3, 1)
            aie.next_bd(block[7])

        with block[9]:
            s3 = aie.dma_start(aie.DMAChannelDir.S2MM, 3, dest = block[10], chain = block[12])

        with block[10]:
            lock_acquire(y_prod_lock_3, 1)
            aie.dma_bd(y_buffer_0, offset = 3 * m + 2, len = m)
            lock_release(y_cons_lock, 1)
            aie.next_bd(block[11])

        with block[11]:
            lock_acquire(y_prod_lock_3, 1)
            aie.dma_bd(y_buffer_1, offset = 3 * m + 2, len = m)
            lock_release(y_cons_lock, 1)
            aie.next_bd(block[10])
            
        with block[12]:
            s4 = aie.dma_start(aie.DMAChannelDir.S2MM, 4, dest = block[13], chain = block[15])

        with block[13]:
            lock_acquire(w_p0_prod_lock, 1)
            aie.dma_bd(w_buffer_0, offset = 0, len = 2 * w_block_size)
            lock_release(w_p0_cons_lock_0, 1)
            aie.next_bd(block[14])
            
        with block[14]:
            lock_acquire(w_p0_prod_lock, 1)
            aie.dma_bd(w_buffer_1, offset = 0, len = 2 * w_block_size)
            lock_release(w_p0_cons_lock_0, 1)
            aie.next_bd(block[13])

        with block[15]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[16], chain = block[18])

        with block[16]:
            lock_acquire(w_p0_cons_lock_0, 1)
            aie.dma_bd(w_buffer_0, offset = 0, len = w_block_size)
            lock_release(w_p0_cons_lock_1, 1)
            aie.next_bd(block[17])

        with block[17]:
            lock_acquire(w_p0_cons_lock_0, 1)
            aie.dma_bd(w_buffer_1, offset = 0, len = w_block_size)
            lock_release(w_p0_cons_lock_1, 1)
            aie.next_bd(block[16])

        with block[18]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[19], chain = block[21])

        with block[19]:
            lock_acquire(w_p0_cons_lock_1, 1)
            aie.dma_bd(w_buffer_0, offset = w_block_size, len = w_block_size)
            lock_release(w_p0_prod_lock, 1)
            aie.next_bd(block[20])
            
        with block[20]:
            lock_acquire(w_p0_cons_lock_1, 1)
            aie.dma_bd(w_buffer_1, offset = w_block_size, len = w_block_size)
            lock_release(w_p0_prod_lock, 1)
            aie.next_bd(block[19])

        with block[21]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 2, dest = block[22], chain = block[24])

        with block[22]:
            lock_acquire(w_p1_cons_lock_0, 1)
            aie.dma_bd(w_buffer_0, offset = 2 * w_block_size, len = w_block_size)
            lock_release(w_p1_cons_lock_1, 1)
            aie.next_bd(block[23])

        with block[23]:
            lock_acquire(w_p1_cons_lock_0, 1)
            aie.dma_bd(w_buffer_1, offset = 2 * w_block_size, len = w_block_size)
            lock_release(w_p1_cons_lock_1, 1)
            aie.next_bd(block[22])
            
        with block[24]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 3, dest = block[25], chain = block[27])

        with block[25]:
            lock_acquire(w_p1_cons_lock_1, 1)
            aie.dma_bd(w_buffer_0, offset = 3 * w_block_size, len = w_block_size)
            lock_release(w_p1_prod_lock, 1)
            aie.next_bd(block[26])

        with block[26]:
            lock_acquire(w_p1_cons_lock_1, 1)
            aie.dma_bd(w_buffer_1, offset = 3 * w_block_size, len = w_block_size)
            lock_release(w_p1_prod_lock, 1)
            aie.next_bd(block[25])

        with block[27]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 4, dest = block[28], chain = block[30])

        with block[28]:
            lock_acquire(y_cons_lock, 1)
            aie.dma_bd(y_buffer_0)
            lock_release(y_prod_lock_0, 1)
            aie.next_bd(block[29])

        with block[29]:
            lock_acquire(y_cons_lock, 1)
            aie.dma_bd(y_buffer_1)
            lock_release(y_prod_lock_0, 1)
            aie.next_bd(block[28])

        with block[30]:
            aie.dma_start(aie.DMAChannelDir.S2MM, 5, dest = block[31], chain = block[33])
        
        with block[31]:
            lock_acquire(w_p1_prod_lock, 1)
            aie.dma_bd(w_buffer_0, offset = 2 * w_block_size, len = 2 * w_block_size)
            lock_release(w_p1_cons_lock_0, 1)
            aie.next_bd(block[32])
            
        with block[32]:
            lock_acquire(w_p1_prod_lock, 1)
            aie.dma_bd(w_buffer_1, offset = 2 * w_block_size, len = 2 * w_block_size)
            lock_release(w_p1_cons_lock_0, 1)
            aie.next_bd(block[31])

        with block[33]:
            aie.EndOp()
            

def mvm_column(CTs: List[tile_with_location], MT: tile_with_location, IT: tile_with_location, X_IN_TILE: tile_with_location, X_IN_PORT: int, ctx: mlir_context):

    memtile_assemble(MT, ctx)

    connect_dma(IT, MT, 0, 4)
    connect_dma(IT, MT, 1, 5)
    for i, proj_core in enumerate(CTs):
        proj_tile(
            proj_core,
            ctx
        ) 
        connect_dma(X_IN_TILE, proj_core, X_IN_PORT, 0)
        connect_dma(MT, proj_core, i, 1)
        if i == 0:
            keep_pkt_header = True
        else:
            keep_pkt_header = False
        connect_dma(proj_core, MT, 0, i, 1, keep_pkt_header)
        connect_dma(proj_core, MT, 0, i, 4, keep_pkt_header)
        connect_dma(proj_core, MT, 0, i, 8, keep_pkt_header) 
        connect_dma(proj_core, MT, 0, i, 5, keep_pkt_header) 

