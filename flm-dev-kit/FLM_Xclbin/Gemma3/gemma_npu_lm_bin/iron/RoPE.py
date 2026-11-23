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

#                  ┌────────────────┐                        
# Input: Q, K, V   │                │  To MQ To MT: Q        
#   ───────────────▶ 0    ROPE     1├───────────▶            
#                  │        0       │                        
#                  └────────┬───────┘                        
#                           │                                
#                           │                                
#                           ▼ To DDR: K, V                   
# 
# Q: (D, 1)
# K, V: (D / 4, 1)

def RoPE(tile: tile_with_location, ctx: mlir_context):
    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    qkv_ty = ctx.data_tys["qkv_ty"]
    q_ty = ctx.data_tys["q_ty"]
    k_ty = ctx.data_tys["k_ty"]
    v_ty = ctx.data_tys["v_ty"]
    rope_ty = ctx.data_tys["rope_ty"]

    rope = ctx.funcs["rope"]


    qkv_prod_lock = aie.lock(ct, lock_id = 0, init = 1)
    qkv_cons_lock = aie.lock(ct, lock_id = 1, init = 0)

    q_prod_lock = aie.lock(ct, lock_id = 2, init = 1)
    q_cons_lock = aie.lock(ct, lock_id = 3, init = 0)

    k_prod_lock = aie.lock(ct, lock_id = 4, init = 2)
    k_cons_lock = aie.lock(ct, lock_id = 5, init = 0)

    v_prod_lock = aie.lock(ct, lock_id = 6, init = 2)
    v_cons_lock = aie.lock(ct, lock_id = 7, init = 0)

    rope_prod_lock = aie.lock(ct, lock_id = 8, init = 1)
    rope_cons_lock = aie.lock(ct, lock_id = 9, init = 0)

    qkv_buffer = aie.buffer(ct, qkv_ty, name = f"qkv_buffer_{ct_row}_{ct_col}")

    q_buffer = aie.buffer(ct, q_ty, name = f"q_buffer_{ct_row}_{ct_col}")
    k_buffer = aie.buffer(ct, k_ty, name = f"k_buffer_{ct_row}_{ct_col}")
    v_buffer = aie.buffer(ct, v_ty, name = f"v_buffer_{ct_row}_{ct_col}")
    
    DV = ctx.parameters["DV"]

    rope_buffer = aie.buffer(ct, rope_ty, name = f"rope_buffer_{ct_row}_{ct_col}")

    @aie.core(ct, link_with = "rope.o", stack_size = 1024 * 8)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            rope(q_buffer, k_buffer, v_buffer, qkv_buffer, rope_buffer)

    @aie.mem(ct)
    def mem_dma_body(block):
        s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[2])

        with block[1]:
            lock_acquire(qkv_prod_lock, 1)
            aie.dma_bd(qkv_buffer)
            lock_release(qkv_cons_lock, 1)
            aie.next_bd(block[1])

        with block[2]:
            s1 = aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[3], chain = block[4])

        with block[3]:
            lock_acquire(q_cons_lock, 1)
            aie.dma_bd(q_buffer)
            lock_release(q_prod_lock, 1)
            aie.next_bd(block[3])

        with block[4]:
            s2 = aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[5], chain = block[9])

        with block[5]:
            lock_acquire(k_cons_lock, 1)
            aie.dma_bd_packet(0, 14)
            aie.dma_bd(k_buffer, offset = 0, len = DV // 2)
            lock_release(k_prod_lock, 1)
            aie.next_bd(block[6])

        with block[6]:
            lock_acquire(v_cons_lock, 1)
            aie.dma_bd_packet(0, 15)
            aie.dma_bd(v_buffer, offset = 0, len = DV // 2)
            lock_release(v_prod_lock, 1)
            aie.next_bd(block[7])

        with block[7]:
            lock_acquire(k_cons_lock, 1)
            aie.dma_bd_packet(0, 14)
            aie.dma_bd(k_buffer, offset = DV // 2, len = DV // 2)
            lock_release(k_prod_lock, 1)
            aie.next_bd(block[8])

        with block[8]:
            lock_acquire(v_cons_lock, 1)
            aie.dma_bd_packet(0, 15)
            aie.dma_bd(v_buffer, offset = DV // 2, len = DV // 2)
            lock_release(v_prod_lock, 1)
            aie.next_bd(block[5])

        with block[9]:
            s3 = aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[10], chain = block[11])

        with block[10]:
            lock_acquire(rope_prod_lock, 1)
            aie.dma_bd(rope_buffer)
            lock_release(rope_cons_lock, 1)
            aie.next_bd(block[10])

        with block[11]:
            aie.EndOp()