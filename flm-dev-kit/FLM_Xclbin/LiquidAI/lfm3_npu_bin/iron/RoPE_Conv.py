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

def RoPE_Conv(rope_tile: tile_with_location, conv_tile: tile_with_location, ctx: mlir_context):
    rope_ct = rope_tile.tile
    conv_ct = conv_tile.tile
    rope_ct_row = rope_tile.row
    rope_ct_col = rope_tile.col
    conv_ct_row = conv_tile.row
    conv_ct_col = conv_tile.col
    ct_col = rope_tile.col

    qkv_ty = ctx.data_tys["qkv_ty"]
    q_pkt_ty = ctx.data_tys["q_pkt_ty"]
    q_ty = ctx.data_tys["q_ty"]
    k_ty = ctx.data_tys["k_ty"]
    v_ty = ctx.data_tys["v_ty"]
    rope_ty = ctx.data_tys["rope_ty"]
    conv_x_ty = ctx.data_tys["conv_x_ty"]
    conv_bcx_ty = ctx.data_tys["conv_bcx_ty"]
    past_bx_ty = ctx.data_tys["past_bx_ty"]
    rope = ctx.funcs["rope"]
    DV = ctx.parameters["DV"]
    D = ctx.parameters["D"]
    repeats = D // 32 // 16

    qkv_prod_lock = aie.lock(rope_ct, lock_id = 0, init = 1)
    qkv_cons_lock = aie.lock(rope_ct, lock_id = 1, init = 0)

    q_prod_lock = aie.lock(rope_ct, lock_id = 2, init = 1)
    q_cons_lock = aie.lock(rope_ct, lock_id = 3, init = 0)

    k_prod_lock = aie.lock(rope_ct, lock_id = 4, init = 2)
    k_cons_lock = aie.lock(rope_ct, lock_id = 5, init = 0)

    v_prod_lock = aie.lock(rope_ct, lock_id = 6, init = 2)
    v_cons_lock = aie.lock(rope_ct, lock_id = 7, init = 0)

    rope_prod_lock = aie.lock(rope_ct, lock_id = 8, init = 1)
    rope_cons_lock = aie.lock(rope_ct, lock_id = 9, init = 0)

    conv_start_lock = aie.lock(conv_ct, lock_id = 0, init = 0)
    conv_end_lock = aie.lock(conv_ct, lock_id = 1, init = 0)

    bx_prod_lock = aie.lock(conv_ct, lock_id = 2, init = 1)
    bx_cons_lock = aie.lock(conv_ct, lock_id = 3, init = 0)

    y_prod_lock = aie.lock(conv_ct, lock_id = 4, init = repeats)
    y_cons_lock = aie.lock(conv_ct, lock_id = 5, init = 0)

    past_bx_prod_lock = aie.lock(conv_ct, lock_id = 6, init = 1)
    past_bx_cons_lock = aie.lock(conv_ct, lock_id = 7, init = 0)

    qkv_buffer = aie.buffer(rope_ct, qkv_ty, name = f"qkv_buffer_{rope_ct_row}_{rope_ct_col}")

    q_buffer = aie.buffer(rope_ct, q_ty, name = f"q_buffer_{rope_ct_row}_{rope_ct_col}")
    k_buffer = aie.buffer(rope_ct, k_ty, name = f"k_buffer_{rope_ct_row}_{rope_ct_col}")
    v_buffer = aie.buffer(rope_ct, v_ty, name = f"v_buffer_{rope_ct_row}_{rope_ct_col}")



    conv_bcx_buffer = aie.buffer(conv_ct, conv_bcx_ty, name = f"conv_bcx_buffer_{conv_ct_row}_{conv_ct_col}")
    bx_buffer = aie.buffer(conv_ct, conv_x_ty, name = f"bx_buffer_{conv_ct_row}_{conv_ct_col}")
    past_bx_buffer = aie.buffer(conv_ct, past_bx_ty, name = f"past_bx_buffer_{conv_ct_row}_{conv_ct_col}")
    y_buffer = aie.buffer(conv_ct, conv_x_ty, name = f"y_buffer_{conv_ct_row}_{conv_ct_col}")
    L_buffer = ctx.buffers["IS_ATTN_ROPE_BUFFERS"]

    rope_buffer = aie.buffer(rope_ct, rope_ty, name = f"rope_buffer_{rope_ct_row}_{rope_ct_col}")

    @aie.core(rope_ct, link_with = "rope.o")
    def core_body():
        for _ in range_(0xFFFFFFFF):
            rope(q_buffer, k_buffer, v_buffer, qkv_buffer, rope_buffer, conv_bcx_buffer, L_buffer)

    conv = ctx.funcs["conv"]
    @aie.core(conv_ct, link_with = "conv.o")
    def conv_core_body():
        for _ in range_(0xFFFFFFFF):
            conv(bx_buffer, y_buffer, past_bx_buffer, conv_bcx_buffer, rope_buffer)

    @aie.mem(rope_ct)
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

    @aie.mem(conv_ct)
    def mem_dma_body_conv(block):
        s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[2])

        with block[1]:
            lock_acquire(past_bx_prod_lock, 1)
            aie.dma_bd(past_bx_buffer)
            lock_release(past_bx_cons_lock, 1)
            aie.next_bd(block[1])

        with block[2]:
            s1 = aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[3], chain = block[4])

        with block[3]:
            lock_acquire(y_cons_lock, 1)
            aie.dma_bd_packet(0, 0)
            aie.dma_bd(y_buffer)
            lock_release(y_prod_lock, 1)
            aie.next_bd(block[3])

        with block[4]:
            s2 = aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[5], chain = block[6])

        with block[5]:
            lock_acquire(bx_cons_lock, 1)
            aie.dma_bd(bx_buffer)
            lock_release(bx_prod_lock, 1)
            aie.next_bd(block[5])

        with block[6]:
            aie.EndOp()