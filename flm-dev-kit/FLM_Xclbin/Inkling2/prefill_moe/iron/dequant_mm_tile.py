# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import numpy as np
from aie.ir import IntegerAttr, IntegerType
import sys

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
try:  # mlir-aie as pinned by CI
    from aie.helpers.dialects.ext.scf import _for as range_
except ImportError:  # newer mlir-aie moved it
    from aie.helpers.dialects.scf import _for as range_
from typing import List
from utils import core_link_kwargs, dma_bd_compat, mlir_context, tile_with_location, connect_dma, connect_port, lock_acquire, lock_release, ENABLE_CHESSCC

# x arrives as bfp16 tiles packed by the host (FLM_DLL inkling2_bfp16.hpp), in the order the cores use them:
# per core row [256 k slice][m / 16 groups][4608 B], a 16-row group of one slice being [k8 32][A0, A1] 72 B tiles.
# The memtile keeps a block's x (m rows x Din) for all its passes: the shim sends it once, the memtile sends it
# once per pass. The buffer is two halves with a lock pair each, so the next block refills half a while the last
# pass still reads half b. The host sets the passes P (the S2MM takes P credits of a half and gives P, the MM2S
# one per send), the halves' lengths and m (inkling2_npu_mm.hpp rearm); built for P = 1, m = 128, Din = 2304.
# Buffer sizes stay in bf16 elements (2 B). A core receives an m16 group's slice (4608 B) in one DMA buffer and
# multiplies it in place.
X_SLICE_ELEMS = 2304                 # 4608 B
X_RES_ELEMS = 128 * 2304 * 9 // 16   # 331776 B: m x Din <= 128 x 2304
X_HALF_ELEMS = X_RES_ELEMS // 2
# y: one region per core of the column (m <= 128 rows x 48 n)
Y_REGION_ELEMS = 128 * 48
# one 32 x 256 Q4_K block in u32
QW_BLOCK_U32 = 1184


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

    (x_buffer_e_0, x_buffer_e_1, qw_buffer_0, qw_buffer_1, y_buffer_e_0, y_buffer_e_1, y_acc_buffer_e, w_buffer_s0) = \
        _pinned_even_buffers(mctx, ct_e, ct_e_row, ct_e_col, x_ty, qw_ty, y_ty, y_acc_ty, w_ty)
    K_div_k_buffer_e = mctx.buffers[f"K_div_k_{ct_e_row}_{ct_e_col}"]
    rtp_activate_e = mctx.buffers[f"rtp_act_{ct_e_row}_{ct_e_col}"]

    w_buffer_s1, x_buffer_o_0, x_buffer_o_1, y_buffer_o_0, y_buffer_o_1, y_acc_buffer_o = \
        _pinned_odd_buffers(mctx, ct_o, ct_o_row, ct_o_col, w_ty, x_ty, y_ty, y_acc_ty)
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
    
    @aie.core(ct_e, stack_size = 1024 * 2, **core_link_kwargs("dequant_mm_e.o"))
    def core_body():
        for _ in range_(0xFFFFFFFF):
            dequant_mm_e(y_buffer_e_0, y_buffer_e_1, x_buffer_e_0, x_buffer_e_1, qw_buffer_0, qw_buffer_1, w_buffer_s0, w_buffer_s1, y_acc_buffer_e, K_div_k_buffer_e, rtp_activate_e)
    
    @aie.mem(ct_e)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

        with block[1]:
            lock_acquire(x_prod_lock, 1)
            dma_bd_compat(x_buffer_e_0, offset = 0, len = X_SLICE_ELEMS)
            lock_release(x_cons_lock, 1)
            aie.next_bd(block[2])
        
        with block[2]:
            lock_acquire(x_prod_lock, 1)
            dma_bd_compat(x_buffer_e_1, offset = 0, len = X_SLICE_ELEMS)
            lock_release(x_cons_lock, 1)
            aie.next_bd(block[1])
        
        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])

        with block[4]:
            lock_acquire(qw_prod_lock, 2)
            dma_bd_compat(qw_buffer_0)
            lock_release(qw_cons_lock, 2)
            aie.next_bd(block[5])
            
        with block[5]:
            lock_acquire(qw_prod_lock, 2)
            dma_bd_compat(qw_buffer_1)
            lock_release(qw_cons_lock, 2)
            aie.next_bd(block[4])
            
        with block[6]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[7], chain=block[9])

        with block[7]:
            lock_acquire(y_cons_lock, 1)
            dma_bd_compat(y_buffer_e_0)
            lock_release(y_prod_lock, 1)
            aie.next_bd(block[8])

        with block[8]:
            lock_acquire(y_cons_lock, 1)
            dma_bd_compat(y_buffer_e_1)
            lock_release(y_prod_lock, 1)
            aie.next_bd(block[7])

        with block[9]:
            aie.EndOp()

    @aie.core(ct_o, stack_size = 1024 * 2, **core_link_kwargs("dequant_mm_o.o"))
    def core_body():
        for _ in range_(0xFFFFFFFF):
            dequant_mm_o(y_buffer_o_0, y_buffer_o_1, x_buffer_o_0, x_buffer_o_1, qw_buffer_0, qw_buffer_1, w_buffer_s0, w_buffer_s1, y_acc_buffer_o, K_div_k_buffer_o, rtp_activate_o)
    
    @aie.mem(ct_o)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

        with block[1]:
            lock_acquire(x_prod_lock_o, 1)
            dma_bd_compat(x_buffer_o_0, offset = 0, len = X_SLICE_ELEMS)
            lock_release(x_cons_lock_o, 1)
            aie.next_bd(block[2])
        
        with block[2]:
            lock_acquire(x_prod_lock_o, 1)
            dma_bd_compat(x_buffer_o_1, offset = 0, len = X_SLICE_ELEMS)
            lock_release(x_cons_lock_o, 1)
            aie.next_bd(block[1])
        
        with block[3]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

        with block[4]:
            lock_acquire(y_cons_lock_o, 1)
            dma_bd_compat(y_buffer_o_0)
            lock_release(y_prod_lock_o, 1)
            aie.next_bd(block[5])

        with block[5]:
            lock_acquire(y_cons_lock_o, 1)
            dma_bd_compat(y_buffer_o_1)
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

    # a core's y tiles [m8 row][n8 tile (6)][8][8] -> its region [m][n]
    y_in_dims = [(m // 8, 8 * n), (n // 8, 8), (8, n), (8, 1)]
    y_out_dims = [(2, Y_REGION_ELEMS), (m, n), (2, 2 * Y_REGION_ELEMS), (n, 1)]

    x_res = aie.buffer(tile = mt, datatype = x_mem_ty, name = f"x_{mt_row}_{mt_col}")
    y_buffer_0 = aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_0_{mt_row}_{mt_col}")
    y_buffer_1 = aie.buffer(tile = mt, datatype = y_mem_ty, name = f"y_1_{mt_row}_{mt_col}")
    qw_buffer_0 = aie.buffer(tile = mt, datatype = qw_mem_ty, name = f"qw_0_{mt_row}_{mt_col}")
    qw_buffer_1 = aie.buffer(tile = mt, datatype = qw_mem_ty, name = f"qw_1_{mt_row}_{mt_col}")

    x_prod_a = aie.lock(mt, lock_id = 0, init = 1)
    x_cons_a = aie.lock(mt, lock_id = 1, init = 0)
    x_prod_b = aie.lock(mt, lock_id = 2, init = 1)
    x_cons_b = aie.lock(mt, lock_id = 3, init = 0)
    
    # y join: the four cores' tiles land concurrently (each S2MM channel takes one of the 4 producer
    # credits of a buffer and adds one consumer credit); the output DMA waits for all 4 and gives the 4 back
    y_prod_lock_0 = aie.lock(mt, lock_id = 5, init = 4)
    y_prod_lock_1 = aie.lock(mt, lock_id = 6, init = 4)
    y_cons_lock_0 = aie.lock(mt, lock_id = 7, init = 0)
    y_cons_lock_1 = aie.lock(mt, lock_id = 8, init = 0)

    qw_prod_lock = aie.lock(mt, lock_id = 10, init = 2)
    qw_cons_a = aie.lock(mt, lock_id = 11, init = 0)
    qw_cons_b = aie.lock(mt, lock_id = 12, init = 0)

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])
        
        with block[1]:
            lock_acquire(x_prod_a, 1)
            dma_bd_compat(x_res, offset = 0, len = X_HALF_ELEMS)
            lock_release(x_cons_a, 1)
            aie.next_bd(block[2])

        with block[2]:
            lock_acquire(x_prod_b, 1)
            dma_bd_compat(x_res, offset = X_HALF_ELEMS, len = X_HALF_ELEMS)
            lock_release(x_cons_b, 1)
            aie.next_bd(block[1])

        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[12])

        # one send per half and pass
        with block[4]:
            lock_acquire(x_cons_a, 1)
            dma_bd_compat(x_res, offset = 0, len = X_HALF_ELEMS)
            lock_release(x_prod_a, 1)
            aie.next_bd(block[5])

        with block[5]:
            lock_acquire(x_cons_b, 1)
            dma_bd_compat(x_res, offset = X_HALF_ELEMS, len = X_HALF_ELEMS)
            lock_release(x_prod_b, 1)
            aie.next_bd(block[4])

        with block[12]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[13], chain=block[15])

        with block[13]:
            lock_acquire(y_prod_lock_0, 1)
            dma_bd_compat(y_buffer_0, offset = 0 * Y_REGION_ELEMS, len = m * n, dimensions = y_in_dims)
            lock_release(y_cons_lock_0, 1)
            aie.next_bd(block[14])

        with block[14]:
            lock_acquire(y_prod_lock_1, 1)
            dma_bd_compat(y_buffer_1, offset = 0 * Y_REGION_ELEMS, len = m * n, dimensions = y_in_dims)
            lock_release(y_cons_lock_1, 1)
            aie.next_bd(block[13])

        with block[15]:
            s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[16], chain=block[18])

        with block[16]:
            lock_acquire(y_prod_lock_0, 1)
            dma_bd_compat(y_buffer_0, offset = 1 * Y_REGION_ELEMS, len = m * n, dimensions = y_in_dims)
            lock_release(y_cons_lock_0, 1)
            aie.next_bd(block[17])

        with block[17]:
            lock_acquire(y_prod_lock_1, 1)
            dma_bd_compat(y_buffer_1, offset = 1 * Y_REGION_ELEMS, len = m * n, dimensions = y_in_dims)
            lock_release(y_cons_lock_1, 1)
            aie.next_bd(block[16])

        with block[18]:
            s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest=block[19], chain=block[21])

        with block[19]:
            lock_acquire(y_prod_lock_0, 1)
            dma_bd_compat(y_buffer_0, offset = 2 * Y_REGION_ELEMS, len = m * n, dimensions = y_in_dims)
            lock_release(y_cons_lock_0, 1)
            aie.next_bd(block[20])

        with block[20]:
            lock_acquire(y_prod_lock_1, 1)
            dma_bd_compat(y_buffer_1, offset = 2 * Y_REGION_ELEMS, len = m * n, dimensions = y_in_dims)
            lock_release(y_cons_lock_1, 1)
            aie.next_bd(block[19])

        with block[21]:
            s5 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 4, dest=block[22], chain=block[24])

        with block[22]:
            lock_acquire(y_prod_lock_0, 1)
            dma_bd_compat(y_buffer_0, offset = 3 * Y_REGION_ELEMS, len = m * n, dimensions = y_in_dims)
            lock_release(y_cons_lock_0, 1)
            aie.next_bd(block[23])

        with block[23]:
            lock_acquire(y_prod_lock_1, 1)
            dma_bd_compat(y_buffer_1, offset = 3 * Y_REGION_ELEMS, len = m * n, dimensions = y_in_dims)
            lock_release(y_cons_lock_1, 1)
            aie.next_bd(block[22])

        # regions [A even, A odd, B even, B odd] -> [2m rows][2n]: rows 0..m-1 from the even cores, m..2m-1 from
        # the odd ones, n 0..47 from pair A and 48..95 from pair B
        with block[24]:
            s8 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[25], chain=block[27])

        with block[25]:
            lock_acquire(y_cons_lock_0, 4)
            dma_bd_compat(y_buffer_0, offset = 0, len = 4 * m * n, dimensions = y_out_dims)
            lock_release(y_prod_lock_0, 4)
            aie.next_bd(block[26])

        with block[26]:
            lock_acquire(y_cons_lock_1, 4)
            dma_bd_compat(y_buffer_1, offset = 0, len = 4 * m * n, dimensions = y_out_dims)
            lock_release(y_prod_lock_1, 4)
            aie.next_bd(block[25])

        with block[27]:
            s9 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 5, dest=block[28], chain=block[30])
        
        with block[28]:
            lock_acquire(qw_prod_lock, 1)
            dma_bd_compat(qw_buffer_0)
            lock_release(qw_cons_a, 1)
            aie.next_bd(block[29])

        with block[29]:
            lock_acquire(qw_prod_lock, 1)
            dma_bd_compat(qw_buffer_1)
            lock_release(qw_cons_a, 1)
            aie.next_bd(block[28])

        # pair A: blocks 0, 1 of the triple
        with block[30]:
            s10 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[31], chain=block[33])

        with block[31]:
            lock_acquire(qw_cons_a, 1)
            dma_bd_compat(qw_buffer_0, offset = 0, len = 2 * QW_BLOCK_U32)
            lock_release(qw_cons_b, 1)
            aie.next_bd(block[32])

        with block[32]:
            lock_acquire(qw_cons_a, 1)
            dma_bd_compat(qw_buffer_1, offset = 0, len = 2 * QW_BLOCK_U32)
            lock_release(qw_cons_b, 1)
            aie.next_bd(block[31])

        # pair B: blocks 1, 2
        with block[33]:
            s11 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 3, dest=block[34], chain=block[36])

        with block[34]:
            lock_acquire(qw_cons_b, 1)
            dma_bd_compat(qw_buffer_0, offset = QW_BLOCK_U32, len = 2 * QW_BLOCK_U32)
            lock_release(qw_prod_lock, 1)
            aie.next_bd(block[35])

        with block[35]:
            lock_acquire(qw_cons_b, 1)
            dma_bd_compat(qw_buffer_1, offset = QW_BLOCK_U32, len = 2 * QW_BLOCK_U32)
            lock_release(qw_prod_lock, 1)
            aie.next_bd(block[34])

        with block[36]:
            aie.EndOp()


def _pin_rtp(mctx, row, col, kdiv, act):
    i32 = IntegerType.get_signless(32)
    for name, address in (("K_div_k", kdiv), ("rtp_act", act)):
        mctx.buffers[f"{name}_{row}_{col}"].owner.attributes["address"] = IntegerAttr.get(i32, address)


def _pinned_even_buffers(mctx, ct, row, col, x_ty, qw_ty, y_ty, y_acc_ty, w_ty):
    """Even tile of a pair: x ping/pong, the two qw buffers the pair shares, y, y_acc,
    and n slice 0 of the shared bfp16 w tiles.  K_div_k / rtp_act stay at the
    addresses FLM_DLL writes (54784 / 54848); the 2 KB stack is the core's stack_size."""
    def pinned(ty, name, address):
        return aie.buffer(tile = ct, datatype = ty, name = f"{name}_{row}_{col}", address = address)
    x_0 = pinned(x_ty, "x_0", 2048)                   # 4608 B each -> 11264
    x_1 = pinned(x_ty, "x_1", 6656)
    y_acc = pinned(y_acc_ty, "y_acc", 11264)          # 32 KB -> 44032
    qw_0 = pinned(qw_ty, "w_0", 44032)                # 4736 B each -> 53504
    qw_1 = pinned(qw_ty, "w_1", 48768)
    y_0 = pinned(y_ty, "y_0", 53504)                  # 512 B each -> 54528
    y_1 = pinned(y_ty, "y_1", 54016)
    _pin_rtp(mctx, row, col, 54784, 54848)
    w_s0 = pinned(w_ty, "ws0", 54912)                 # 9216 B -> 64128
    return x_0, x_1, qw_0, qw_1, y_0, y_1, y_acc, w_s0


def _pinned_odd_buffers(mctx, ct_o, ct_o_row, ct_o_col, w_ty, x_ty, y_ty, y_acc_ty):
    """Odd tile: x, y_acc and n slice 1 of the shared bfp16 w tiles.  RTPs stay where FLM_DLL
    writes them; the y buffers sit behind them."""
    def pinned(ty, name, address):
        return aie.buffer(tile = ct_o, datatype = ty, name = f"{name}_{ct_o_row}_{ct_o_col}", address = address)
    x_0 = pinned(x_ty, "x_0", 2048)
    x_1 = pinned(x_ty, "x_1", 6656)
    y_acc = pinned(y_acc_ty, "y_acc", 11264)          # -> 44032
    w = pinned(w_ty, "ws1", 44032)                    # 9216 B -> 53248
    _pin_rtp(mctx, ct_o_row, ct_o_col, 61696, 61760)
    y_0 = pinned(y_ty, "y_0", 61824)
    y_1 = pinned(y_ty, "y_1", 62336)
    return w, x_0, x_1, y_0, y_1, y_acc
