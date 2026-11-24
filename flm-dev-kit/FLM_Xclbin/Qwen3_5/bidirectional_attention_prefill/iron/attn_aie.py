# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.extras import types as T
import aie.dialects.aie as aie
import aie.dialects.aiex as aiex
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils
from aie.utils.trace import MemTilePortEvent, MemTileEvent, PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from aie.dialects import memref

from utils import dma_mm2s_ch, dma_s2mm_ch, flow_mm2s_ch, flow_s2mm_ch, connect_dma, generate_tile_map, mlir_context
from utils import dma_port, connect_port
from attn_tile import attn_cttile, attn_memtile_even, attn_memtile_odd

def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4
    d = 256
    lq = 16
    lk = 16 
    lk_mt = 128
    lq_mt = 64

    lq_ct = 32

    q_dtype = np.dtype[bfloat16]
    k_dtype = np.dtype[bfloat16]
    v_dtype = np.dtype[bfloat16]
    o_dtype = np.dtype[bfloat16]
    y_dtype = np.dtype[np.float32]

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            q_ty = mctx.data_ty('q_ty', (lq_ct, d), q_dtype)
            in_ty = mctx.data_ty('in_ty', (lk, d), k_dtype) 
            q_mem_ty = mctx.data_ty('q_mem_ty', (lq_mt * 2, d), q_dtype)
            in_mem_ty = mctx.data_ty('in_mem_ty', (lk_mt * 2, d), k_dtype)


            o_ty = mctx.data_ty('o_ty', (64, ), o_dtype)
            o_mem_ty = mctx.data_ty('o_mem_ty', (lq * 4, d), o_dtype)
            s_ty = mctx.data_ty('s_ty', (lq, lk_mt), v_dtype)
            m_ty = mctx.data_ty('m_ty', (lq, lk), v_dtype)
            y_ty = mctx.data_ty('y_ty', (lq, d), y_dtype)
            L_ty = mctx.data_ty('L_ty', (8,), np.dtype[np.int32])

            mctx.parameter('lq', lq)
            mctx.parameter('lk', lk)
            mctx.parameter('lk_mt', lk_mt)
            mctx.parameter('lq_mt', lq_mt)
            mctx.parameter('lq_ct', lq_ct)
            mctx.parameter('d', d)
            mctx.func("attn", [o_ty, o_ty, q_ty, in_ty, in_ty, s_ty, m_ty, y_ty, L_ty, L_ty, L_ty, T.i32(), T.i32()])

            for j in range(total_cols):
                memref.global_(f"q_in_{j}", T.memref(lq_ct * 4 * d, T.bf16()), sym_visibility = "public")
                memref.global_(f"o_out_{j}", T.memref(lq_ct * 4 * d, T.bf16()), sym_visibility = "public")

            attn_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]

            L_begin_buffers = []
            L_chunks_buffers = []
            window_size_buffers = []
            for i in range(total_rows):
                for j in range(total_cols):
                    L_begin_buffers.append(mctx.buffer(f"L_begin_attn_core_{attn_core[i][j].row}_{attn_core[i][j].col}", attn_core[i][j], L_ty, use_write_rtp = True))
                    L_chunks_buffers.append(mctx.buffer(f"L_chunks_attn_core_{attn_core[i][j].row}_{attn_core[i][j].col}", attn_core[i][j], L_ty, use_write_rtp = True))
                    window_size_buffers.append(mctx.buffer(f"window_size_attn_core_{attn_core[i][j].row}_{attn_core[i][j].col}", attn_core[i][j], L_ty, use_write_rtp = True))

            for j in range(total_cols):
                for i in range(total_rows):
                    attn_cttile(attn_core[i][j], mctx)
            
            for j in range(total_cols):
                if j & 0x1 == 0:
                    attn_memtile_even(MT[j], mctx)
                else:
                    attn_memtile_odd(MT[j], mctx)
        
            # q_in
            for j in range(4):
                connect_dma(IT[j * 2], MT[j * 2], 0, 0)
                connect_dma(IT[j * 2], MT[j * 2], 1, 1)
                for i in range(4):
                    connect_dma(MT[j * 2], attn_core[i][j * 2], i, 0)
                    connect_dma(MT[j * 2], attn_core[i][j * 2 + 1], i, 0)

            # sharing k and v between 2 CUs
            # k_in & v_in
            connect_dma(IT[3], MT[3], 0, 0)
            connect_dma(IT[3], MT[3], 1, 1)
            for j in range(total_cols):
                for i in range(total_rows):
                    connect_dma(MT[3], attn_core[i][j], 0, 1)
    
            # o_out
            for j in range(4):
                connect_dma(attn_core[0][2 * j], MT[2 * j], 0, 2)
                connect_dma(attn_core[0][2 * j + 1], MT[2 * j], 0, 3)
                connect_dma(attn_core[1][2 * j], MT[2 * j], 0, 4)
                connect_dma(attn_core[1][2 * j + 1], MT[2 * j], 0, 5)

                connect_dma(attn_core[2][2 * j], MT[2 * j + 1], 0, 2)
                connect_dma(attn_core[2][2 * j + 1], MT[2 * j + 1], 0, 3)
                connect_dma(attn_core[3][2 * j], MT[2 * j + 1], 0, 4)
                connect_dma(attn_core[3][2 * j + 1], MT[2 * j + 1], 0, 5)
            for j in range(total_cols):
                connect_dma(MT[j], IT[j], 5, 0)

        print(ctx.module)

if __name__ == "__main__":
    flow_arch()
