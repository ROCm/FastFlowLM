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
from attn_tile import attn_cttile0, attn_memtile0, attn_cttile1, attn_memtile1 

def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 4
    total_rows = 4
    d = 64
    lq = 64
    lk = 64

    q_dtype = np.dtype[bfloat16]
    k_dtype = np.dtype[bfloat16]
    v_dtype = np.dtype[bfloat16]
    o_dtype = np.dtype[bfloat16]
    y_dtype = np.dtype[np.float32]

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(4 + 2, 8)

            mctx = mlir_context()
            in_ty = mctx.data_ty('in_ty', (lk, d), k_dtype) 
            q_ty_0 = mctx.data_ty('q_ty_0', (lq, d), q_dtype)
            q_mem_ty_0 = mctx.data_ty('q_mem_ty_0', (lq, d * 3), q_dtype)
            in_mem_ty_0 = mctx.data_ty('in_mem_ty_0', (2 * lk, d), k_dtype)

            q_ty_1 = mctx.data_ty('q_ty_1', (lq, d * 2), q_dtype)
            q_mem_ty_1 = mctx.data_ty('q_mem_ty_1', (lq, d * 4), q_dtype)
            in_mem_ty_1 = mctx.data_ty('in_mem_ty_1', (4 * lk, d), k_dtype)

            o_ty = mctx.data_ty('o_ty', (64, ), o_dtype)
            o_mem_ty_0 = mctx.data_ty('o_mem_ty_0', (lq, d * 3), o_dtype)
            o_mem_ty_1 = mctx.data_ty('o_mem_ty_1', (lq, d * 4), o_dtype)
            s_ty = mctx.data_ty('s_ty', (lq, lk), v_dtype)
            y_ty = mctx.data_ty('y_ty', (lq, d), y_dtype)
            L_ty = mctx.data_ty('L_ty', (16,), np.dtype[np.int32])

            mctx.parameter('lq', lq)
            mctx.parameter('lk', lk)
            mctx.parameter('d', d)
            mctx.func("attn0", [o_ty, o_ty, q_ty_0, in_ty, in_ty, s_ty, y_ty, L_ty])
            mctx.func("attn1", [o_ty, o_ty, q_ty_1, in_ty, in_ty, s_ty, y_ty, L_ty, T.i32()])

            attn_core_0 = [CT[i][6] for i in range(3)]
            attn_core_1 = [[CT[i][0], CT[i][1], CT[i][7]] for i in range(total_rows)]

            rtp_buffers = []
            for i in range(total_rows):
                for j in range(3):
                    rtp_buffers.append(mctx.buffer(f"rtp_{attn_core_1[i][j].row}_{attn_core_1[i][j].col}", attn_core_1[i][j], L_ty, use_write_rtp = True))

            for i in range(3):
                    rtp_buffers.append(mctx.buffer(f"rtp_{attn_core_0[i].row}_{attn_core_0[i].col}", attn_core_0[i], L_ty, use_write_rtp = True))

            for j in range(3):
                for i in range(total_rows):
                    attn_cttile1(attn_core_1[i][j], mctx)
            attn_memtile1(MT[0], mctx)
            attn_memtile1(MT[1], mctx)
            attn_memtile1(MT[7], mctx)
                
            for i in range(3):
                attn_cttile0(attn_core_0[i], mctx)
            attn_memtile0(MT[6], mctx)
        
            # q_in
            for j in range(2):
                connect_dma(IT[j], MT[j], 0, 0)
                connect_dma(MT[j], attn_core_1[0][j], 0, 0)
                connect_dma(MT[j], attn_core_1[1][j], 0, 0)
                connect_dma(MT[j], attn_core_1[2][j], 1, 0)
                connect_dma(MT[j], attn_core_1[3][j], 1, 0)

            connect_dma(IT[6], MT[6], 0, 0)
            for i in range(3):
                connect_dma(MT[6], attn_core_0[i], i, 0)

            connect_dma(IT[7], MT[7], 0, 0)
            connect_dma(MT[7], attn_core_1[0][2], 0, 0)
            connect_dma(MT[7], attn_core_1[1][2], 0, 0)
            connect_dma(MT[7], attn_core_1[2][2], 1, 0)
            connect_dma(MT[7], attn_core_1[3][2], 1, 0)

            # k_in & v_in
            connect_dma(IT[0], MT[0], 1, 1)
            connect_dma(MT[0], attn_core_1[0][0], 2, 1)
            connect_dma(MT[0], attn_core_1[1][0], 2, 1)
            connect_dma(MT[0], attn_core_1[2][0], 2, 1)
            connect_dma(MT[0], attn_core_1[3][0], 3, 1)
            connect_dma(IT[1], MT[1], 1, 1)
            connect_dma(MT[1], attn_core_1[0][1], 2, 1)
            connect_dma(MT[1], attn_core_1[1][1], 2, 1)
            connect_dma(MT[1], attn_core_1[2][1], 3, 1)
            connect_dma(MT[1], attn_core_1[3][1], 3, 1)

            connect_dma(IT[6], MT[6], 1, 1)
            for i in range(3):
                connect_dma(MT[6], attn_core_0[i], 3, 1)

            connect_dma(IT[7], MT[7], 1, 1)
            connect_dma(MT[7], attn_core_1[0][2], 2, 1)
            connect_dma(MT[7], attn_core_1[1][2], 3, 1)
            connect_dma(MT[7], attn_core_1[2][2], 3, 1)
            connect_dma(MT[7], attn_core_1[3][2], 3, 1)
                    
            # o_out
            for j in range(2):
                for i in range(total_rows):
                    connect_dma(attn_core_1[i][j], MT[j], 0, 2 + i)
                connect_dma(MT[j], IT[j], 5, 0)
 
            for i in range(3):
                connect_dma(attn_core_0[i], MT[6], 0, 2 + i)
            connect_dma(MT[6], IT[6], 4, 0)

            for i in range(total_rows):
                connect_dma(attn_core_1[i][2], MT[7], 0, 2 + i)
            connect_dma(MT[7], IT[7], 5, 0)

        print(ctx.module)

if __name__ == "__main__":
    flow_arch()
