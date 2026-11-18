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
from mm_tile import mm_cttile, mm_memtile

def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    M = 4096
    K = 4096
    N = 4096

    m = 64
    k = 512
    n = 128
    m_slice = 64
    k_slice = 64
    n_slice = 16
    y_slice = 128

    x_dtype = np.dtype[bfloat16]
    w_dtype = np.dtype[bfloat16]
    y_dtype = np.dtype[bfloat16]
    y_acc_dtype = np.dtype[np.float32]

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            x_ty = mctx.data_ty('x_ty', (m_slice, k_slice), x_dtype)
            x_mem_ty = mctx.data_ty('x_mem_ty', (m, k), x_dtype)
            w_ty = mctx.data_ty('w_ty', (n_slice, k_slice), w_dtype) 
            w_mem_ty = mctx.data_ty('w_mem_ty', (n, k), w_dtype)
            y_ty = mctx.data_ty('y_ty', (y_slice,), y_dtype)
            y_acc_ty = mctx.data_ty('y_acc_ty', (m * n, ), y_acc_dtype)
            y_mem_ty = mctx.data_ty('y_mem_ty', (m, n * 4), y_dtype)
            K_div_k_ty = mctx.data_ty('K_div_k_ty', (16, ), np.dtype[np.int32])

            mctx.parameter('m', m)
            mctx.parameter('k', k)
            mctx.parameter('n', n)
            mctx.parameter('m_slice', m_slice)
            mctx.parameter('k_slice', k_slice)
            mctx.parameter('n_slice', n_slice)

            mctx.func("mm", [mctx.x_ty, mctx.x_ty, mctx.w_ty, mctx.w_ty, mctx.y_ty, mctx.y_ty, mctx.y_acc_ty, mctx.K_div_k_ty, mctx.K_div_k_ty])

            for j in range(total_cols):
                memref.global_(f"w_in_{j}", T.memref(N * K, T.bf16()), sym_visibility = "public")
                memref.global_(f"y_out_{j}", T.memref(M * N // total_rows, T.bf16()), sym_visibility = "public")
                memref.global_(f"x_in_{j}", T.memref(M * K // total_rows, T.bf16()), sym_visibility = "public")

            mm_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]
            K_div_k_buffers = []
            rtp_buffers = []
            for i in range(total_rows):
                for j in range(total_cols):
                    K_div_k_buffers.append(mctx.buffer(f"K_div_k_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp = True))
                    rtp_buffers.append(mctx.buffer(f"rtp_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp = True))
                    
            for j in range(total_cols):
                for i in range(total_rows):
                    mm_cttile(mm_core[i][j], mctx)
                mm_memtile(MT[j], mctx)
  
            # w_in
            for j in range(total_cols):
                connect_dma(IT[j], MT[j], 1, 1)
                for i in range(total_rows):
                    connect_dma(MT[j], mm_core[i][j], 1, 1)

            # x_in
            for i in range(total_rows):
                connect_dma(IT[i], MT[i], 0, 0)
                connect_dma(IT[i + 4], MT[i + 4], 0, 0)
                for j in range(4):
                    connect_dma(MT[i], mm_core[i][j], 0, 0)
                    connect_dma(MT[i + 4], mm_core[i][j + 4], 0, 0)

            # y_out 
            for i in range(total_rows):
                connect_dma(mm_core[i][0], MT[i], 0, 2)
                connect_dma(mm_core[i][1], MT[i], 0, 3)
                connect_dma(mm_core[i][2], MT[i], 0, 4)
                connect_dma(mm_core[i][3], MT[i], 0, 5)
                connect_dma(mm_core[i][4], MT[i + 4], 0, 2)
                connect_dma(mm_core[i][5], MT[i + 4], 0, 3)
                connect_dma(mm_core[i][6], MT[i + 4], 0, 4)
                connect_dma(mm_core[i][7], MT[i + 4], 0, 5)

            for j in range(total_cols):
                connect_dma(MT[j], IT[j], 2, 0)

        print(ctx.module)


if __name__ == "__main__":
    flow_arch()