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

    q_dtype = np.dtype[bfloat16]
    y_acc_dtype = np.dtype[np.float32]
    x_dtype = np.dtype[bfloat16]
    w_dtype = np.dtype[bfloat16]
    y_dtype = np.dtype[bfloat16]

    # mm: 4 core rows split M (m per row), 8 core columns split N (n per column)
    m = 128
    k = 256
    n = 48
    m_slice = 16
    k_slice = 64
    n_slice = 48
    y_slice = 64

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(4 + 2, 8)

            mctx = mlir_context()
            x_ty = mctx.data_ty('x_ty', (m_slice, k_slice), x_dtype)
            x_mem_ty = mctx.data_ty('x_mem_ty', (m, k), x_dtype)
            w_ty = mctx.data_ty('w_ty', (n_slice, k_slice), w_dtype)
            w_mem_ty = mctx.data_ty('w_mem_ty', (n, k), w_dtype)
            mm_y_ty = mctx.data_ty('mm_y_ty', (y_slice,), y_dtype)
            y_acc_ty = mctx.data_ty('y_acc_ty', (m * n, ), y_acc_dtype)
            b_ty = mctx.data_ty('b_ty', (n, ), y_acc_dtype)
            y_mem_ty = mctx.data_ty('y_mem_ty', (m, n * 4), y_dtype)
            K_div_k_ty = mctx.data_ty('K_div_k_ty', (16, ), np.dtype[np.int32])

            mctx.parameter('m', m)
            mctx.parameter('k', k)
            mctx.parameter('n', n)
            mctx.parameter('m_slice', m_slice)
            mctx.parameter('k_slice', k_slice)
            mctx.parameter('n_slice', n_slice)

            mctx.func("mm", [mctx.x_ty, mctx.x_ty, mctx.w_ty, mctx.w_ty, mctx.mm_y_ty, mctx.mm_y_ty, mctx.y_acc_ty, mctx.b_ty, mctx.K_div_k_ty])

            mm_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]

            for i in range(total_rows):
                for j in range(total_cols):
                    mctx.buffer(f"rtp_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp = True)

            for j in range(total_cols):
                for i in range(total_rows):
                    mm_cttile(mm_core[i][j], mctx)
                mm_memtile(MT[j], mctx)

            # Memtile / shim column c: w for core column c (broadcast down its 4 rows), x and y for core row
            # c % 4, half c // 4 (cores 4 * (c // 4) .. +3 of that row).
            for c in range(total_cols):
                i = c % 4
                g = c // 4
                connect_dma(IT[c], MT[c], 1, 1)
                for r in range(total_rows):
                    connect_dma(MT[c], mm_core[r][c], 1, 1)
                if g == 0:
                    # x of core row i is the same for both column groups: one shim read, multicast to both memtiles
                    connect_dma(IT[c], MT[c], 0, 0)
                    connect_dma(IT[c], MT[c + 4], 0, 0)
                for t in range(4):
                    connect_dma(MT[c], mm_core[i][4 * g + t], 0, 0)
                    connect_dma(mm_core[i][4 * g + t], MT[c], 0, 2 + t)
                connect_dma(MT[c], IT[c], 2, 0)

        print(ctx.module)

if __name__ == "__main__":
    flow_arch()
