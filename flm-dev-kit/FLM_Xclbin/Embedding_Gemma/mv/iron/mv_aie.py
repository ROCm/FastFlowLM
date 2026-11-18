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
from mv_tile import mv_tile, mv_memtile

def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    w_dtype = np.dtype[bfloat16]
    x_dtype = np.dtype[bfloat16]
    y_dtype = np.dtype[bfloat16]
    m_dtype = np.dtype[np.uint32]
    y_acc_dtype = np.dtype[np.float32]

    M = 3072
    D = 768
    m = 32
    d = 256

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            w_ty = mctx.data_ty('w_ty', (m, d), w_dtype)
            x_ty = mctx.data_ty('x_ty', (d, 1), x_dtype)
            y_ty = mctx.data_ty('y_ty', (m, 1), y_dtype)
            y_acc_ty = mctx.data_ty('y_acc_ty', (m, 1), y_acc_dtype)
            w_mem_ty = mctx.data_ty('w_mem_ty', (m * 4, d), w_dtype)
            x_mem_ty = mctx.data_ty('x_mem_ty', (d, 1), x_dtype)
            y_mem_ty = mctx.data_ty('y_mem_ty', (m * 4, 1), y_dtype)
            M_div_m_ty = mctx.data_ty('M_div_m_ty', (8, 1), m_dtype)
            K_div_k_ty = mctx.data_ty('K_div_k_ty', (8, 1), m_dtype)

            mctx.parameter("m", m)
            mctx.parameter("d", d)

            mctx.func("mv", [mctx.y_ty, mctx.y_ty, mctx.w_ty, mctx.w_ty, mctx.x_ty, mctx.x_ty, mctx.y_acc_ty, mctx.M_div_m_ty, mctx.K_div_k_ty])
            
            for j in range(total_cols):
                memref.global_(f"w_in_{j}", T.memref(M * D, T.bf16()), sym_visibility = "public")
                memref.global_(f"x_in_{j}", T.memref(D * 1, T.bf16()), sym_visibility = "public")
                memref.global_(f"y_out_{j}", T.memref(M * 1, T.bf16()), sym_visibility = "public")

            mv_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]

            M_div_m_buffers = []
            K_div_k_buffers = []
            for i in range(total_rows):
                for j in range(total_cols):
                    M_div_m_buffers.append(mctx.buffer(f"M_div_m_buffer_{mv_core[i][j].row}_{mv_core[i][j].col}", mv_core[i][j], M_div_m_ty, use_write_rtp = True))
                    K_div_k_buffers.append(mctx.buffer(f"K_div_k_buffer_{mv_core[i][j].row}_{mv_core[i][j].col}", mv_core[i][j], K_div_k_ty, use_write_rtp = True))


            for j in range(total_cols):
                for i in range(total_rows):
                    mv_tile(mv_core[i][j], mctx)
                mv_memtile(MT[j], mctx)

            # w_in
            for j in range(total_cols):
                connect_dma(IT[j], MT[j], 0, 0)
                for i in range(total_rows):
                    connect_dma(MT[j], mv_core[i][j], i, 0)

            # x_in
            for j in range(total_cols):
                connect_dma(IT[j], MT[j], 1, 1)
                for i in range(total_rows):
                    connect_dma(MT[j], mv_core[i][j], 4, 1)

            # c_out
            for j in range(total_cols):
                for i in range(total_rows):
                    connect_dma(mv_core[i][j], MT[j], 0, i + 2)
                connect_dma(MT[j], IT[j], 5, 0)

        print(ctx.module)

if __name__ == '__main__':
    flow_arch()