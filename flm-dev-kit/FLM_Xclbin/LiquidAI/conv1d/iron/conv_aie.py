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
from conv_tile import conv1d_tile, conv1d_memtile

def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    w_dtype = np.dtype[bfloat16]
    x_dtype = np.dtype[bfloat16]
    y_dtype = np.dtype[bfloat16]

    D = 2048
    d = 64
    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            w_ty = mctx.data_ty('w_ty', (3, d * 4), w_dtype)
            x_ty = mctx.data_ty('x_ty', (1, d), x_dtype)
            y_ty = mctx.data_ty('y_ty', (1, d), y_dtype)
            x_mem_ty = mctx.data_ty('x_mem_ty', (4, d), x_dtype)
            y_mem_ty = mctx.data_ty('y_mem_ty', (4, d), y_dtype)
            L_ty = mctx.data_ty('L_ty', (8, ), np.dtype[np.int32])

            mctx.parameter("d", d)
            mctx.func("conv1d", [mctx.y_ty, mctx.y_ty, mctx.x_ty, mctx.x_ty, mctx.w_ty, L_ty, T.i32()])

            conv_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]    

            L_buffer = []
            for j in range(total_cols):
                for i in range(total_rows):
                    L_buffer.append(mctx.buffer(f"L_{conv_core[i][j].row}_{conv_core[i][j].col}", tile = conv_core[i][j], dtype = L_ty, use_write_rtp = True))
            
            # for j in range(total_cols):
            #     memref.global_(f"w_in_{j}", T.memref(3 * D, T.bf16()), sym_visibility = "public")
            #     memref.global_(f"x_in_{j}", T.memref((L + 2) * D, T.bf16()), sym_visibility = "public")
            #     memref.global_(f"y_out_{j}", T.memref(L * D, T.bf16()), sym_visibility = "public")


            for j in range(total_cols):
                for i in range(total_rows):
                    conv1d_tile(conv_core[i][j], mctx)
                conv1d_memtile(MT[j], mctx)

            for j in range(total_cols):
                for i in range(total_rows):
                    connect_dma(IT[j], conv_core[i][j], 0, 0) #w_in

            for j in range(total_cols):
                for i in range(total_rows):
                    connect_dma(IT[j], MT[j], 1, 0) #x_in
                    connect_dma(MT[j], conv_core[i][j], i, 1)
            
            for j in range(total_cols):
                for i in range(total_rows):
                    connect_dma(conv_core[i][j], MT[j], 0, 1 + i) #y_out
                    connect_dma(MT[j], IT[j], 4, 0)


        print(ctx.module)

if __name__ == '__main__':
    flow_arch()