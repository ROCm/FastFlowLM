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
from conv_tile import conv_cu, conv_memtile


def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4
    
    d = 256

    w_dtype = np.dtype[bfloat16]
    x_dtype = np.dtype[bfloat16]
    o_dtype = np.dtype[bfloat16]

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            x_in_ty = mctx.data_ty('x_in_ty', (d, ), x_dtype)
            x_ty = mctx.data_ty('x_ty', (5, d), x_dtype)
            x_mem_ty = mctx.data_ty('x_mem_ty', (1, d * 4), x_dtype) 
            w_ty = mctx.data_ty('w_ty', (5, d * 4), w_dtype) 
            w_mem_ty = mctx.data_ty('w_mem_ty', (5, d * 4), w_dtype)

            o_ty = mctx.data_ty('o_ty', (d, ), o_dtype)
            o_mem_ty = mctx.data_ty('o_mem_ty', (1, d * 4), o_dtype)
            L_ty = mctx.data_ty('L_ty', (16,), np.dtype[np.int32])
            min_max_ty = mctx.data_ty('min_max_ty', (16,), np.dtype[np.float32])

            mctx.parameter('d', d)
            mctx.func("conv", [o_ty, o_ty, x_in_ty, x_in_ty, x_ty, w_ty, min_max_ty, min_max_ty, L_ty, L_ty, T.i32()])

            for j in range(total_cols):
                memref.global_(f"x_{j}", T.memref(1 * d * 4, T.bf16()), sym_visibility = "public")
                memref.global_(f"w_{j}", T.memref(5 * d * 4, T.bf16()), sym_visibility = "public")
                memref.global_(f"o_{j}", T.memref(1 * d * 4, T.bf16()), sym_visibility = "public")

            conv_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]

            L_buffers = []
            round_buffers = []
            min_buffers = []
            max_buffers = []

            for i in range(total_rows):
                for j in range(total_cols):
                    L_buffers.append(mctx.buffer(f"L_buffer_{conv_core[i][j].row}_{conv_core[i][j].col}", conv_core[i][j], L_ty, use_write_rtp = True))
                    round_buffers.append(mctx.buffer(f"round_buffer_{conv_core[i][j].row}_{conv_core[i][j].col}", conv_core[i][j], L_ty, use_write_rtp = True))
                    min_buffers.append(mctx.buffer(f"min_buffer_{conv_core[i][j].row}_{conv_core[i][j].col}", conv_core[i][j], min_max_ty, use_write_rtp = True))
                    max_buffers.append(mctx.buffer(f"max_buffer_{conv_core[i][j].row}_{conv_core[i][j].col}", conv_core[i][j], min_max_ty, use_write_rtp = True))
            
            
            for j in range(total_cols):
                for i in range(total_rows):
                    conv_cu(CT[i][j], mctx)
                conv_memtile(MT[j], mctx)
        
            # x_in
            for j in range(total_cols):
                connect_dma(IT[j], MT[j], 0, 0)
                for i in range(total_rows):
                    connect_dma(MT[j], conv_core[i][j], i, 0)

            # # w_in
            for j in range(total_cols): 
                connect_dma(IT[j], MT[j], 1, 1)
                for i in range(total_rows):
                    connect_dma(MT[j], conv_core[i][j], 4, 1)
                    
            # o_out
            for j in range(total_cols):
                for i in range(total_rows):
                    connect_dma(conv_core[i][j], MT[j], 0, i + 2)
                connect_dma(MT[j], IT[j], 5, 0)

        print(ctx.module)

if __name__ == "__main__":
    flow_arch()
