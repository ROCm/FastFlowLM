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
from non_linear_tile import non_linear_tile, non_linear_memtile


def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 1    
    total_rows = 1
    
    D = 4096

    x_dtype = np.dtype[bfloat16]
    o_dtype = np.dtype[bfloat16]

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            x_ty = mctx.data_ty('x_ty', (D, ), x_dtype)
            x_mem_ty = mctx.data_ty('x_mem_ty', (D, ), x_dtype)
            L_ty = mctx.data_ty('L_ty', (8,), np.dtype[np.int32])

            o_ty = mctx.data_ty('o_ty', (D, ), o_dtype)
            o_mem_ty = mctx.data_ty('o_mem_ty', (D, ), o_dtype)

            mctx.parameter('D', D)
            mctx.func("non_linear", [o_ty, o_ty, x_ty, x_ty, L_ty, T.i32()])

            for j in range(total_cols):
                memref.global_(f"x_{j}", T.memref(D, T.bf16()), sym_visibility = "public")
                memref.global_(f"o_{j}", T.memref(D, T.bf16()), sym_visibility = "public")

            non_linear_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]
            
            rtp_buffers = []
            for i in range(total_rows):
                for j in range(total_cols):
                    rtp_buffers.append(mctx.buffer(f"rtp_activate_func_impl_{non_linear_core[i][j].row}_{non_linear_core[i][j].col}", non_linear_core[i][j], L_ty, use_write_rtp = True))

            
            non_linear_tile(CT[0][0], mctx)
            non_linear_memtile(MT[0], mctx)
        
            # x_in
            connect_dma(IT[0], MT[0], 0, 0)
            connect_dma(MT[0], non_linear_core[0][0], 0, 0)

                    
            # o_out
            connect_dma(non_linear_core[0][0], MT[0], 0, 1)
            connect_dma(MT[0], IT[0], 1, 0)

        print(ctx.module)

if __name__ == "__main__":
    flow_arch()
