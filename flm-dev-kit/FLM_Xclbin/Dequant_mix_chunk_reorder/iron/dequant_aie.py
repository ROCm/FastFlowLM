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
from dequant_tile import dequant_memtile, dequant_tile

def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    Q4NX_ROW_BLOCK_SIZE: int = 32
    Q4NX_COL_BLOCK_SIZE: int = 256

    W_BLOCK_SIZE = 256
    W_SIZE = Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE

    w_dtype = np.dtype[bfloat16]
    w16_dtype = np.dtype[bfloat16]

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            w_ty = mctx.data_ty('w_ty', (W_BLOCK_SIZE, 1), w_dtype)
            w8_ty = mctx.data_ty('w8_ty', (W_BLOCK_SIZE, 8), w_dtype)
            w16_ty = mctx.data_ty('w16_ty', (W_SIZE, 1), w16_dtype)
            w_mem_ty = mctx.data_ty('w_mem_ty', (W_BLOCK_SIZE * 2, 1), w_dtype) # 2 matches to the vertical_blocks in the reorder_copy function
            w16_mem_ty = mctx.data_ty('w16_mem_ty', (W_SIZE * 2, 1), w16_dtype)
            rtp_ty = mctx.data_ty('rtp_ty', (16, ), np.dtype[np.int32])

            mctx.parameter('W_BLOCK_SIZE', W_BLOCK_SIZE) 
            mctx.parameter('W_SIZE', W_SIZE) 
            mctx.parameter('Q4NX_ROW_BLOCK_SIZE', Q4NX_ROW_BLOCK_SIZE)
            mctx.parameter('Q4NX_COL_BLOCK_SIZE', Q4NX_COL_BLOCK_SIZE)

            mctx.func("dequant", [mctx.w16_ty, mctx.w_ty, mctx.w16_ty, mctx.w_ty, mctx.w_ty, mctx.w_ty, mctx.w8_ty, mctx.rtp_ty, mctx.rtp_ty, mctx.rtp_ty, mctx.rtp_ty])

            
            dequant_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]
            
            rtp_buffers = []
            rtp_transpose_buffers = []
            Din_buffers = []
            Dout_buffers = []
            for i in range(total_rows):
                for j in range(total_cols):
                    rtp_buffers.append(mctx.buffer(f"rtp_{dequant_core[i][j].row}_{dequant_core[i][j].col}", dequant_core[i][j], rtp_ty, use_write_rtp = True))
                    rtp_transpose_buffers.append(mctx.buffer(f"rtp_transpose_{dequant_core[i][j].row}_{dequant_core[i][j].col}", dequant_core[i][j], rtp_ty, use_write_rtp = True))
                    Din_buffers.append(mctx.buffer(f"Din_{dequant_core[i][j].row}_{dequant_core[i][j].col}", dequant_core[i][j], rtp_ty, use_write_rtp = True))
                    Dout_buffers.append(mctx.buffer(f"Dout_{dequant_core[i][j].row}_{dequant_core[i][j].col}", dequant_core[i][j], rtp_ty, use_write_rtp = True))

            for j in range(total_cols):
                for i in range(total_rows):
                    dequant_tile(dequant_core[i][j], mctx)
                dequant_memtile(MT[j], mctx)

             # w_in
            for j in range(total_cols):
                connect_dma(IT[j], MT[j], 0, 0)
                connect_dma(IT[j], MT[j], 1, 1)
                for i in range(total_rows):
                    connect_dma(MT[j], dequant_core[i][j], i, 0)

            # y_out 
            for j in range(total_cols):
                for i in range(total_rows):
                    connect_dma(dequant_core[i][j], MT[j], 0, i + 2)
                connect_dma(MT[j], IT[j], 4, 0)
                connect_dma(MT[j], IT[j], 5, 1)

        print(ctx.module)


if __name__ == "__main__":
    flow_arch()