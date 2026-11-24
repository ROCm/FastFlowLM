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
from dequant_mm_tile import dequant_mm_memtile, dequant_mm_tile

def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    m = 128
    k = 512
    n = 32
    
    m_slice = 16
    k_slice = 256
    n_slice = 32
    y_slice = 64
    
    W_BLOCK_SIZE = 256

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
            qw_ty = mctx.data_ty('qw_ty', (W_BLOCK_SIZE, 1), w_dtype)
            local_w = mctx.data_ty('w8_ty', (W_BLOCK_SIZE, 8), w_dtype)
            w_ty = mctx.data_ty('w_ty', (n_slice, k_slice), w_dtype)
            qw_mem_ty = mctx.data_ty('qw_mem_ty', (W_BLOCK_SIZE * 2, 1), w_dtype) # 2 matches to the vertical_blocks in the reorder_copy function
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
            mctx.parameter('W_BLOCK_SIZE', W_BLOCK_SIZE) 

            mctx.func("dequant_mm_e", [mctx.y_ty, mctx.y_ty, mctx.x_ty, mctx.x_ty, mctx.qw_ty, mctx.qw_ty, mctx.w8_ty, mctx.w_ty,mctx.w_ty, mctx.qw_ty, mctx.qw_ty, mctx.y_acc_ty, mctx.K_div_k_ty, mctx.K_div_k_ty, mctx.K_div_k_ty, mctx.K_div_k_ty])
            mctx.func("dequant_mm_o", [mctx.y_ty, mctx.y_ty, mctx.x_ty, mctx.x_ty, mctx.qw_ty, mctx.qw_ty, mctx.w8_ty, mctx.w_ty,mctx.w_ty, mctx.qw_ty, mctx.qw_ty, mctx.y_acc_ty, mctx.K_div_k_ty, mctx.K_div_k_ty, mctx.K_div_k_ty, mctx.K_div_k_ty])

            mm_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]
            K_div_k_buffers = []
            rtp_activate_buffers = []
            rtp_q4_buffers = []
            rtp_zp_buffers = []
            w_buffers = []

            for i in range(total_rows):
                for j in range(total_cols):
                    K_div_k_buffers.append(mctx.buffer(f"K_div_k_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp = True))
                    rtp_activate_buffers.append(mctx.buffer(f"rtp_act_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp = True))
                    rtp_q4_buffers.append(mctx.buffer(f"rtp_q4_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp = True))
                    rtp_zp_buffers.append(mctx.buffer(f"rtp_zp_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp = True))

            # The dequantized weight tile is shared by each (even, odd) row pair:
            # the two cores split the weight slices between them and dequantize
            # into one buffer.  It is allocated on the even tile of the pair; the
            # odd core reaches it as its neighbour's local memory.
            for i in range(total_rows):
                for j in range(total_cols):
                    w_buffers.append(mctx.buffer(f"w_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], w_ty))

            for j in range(total_cols):
                for i in range(total_rows):
                    dequant_mm_tile(mm_core[i][j], mctx)
                dequant_mm_memtile(MT[j], mctx)
  
            # w_in
            for j in range(4):
                connect_dma(IT[2 * j], MT[2 * j], 1, 5)
                for i in range(total_rows):
                    connect_dma(MT[2 * j], mm_core[i][2 * j], 2, 1)
                    connect_dma(MT[2 * j], mm_core[i][2 * j + 1], 3, 1)

            # x_in
            for i in range(total_rows):
                connect_dma(IT[i * 2], MT[i * 2], 0, 0)
                for j in range(total_cols):
                    connect_dma(MT[i * 2], mm_core[i][j], 0, 0)

            # y_out 
            for i in range(total_rows):
                connect_dma(mm_core[i][0], MT[i], 0, 1)
                connect_dma(mm_core[i][1], MT[i], 0, 2)
                connect_dma(mm_core[i][2], MT[i], 0, 3)
                connect_dma(mm_core[i][3], MT[i], 0, 4)
                connect_dma(mm_core[i][4], MT[i + 4], 0, 1)
                connect_dma(mm_core[i][5], MT[i + 4], 0, 2)
                connect_dma(mm_core[i][6], MT[i + 4], 0, 3)
                connect_dma(mm_core[i][7], MT[i + 4], 0, 4)

            for j in range(total_cols):
                connect_dma(MT[j], IT[j], 1, 0)

        print(ctx.module)


if __name__ == "__main__":
    flow_arch()