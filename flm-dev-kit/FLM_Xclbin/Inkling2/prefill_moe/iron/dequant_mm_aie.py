# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.extras import types as T
import aie.dialects.aie as aie
import aie.dialects.aiex as aiex
try:  # mlir-aie as pinned by CI
    from aie.helpers.dialects.ext.scf import _for as range_
except ImportError:  # newer mlir-aie moved it
    from aie.helpers.dialects.scf import _for as range_
from aie.dialects import memref

from utils import dma_mm2s_ch, dma_s2mm_ch, flow_mm2s_ch, flow_s2mm_ch, connect_dma, generate_tile_map, mlir_context
from utils import dma_port, connect_port
from dequant_mm_tile import dequant_mm_memtile, dequant_mm_tile, X_RES_ELEMS, X_SLICE_ELEMS

def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    m = 128
    k = 256
    n = 48     # n per core and pass (3 regions of 16)
    
    m_slice = 16
    k_slice = 256
    n_slice = 32
    y_slice = 256

    x_dtype = np.dtype[bfloat16]
    w_dtype = np.dtype[bfloat16]
    y_dtype = np.dtype[bfloat16]
    y_acc_dtype = np.dtype[np.float32]
    qw_dtype = np.dtype[np.uint32]

    qw_size = int(k_slice * n_slice * 4.625 // 8 // 4)
    qw_mem_size = int(k * 96 * 4.625 // 8 // 4)   # a column's block triple
    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(4 + 2, 8)

            mctx = mlir_context()
            x_ty = mctx.data_ty('x_ty', (X_SLICE_ELEMS,), x_dtype)  # one m16 group's 256 k slice, bfp16 A tiles
            x_mem_ty = mctx.data_ty('x_mem_ty', (X_RES_ELEMS,), x_dtype)
            qw_ty = mctx.data_ty('qw_ty', (qw_size, 1), qw_dtype)
            w_ty = mctx.data_ty('w_ty', (9216,), np.dtype[np.uint8])  # one n slice of a k slice, bfp16 tiles (see mm_k3_common.h)
            qw_mem_ty = mctx.data_ty('qw_mem_ty', (qw_mem_size, 1), qw_dtype)
            y_ty = mctx.data_ty('y_ty', (y_slice,), y_dtype)
            y_acc_ty = mctx.data_ty('y_acc_ty', (m * 64, ), y_acc_dtype)  # 32 KB, keeps the L1 map
            y_mem_ty = mctx.data_ty('y_mem_ty', (4 * m * n, ), y_dtype)
            K_div_k_ty = mctx.data_ty('K_div_k_ty', (16, ), np.dtype[np.int32])

            mctx.parameter('m', m)
            mctx.parameter('k', k)
            mctx.parameter('n', n)
            mctx.parameter('m_slice', m_slice)
            mctx.parameter('k_slice', k_slice)
            mctx.parameter('n_slice', n_slice)
            mctx.parameter('qw_size', qw_size)
            mctx.parameter('qw_mem_size', qw_mem_size)

            mctx.func("dequant_mm_e", [mctx.y_ty, mctx.y_ty, mctx.x_ty, mctx.x_ty, mctx.qw_ty, mctx.qw_ty, mctx.w_ty, mctx.w_ty, mctx.y_acc_ty, mctx.K_div_k_ty, mctx.K_div_k_ty])
            mctx.func("dequant_mm_o", [mctx.y_ty, mctx.y_ty, mctx.x_ty, mctx.x_ty, mctx.qw_ty, mctx.qw_ty, mctx.w_ty, mctx.w_ty, mctx.y_acc_ty, mctx.K_div_k_ty, mctx.K_div_k_ty])

            mm_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]
            K_div_k_buffers = []
            rtp_activate_buffers = []

            for i in range(total_rows):
                for j in range(total_cols):
                    K_div_k_buffers.append(mctx.buffer(f"K_div_k_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp = True))
                    rtp_activate_buffers.append(mctx.buffer(f"rtp_act_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp = True))

            for j in range(total_cols):
                for i in range(total_rows // 2):
                    dequant_mm_tile(mm_core[2 * i][j], mm_core[2 * i + 1][j], mctx)
                dequant_mm_memtile(MT[j], mctx)
  
            # w_in: the memtile of column j feeds pair A (core row 2) on MM2S 2 and pair B (row 4) on MM2S 3
            for j in range(total_cols):
                connect_dma(IT[j], MT[j], 1, 5)
                connect_dma(MT[j], mm_core[0][j], 2, 1)
                connect_dma(MT[j], mm_core[2][j], 3, 1)

            # x_in: the even cores (rows 2, 4) take the first m rows of a block, the odd ones (3, 5) the next m
            for src, rows in ((0, (0, 2)), (2, (1, 3))):
                connect_dma(IT[src], MT[src], 0, 0)
                for i in rows:
                    for j in range(total_cols):
                        connect_dma(MT[src], mm_core[i][j], 0, 0)

            # y_out: the four cores of column j -> MT[j] S2MM 1..4 -> IT[j]
            for j in range(total_cols):
                for i in range(total_rows):
                    connect_dma(mm_core[i][j], MT[j], 0, i + 1)
                connect_dma(MT[j], IT[j], 1, 0)

        print(ctx.module)


if __name__ == "__main__":
    flow_arch()