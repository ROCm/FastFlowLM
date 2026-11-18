# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Qwen3.8-27B vision-tower matmul.

Array mapping: 4 rows split M, 8 cols split N. One array pass therefore covers
(4*m) x (8*n) of the output:

              col 0      col 1     ...   col 7        <- each col owns n of N
    row 0   CT[0][0]   CT[0][1]    ...  CT[0][7]       <- each row owns m of M
    row 1   CT[1][0]   CT[1][1]    ...  CT[1][7]
    row 2     ...
    row 3   CT[3][0]     ...       ...  CT[3][7]

  x  MT[i] -> CT[i][*]   broadcast along a row (i < total_rows)
  w  MT[j] -> CT[*][j]   broadcast down a column
  y  CT[*][j] -> MT[j]   four row-partials rejoin as a (4m, n) column chunk

Note this is the transpose of the SmolVLA scaffold this design started from,
which used 8 cols for M and 4 rows for N. Rows-for-M gives an M granularity of
4*m (256 / 128) instead of 8*m, which matters because the merger layers run on
L/4 rows after the 2x2 spatial merge.

Because the swap is symmetric it compiles and runs either way round; the host's
rotated-weight control is what actually proves the mapping, not the build.
"""

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
from mm_config import mm_config


def flow_arch(design: str):
    dev = aie.AIEDevice.npu2
    cfg = mm_config(design)

    total_cols = cfg.total_cols
    total_rows = cfg.total_rows

    m = cfg.m
    k = cfg.k
    n = cfg.n
    m_slice = cfg.m_slice
    k_slice = cfg.k_slice
    n_slice = cfg.n_slice
    y_slice = cfg.y_slice

    x_dtype = cfg.x_dtype
    w_dtype = cfg.w_dtype
    y_dtype = cfg.y_dtype
    y_acc_dtype = cfg.y_acc_dtype

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
            b_ty = mctx.data_ty('b_ty', (n, ), y_acc_dtype)
            # One column chunk: the total_rows row-partials stacked along M.
            y_mem_ty = mctx.data_ty('y_mem_ty', (total_rows * m, n), y_dtype)
            K_div_k_ty = mctx.data_ty('K_div_k_ty', (16, ), np.dtype[np.int32])

            mctx.parameter('m', m)
            mctx.parameter('k', k)
            mctx.parameter('n', n)
            mctx.parameter('m_slice', m_slice)
            mctx.parameter('k_slice', k_slice)
            mctx.parameter('n_slice', n_slice)
            mctx.parameter('total_rows', total_rows)
            mctx.parameter('total_cols', total_cols)

            mctx.func("mm", [mctx.x_ty, mctx.x_ty, mctx.w_ty, mctx.w_ty, mctx.y_ty, mctx.y_ty, mctx.y_acc_ty, mctx.b_ty, mctx.K_div_k_ty])

            mm_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]
            rtp_buffers = []
            for i in range(total_rows):
                for j in range(total_cols):
                    rtp_buffers.append(mctx.buffer(f"rtp_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp = True))

            for j in range(total_cols):
                for i in range(total_rows):
                    mm_cttile(mm_core[i][j], mctx)
                mm_memtile(MT[j], mctx)

            # x_in: memtile i feeds array row i, broadcast across that row.
            for i in range(total_rows):
                connect_dma(IT[i], MT[i], 0, 0)
                for j in range(total_cols):
                    connect_dma(MT[i], mm_core[i][j], 0, 0)

            # w_in: memtile j feeds array column j, broadcast down that column.
            for j in range(total_cols):
                connect_dma(IT[j], MT[j], 1, 1)
                for i in range(total_rows):
                    connect_dma(MT[j], mm_core[i][j], 1, 1)

            # y_out: the four row-partials of column j land on their own S2MM
            # channel of memtile j, which then streams the joined chunk out.
            for j in range(total_cols):
                for i in range(total_rows):
                    connect_dma(mm_core[i][j], MT[j], 0, 2 + i)

            for j in range(total_cols):
                connect_dma(MT[j], IT[j], 2, 0)

        print(ctx.module)


if __name__ == "__main__":
    design = sys.argv[1] if len(sys.argv) > 1 else "VISION_MM_A"
    flow_arch(design)
