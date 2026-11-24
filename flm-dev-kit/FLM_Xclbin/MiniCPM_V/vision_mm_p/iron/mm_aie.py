# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""MiniCPM-V vision-tower matmul, design P: design C with packed bfp16 operands.

Same array mapping, tiles and y path as design C (below). What differs is what
x and w are in DDR and on every link: bfp16 tiles the host packed, 9/16 of the
bf16 bytes, which the kernel multiplies in place (see mm_config.py for the
layouts). The x and w DDR reads were measured to be the limit of design C.

Design C's description:

Array mapping: 8 columns split M, 4 rows split N. One array pass therefore
covers (8*m) x (4*n) of the output (512 x 384 with m 64, n 96):

              col 0      col 1     ...   col 7        <- each col owns m tokens
    row 0   CT[0][0]   CT[0][1]    ...  CT[0][7]       <- each row owns n channels
    row 1   CT[1][0]   CT[1][1]    ...  CT[1][7]
    row 2     ...
    row 3   CT[3][0]     ...       ...  CT[3][7]

  x  MT[c] -> CT[*][c]   column c's m tokens, broadcast down the column (all 8 memtiles)
  w  MT[i] -> CT[i][*]   row i's n channels, broadcast along the row (memtiles 0..3)
  y  CT[*][c] -> MT[c]   the four row outputs of a token join into one (m, 4n) block of
                         full token rows, so the shim writes 768 B row segments to DDR

This is the transpose of the first vision_mm (Qwen3_8/vision_mm, rows split M,
columns split N, n = 48), whose y write-back was 96 B segments at a row stride
and whose x and w were re-streamed per 256-token tile. The kernel here converts
each arriving bf16 chunk once into bfp16 tiles and runs block-vector MMULs (no
per-use conversion or transposes); the operands arrive in the same layouts as
before, so the host's x / w / y formats are unchanged.
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
            # x and w travel as packed bfp16 tiles (9 B per 8 values), typed as int32
            # words so every DMA length and stride is in whole words.
            i32 = np.dtype[np.int32]
            x_ty = mctx.data_ty('x_ty', (cfg.x_chunk_bytes // 4,), i32)
            x_mem_ty = mctx.data_ty('x_mem_ty', (cfg.x_mem_bytes // 4,), i32)
            w_ty = mctx.data_ty('w_ty', (cfg.w_chunk_bytes // 4,), i32)
            w_mem_ty = mctx.data_ty('w_mem_ty', (cfg.w_mem_bytes // 4,), i32)
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

            # MAPPING C (columns split M, rows split N):
            # x_in: memtile c holds column c's token tile, broadcast down that column.
            for c in range(total_cols):
                connect_dma(IT[c], MT[c], 0, 0)
                for i in range(total_rows):
                    connect_dma(MT[c], mm_core[i][c], 0, 0)

            # w_in: memtile i (i < total_rows) holds core row i's channel slice,
            # broadcast along that row to all 8 columns.
            for i in range(total_rows):
                connect_dma(IT[i], MT[i], 1, 1)
                for j in range(total_cols):
                    connect_dma(MT[i], mm_core[i][j], 1, 1)

            # y_out: the four row-partials of column j land on their own S2MM
            # channel of memtile j, which then streams the joined chunk out.
            for j in range(total_cols):
                for i in range(total_rows):
                    connect_dma(mm_core[i][j], MT[j], 0, 2 + i)

            for j in range(total_cols):
                connect_dma(MT[j], IT[j], 2, 0)

        print(ctx.module)


if __name__ == "__main__":
    design = sys.argv[1] if len(sys.argv) > 1 else "VISION_MM_P"
    flow_arch(design)
