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

    # Phase 2 layout (see conv_tile.py): every core owns one 128-wide q/k head and
    # 64 v channels. 32 cores x (128 + 64) = 6144 = MiniCPM-OCR's q + k + v.
    d_qk = 128
    d_v = 64
    d = d_qk + d_v

    x_dtype = np.dtype[bfloat16]
    o_dtype = np.dtype[bfloat16]

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            x_in_ty = mctx.data_ty('x_in_ty', (d, ), x_dtype)        # one row for one core
            x_ty = mctx.data_ty('x_ty', (4, d), x_dtype)             # the core's 4-row ring
            w_ty = mctx.data_ty('w_ty', (4, d), x_dtype)             # the core's 4 weight rows
            o_ty = mctx.data_ty('o_ty', (d, ), o_dtype)
            qk_mem_ty = mctx.data_ty('qk_mem_ty', (d_qk * total_rows, ), x_dtype)   # one q/k row of a column
            v_mem_ty = mctx.data_ty('v_mem_ty', (d_v * total_rows, ), x_dtype)      # one v row of a column
            L_ty = mctx.data_ty('L_ty', (8,), np.dtype[np.int32])

            mctx.parameter('d_qk', d_qk)
            mctx.parameter('d_v', d_v)
            mctx.parameter('n_core', total_rows)
            mctx.func("conv", [o_ty, o_ty, x_in_ty, x_in_ty, x_ty, w_ty, L_ty])

            conv_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]

            for i in range(total_rows):
                for j in range(total_cols):
                    mctx.buffer(f"L_buffer_{conv_core[i][j].row}_{conv_core[i][j].col}", conv_core[i][j], L_ty, use_write_rtp = True)

            for j in range(total_cols):
                for i in range(total_rows):
                    conv_cu(CT[i][j], mctx)
                conv_memtile(MT[j], mctx)

            for j in range(total_cols):
                # in: q/k rows on shim MM2S 0, v rows on shim MM2S 1
                connect_dma(IT[j], MT[j], 0, 0)
                connect_dma(IT[j], MT[j], 1, 1)
                for i in range(total_rows):
                    # mem tile MM2S i -> core i, core i -> mem tile S2MM 2+i
                    connect_dma(MT[j], conv_core[i][j], i, 0)
                    connect_dma(conv_core[i][j], MT[j], 0, i + 2)
                # out: q/k rows to shim S2MM 0, v rows to shim S2MM 1
                connect_dma(MT[j], IT[j], 4, 0)
                connect_dma(MT[j], IT[j], 5, 1)

        print(ctx.module)

if __name__ == "__main__":
    flow_arch()