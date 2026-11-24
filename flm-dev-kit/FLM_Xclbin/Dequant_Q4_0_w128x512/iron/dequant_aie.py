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
from q4nx_dequant import dequant_tile, dequant_memtile

def ceildiv(a, b):
    return (a + b - 1) // b

def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    qw_dtype = np.dtype[np.uint32]
    w_dtype = np.dtype[bfloat16]

    qw_size = 32 * 256 * 5 // 8 // 4
    q_size = 32 * 256

    M = 2048
    N = 2048
    blocks = M * N // 32 // 256


    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            qw_ty = mctx.data_ty('qw_ty', (qw_size, 1), qw_dtype)
            w_ty = mctx.data_ty('w_ty', (q_size, 1), w_dtype)
            qw_mem_ty = mctx.data_ty('qw_mem_ty', (qw_size, total_rows), qw_dtype)
            w_mem_ty = mctx.data_ty('w_mem_ty', (q_size, total_rows), w_dtype)

            mctx.parameter("qw_size", qw_size)
            mctx.parameter("q_size", q_size)

            mctx.func("dequant", [mctx.w_ty, mctx.w_ty, mctx.qw_ty, mctx.qw_ty])

            for j in range(total_cols):
                memref.global_(f"w_out_{j}", T.memref(qw_size * 2048 * 2048 // 32 // 256, T.ui32()), sym_visibility = "public")
                memref.global_(f"wq_in_up_{j}", T.memref(2048 * 2048, T.bf16()), sym_visibility = "public")
                memref.global_(f"wq_in_down_{j}", T.memref(2048 * 2048, T.bf16()), sym_visibility = "public")

            dequant_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]

            for j in range(total_cols):
                for i in range(total_rows):
                    dequant_tile(dequant_core[i][j], mctx)
                dequant_memtile(MT[j], mctx)

            # qw_in
            for j in range(total_cols):
                connect_dma(IT[j], MT[j], 0, 0)
                connect_dma(IT[j], MT[j], 1, 1)
                connect_dma(MT[j], dequant_core[0][j], 0, 0)
                connect_dma(MT[j], dequant_core[1][j], 1, 0)
                connect_dma(MT[j], dequant_core[2][j], 2, 0)
                connect_dma(MT[j], dequant_core[3][j], 3, 0)

            # w_out
            for j in range(total_cols):
                connect_dma(dequant_core[0][j], MT[j], 0, 2)
                connect_dma(dequant_core[1][j], MT[j], 0, 3)
                connect_dma(dequant_core[2][j], MT[j], 0, 4)
                connect_dma(dequant_core[3][j], MT[j], 0, 5)
                connect_dma(MT[j], IT[j], 5, 0)

            for j in range(total_cols):
                aie.shim_dma_allocation(f"wq_in_up_{j}", aie.DMAChannelDir.MM2S, 0, j)
                aie.shim_dma_allocation(f"wq_in_down_{j}", aie.DMAChannelDir.MM2S, 1, j)
                aie.shim_dma_allocation(f"w_out_{j}", aie.DMAChannelDir.S2MM, 0, j)


            # @aiex.runtime_sequence(
            #     np.ndarray[(M, N), np.dtype[bfloat16]],
            #     np.ndarray[(blocks, qw_size), np.dtype[np.uint32]]
            # )

            # def sequence(w, wq):
            #     for round in range(M // 32 // total_cols // 2):
            #         bd_offset = (round % 2) * 8
            #         for c in range(total_cols):
            #             qw_offset = (round * total_cols + c) * qw_size * N // 256 * 2
            #             q_offset = (round * total_cols + c) * 64 * N
            #             aiex.npu_dma_memcpy_nd(
            #                 metadata=f"w_out_{c}",
            #                 bd_id=0 + bd_offset,
            #                 mem=w,
            #                 offsets=[0, 0, 0, q_offset],
            #                 sizes=[1, 1, 1, 64 * N],
            #                 strides=[0, 0, 0, 1],
            #                 issue_token = True
            #             )
            #             aiex.npu_dma_memcpy_nd(
            #                 metadata=f"wq_in_up_{c}",
            #                 bd_id=1 + bd_offset,
            #                 mem=wq,
            #                 offsets=[0, 0, 0, qw_offset],
            #                 sizes=[1, 1, 1, qw_size * N // 256],
            #                 strides=[0, 0, 0, 1],
            #             )
            #             aiex.npu_dma_memcpy_nd(
            #                 metadata=f"wq_in_down_{c}",
            #                 bd_id=2 + bd_offset,
            #                 mem=wq,
            #                 offsets=[0, 0, 0, qw_offset + qw_size * N // 256],
            #                 sizes=[1, 1, 1, qw_size * N // 256],
            #                 strides=[0, 0, 0, 1],
            #             )
            #             if round > 0:
            #                 aiex.dma_wait(f"w_out_{c}")
            #     for c in range(total_cols):
            #         aiex.dma_wait(f"w_out_{c}")


        print(ctx.module)

if __name__ == '__main__':
    flow_arch()
