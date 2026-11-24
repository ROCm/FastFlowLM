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
from mv_dequant_tile import mv_dequant_tile, mv_dequant_memtile
from mv_tile import mv_tile, mv_memtile

# RTP buffers sit at fixed core addresses that the runtime (FLM_DLL/detail/whisper_npu) writes to.
# Pinned so a different compiler or stack size cannot move them. Keep in sync with the host.
M_DIV_M_RTP_ADDRESS = 61440
K_DIV_K_RTP_ADDRESS = 62464


def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    w_dtype = np.dtype[bfloat16]
    x_dtype = np.dtype[bfloat16]
    y_dtype = np.dtype[bfloat16]
    m_dtype = np.dtype[np.uint32]
    y_acc_dtype = np.dtype[np.float32]

    m_dequant = 32
    k_dequant = 128
    w_block_size = m_dequant * k_dequant * 5 // 8 // 2

    m = 16
    k = 64

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            w_dequant_ty = mctx.data_ty('w_dequant_ty', (w_block_size, 1), w_dtype)
            x_dequant_ty = mctx.data_ty('x_dequant_ty', (k_dequant, 1), x_dtype)
            y_dequant_ty = mctx.data_ty('y_dequant_ty', (m_dequant, 1), y_dtype)
            y_acc_dequant_ty = mctx.data_ty('y_acc_dequant_ty', (m_dequant, 1), y_acc_dtype)
            w_mem_dequant_ty = mctx.data_ty('w_mem_dequant_ty', (w_block_size * 4, 1), w_dtype)
            x_mem_dequant_ty = mctx.data_ty('x_mem_dequant_ty', (k_dequant, 1), x_dtype)
            y_mem_dequant_ty = mctx.data_ty('y_mem_dequant_ty', (m_dequant * 4, 1), y_dtype)
            M_div_m_dequant_ty = mctx.data_ty('M_div_m_dequant_ty', (8, 1), m_dtype)
            K_div_k_dequant_ty = mctx.data_ty('K_div_k_dequant_ty', (8, 1), m_dtype)

            w_ty = mctx.data_ty('w_ty', (m, k), w_dtype)
            x_ty = mctx.data_ty('x_ty', (k, 1), x_dtype)
            y_ty = mctx.data_ty('y_ty', (m, 1), y_dtype)
            y_acc_ty = mctx.data_ty('y_acc_ty', (m, 1), y_acc_dtype)
            w_mem_ty = mctx.data_ty('w_mem_ty', (m * 4, k), w_dtype)
            x_mem_ty = mctx.data_ty('x_mem_ty', (k, 1), x_dtype)
            y_mem_ty = mctx.data_ty('y_mem_ty', (m * 4, 1), y_dtype)
            M_div_m_ty = mctx.data_ty('M_div_m_ty', (8, 1), m_dtype)
            K_div_k_ty = mctx.data_ty('K_div_k_ty', (8, 1), m_dtype)

            mctx.parameter("m", m)
            mctx.parameter("k", k)
            mctx.parameter("m_dequant", m_dequant)
            mctx.parameter("k_dequant", k_dequant)
            mctx.parameter("w_block_size", w_block_size)

            mctx.func("mv_dequant", [mctx.y_dequant_ty, mctx.y_dequant_ty, mctx.w_dequant_ty, mctx.w_dequant_ty, mctx.x_dequant_ty, mctx.x_dequant_ty, mctx.y_acc_dequant_ty, mctx.M_div_m_dequant_ty, mctx.K_div_k_dequant_ty])
            mctx.func("mv", [mctx.y_ty, mctx.y_ty, mctx.w_ty, mctx.w_ty, mctx.x_ty, mctx.x_ty, mctx.y_acc_ty, mctx.M_div_m_ty, mctx.K_div_k_ty])
            
            for j in range(total_cols):
                memref.global_(f"w_in_{j}", T.memref(m * k, T.bf16()), sym_visibility = "public")
                memref.global_(f"x_in_{j}", T.memref(k * 1, T.bf16()), sym_visibility = "public")
                memref.global_(f"y_out_{j}", T.memref(m * 1, T.bf16()), sym_visibility = "public")

            mv_dequant_core = [[CT[i][j + 2] for j in range(4)] for i in range(total_rows)]
            mv_core = [[CT[i][0], CT[i][1], CT[i][6], CT[i][7]] for i in range(total_rows)]

            M_div_m_buffers = []
            K_div_k_buffers = []
            M_div_m_dequant_buffers = []
            K_div_k_dequant_buffers = []
            for i in range(total_rows):
                for j in range(4):
                    M_div_m_buffers.append(mctx.buffer(f"M_div_m_buffer_{mv_core[i][j].row}_{mv_core[i][j].col}", mv_core[i][j], M_div_m_ty, use_write_rtp = True, address = M_DIV_M_RTP_ADDRESS))
                    K_div_k_buffers.append(mctx.buffer(f"K_div_k_buffer_{mv_core[i][j].row}_{mv_core[i][j].col}", mv_core[i][j], K_div_k_ty, use_write_rtp = True, address = K_DIV_K_RTP_ADDRESS))
                    M_div_m_dequant_buffers.append(mctx.buffer(f"M_div_m_dequant_buffer_{mv_dequant_core[i][j].row}_{mv_dequant_core[i][j].col}", mv_dequant_core[i][j], M_div_m_dequant_ty, use_write_rtp = True, address = M_DIV_M_RTP_ADDRESS))
                    K_div_k_dequant_buffers.append(mctx.buffer(f"K_div_k_dequant_buffer_{mv_dequant_core[i][j].row}_{mv_dequant_core[i][j].col}", mv_dequant_core[i][j], K_div_k_dequant_ty, use_write_rtp = True, address = K_DIV_K_RTP_ADDRESS))


            for j in range(4):
                for i in range(total_rows):
                    mv_dequant_tile(mv_dequant_core[i][j], mctx)
                    mv_tile(mv_core[i][j], mctx)
                mv_dequant_memtile(MT[j + 2], mctx)
            mv_memtile(MT[0], mctx)
            mv_memtile(MT[1], mctx)
            mv_memtile(MT[6], mctx)
            mv_memtile(MT[7], mctx)

            # w_dequant_in
            for j in range(4):
                connect_dma(IT[j + 2], MT[j + 2], 0, 0)
                for i in range(total_rows):
                    connect_dma(MT[j + 2], mv_dequant_core[i][j], i, 0)

            # x_dequant_in
            for j in range(4):
                connect_dma(IT[j + 2], MT[j + 2], 1, 1)
                for i in range(total_rows):
                    connect_dma(MT[j + 2], mv_dequant_core[i][j], 4, 1)

            # c_out_dequant
            for j in range(4):
                for i in range(total_rows):
                    connect_dma(mv_dequant_core[i][j], MT[j + 2], 0, i + 2)
                connect_dma(MT[j + 2], IT[j + 2], 5, 0)
            
            # w_in
            for j in range(2):
                connect_dma(IT[j], MT[j], 0, 0)
                connect_dma(IT[j + 6], MT[j + 6], 0, 0)
                for i in range(total_rows):
                    connect_dma(MT[j], mv_core[i][j], i, 0)
                    connect_dma(MT[j + 6], mv_core[i][j + 2], i, 0)

            # x_in
            for j in range(2):
                connect_dma(IT[j], MT[j], 1, 1)
                connect_dma(IT[j + 6], MT[j + 6], 1, 1)
                for i in range(total_rows):
                    connect_dma(MT[j], mv_core[i][j], 4, 1)
                    connect_dma(MT[j + 6], mv_core[i][j + 2], 4, 1)

            # c_out
            for j in range(2):
                for i in range(total_rows):
                    connect_dma(mv_core[i][j], MT[j], 0, i + 2)
                    connect_dma(mv_core[i][j + 2], MT[j + 6], 0, i + 2)
                connect_dma(MT[j], IT[j], 5, 0)
                connect_dma(MT[j + 6], IT[j + 6], 5, 0)

        print(ctx.module)

if __name__ == '__main__':
    flow_arch()