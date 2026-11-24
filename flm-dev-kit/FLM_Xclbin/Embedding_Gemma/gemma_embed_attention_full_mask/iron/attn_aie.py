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
from attn_tile import attn_cttile, attn_memtile

def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 3
    d = 256
    D = 768
    DQ = 768
    NH = 3
    G = 1
    HG = NH // G

    L = 2048
    DH = DQ // NH
    DK = DH * G
    DV = DH * G
    d = DH
    lq = 16
    lk = 16
    total_rows = HG # only HG rows can be used

    q_dtype = np.dtype[bfloat16]
    k_dtype = np.dtype[bfloat16]
    v_dtype = np.dtype[bfloat16]
    o_dtype = np.dtype[bfloat16]
    y_dtype = np.dtype[np.float32]

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            q_ty = mctx.data_ty('q_ty', (lq, d), q_dtype)
            q_slice_ty = mctx.data_ty('q_slice_ty', (HG, d), q_dtype)
            k_ty = mctx.data_ty('k_ty', (lk, d), k_dtype) 
            v_ty = mctx.data_ty('v_ty', (lk, d), v_dtype)
            k_mem_ty = mctx.data_ty('k_mem_ty', (lk, d), k_dtype)
            v_mem_ty = mctx.data_ty('v_mem_ty', (lk, d), v_dtype)
            o_ty = mctx.data_ty('o_ty', (64, ), o_dtype)
            o_mem_ty = mctx.data_ty('o_mem_ty', (lq, d * HG), o_dtype)
            s_ty = mctx.data_ty('s_ty', (lq, lk), v_dtype)
            y_ty = mctx.data_ty('y_ty', (lq, d), y_dtype)
            L_ty = mctx.data_ty('L_ty', (8,), np.dtype[np.int32])

            mctx.parameter('lq', lq)
            mctx.parameter('lk', lk)
            mctx.parameter('d', d)
            mctx.parameter('DQ', DQ)
            mctx.parameter('G', G)
            mctx.parameter('HG', HG)
            mctx.func("attn", [q_slice_ty, q_slice_ty, v_ty, v_ty, o_ty, o_ty, q_ty, s_ty, y_ty, L_ty, T.i32()])

            for j in range(total_cols):
                memref.global_(f"q_in_{j}", T.memref(L * DQ, T.bf16()), sym_visibility = "public")
                memref.global_(f"o_out_{j}", T.memref(L * DQ, T.bf16()), sym_visibility = "public")

            attn_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]
            L_end_buffers = []
            for i in range(total_rows):
                for j in range(total_cols):
                    L_end_buffers.append(mctx.buffer(f"L_end_attn_core_{attn_core[i][j].row}_{attn_core[i][j].col}", attn_core[i][j], L_ty, use_write_rtp = True))

            for j in range(total_cols):
                for i in range(total_rows):
                    attn_cttile(attn_core[i][j], mctx)
                attn_memtile(MT[j], mctx)

            # q_in
            for j in range(total_cols):
                for i in range(total_rows):
                    connect_dma(IT[j], attn_core[i][j], 0, 0)

            # k_in
            for j in range(total_cols):
                connect_dma(IT[0], MT[j], 1, 0)
                connect_dma(IT[4], MT[j], 1, 1)
                for i in range(total_rows):
                    connect_dma(MT[j], attn_core[i][j], 0, 1)
    
            # o_out
            for j in range(total_cols):
                for i in range(total_rows):
                    connect_dma(attn_core[i][j], MT[j], 0, 2 + i)
                connect_dma(MT[j], IT[j], 1, 0)

        print(ctx.module)

if __name__ == "__main__":
    flow_arch()
