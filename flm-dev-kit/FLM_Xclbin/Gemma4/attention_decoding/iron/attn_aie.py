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
from attn_tile import attn_cu, aux_memtile, attn_memtile

def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 2
    total_rows = 2

    x_dtype = np.dtype[bfloat16]
    y_dtype = np.dtype[np.float32]
    o_dtype = np.dtype[bfloat16]
    lc = 16
    DH = 512
    Q_HEADS_PER_ROUND = 8

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            q_ty = mctx.data_ty('q_ty', (8, DH), x_dtype)
            q_mem_ty = mctx.data_ty('q_mem_ty', (8, DH), x_dtype)
            k_ty = mctx.data_ty('k_ty', (lc, DH), x_dtype)
            k_mem_ty = mctx.data_ty('k_mem_ty', (lc, DH * 2), x_dtype)
            v_ty = mctx.data_ty('v_ty', (lc, DH), x_dtype)
            v_mem_ty = mctx.data_ty('v_mem_ty', (lc, DH * 2), x_dtype)
            s_ty = mctx.data_ty('s_ty', (512, ), x_dtype)
            y_ty = mctx.data_ty('y_ty', (8, DH), y_dtype)
            o_ty = mctx.data_ty('o_ty', (8, DH), o_dtype)
            o_mem_ty = mctx.data_ty('o_mem_ty', (8, DH), o_dtype)
            L_ty = mctx.data_ty('L_ty', (16,), np.dtype[np.int32])

            mctx.parameter('lc', lc)
            mctx.parameter('DH', DH)
            mctx.parameter('Q_HEADS_PER_ROUND', Q_HEADS_PER_ROUND)
            mctx.func("attn_qk", [q_ty, q_ty, k_ty, k_ty, s_ty, s_ty, L_ty])
            mctx.func("attn_kv", [s_ty, s_ty, v_ty, v_ty, y_ty, o_ty, L_ty])

            for j in range(total_cols):
                memref.global_(f"q_{j}", T.memref(8 * DH, T.bf16()), sym_visibility = "public")
                memref.global_(f"kv_{j}", T.memref(lc * DH * 4, T.bf16()), sym_visibility = "public")
                memref.global_(f"o_{j}", T.memref(8 * DH, T.bf16()), sym_visibility = "public")

            attn_qk = CT[0][0]
            attn_kv = CT[1][0]

            L_qk_buffers = []
            L_kv_buffers = []
            L_qk_buffers.append(mctx.buffer(f"RTP_L_qk_{attn_qk.row}_{attn_qk.col}", attn_qk, L_ty, use_write_rtp = True))
            L_kv_buffers.append(mctx.buffer(f"RTP_L_kv_{attn_kv.row}_{attn_kv.col}", attn_kv, L_ty, use_write_rtp = True))
                    

            attn_cu(attn_qk, attn_kv, mctx)
            attn_memtile(MT[0], mctx)
            aux_memtile(MT[1], mctx)
        
            # q_in
            connect_dma(IT[1], MT[1], 0, 0)
            connect_dma(MT[1], attn_qk, 0, 0)

            # k_in
            connect_dma(IT[0], MT[0], 0, 0)
            connect_dma(MT[0], attn_qk, 0, 1)
            
            # v_in
            connect_dma(IT[0], MT[0], 1, 1)
            connect_dma(MT[0], attn_kv, 1, 1)
                    
            # o_out
            connect_dma(attn_kv, MT[1], 0, 1)
            connect_dma(MT[1], IT[1], 1, 0)

        print(ctx.module)

if __name__ == "__main__":
    flow_arch()
