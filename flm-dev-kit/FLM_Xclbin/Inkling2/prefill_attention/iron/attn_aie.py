# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

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
from attn_tile import attn_ctbuf, attn_cttile, attn_memtile_even, attn_memtile_odd

def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4
    d = 128
    lq = 16
    lk = 16
    lk_mt = 128
    lq_mt = 64
    lq_ct = 32
    nr = 16                     # relative-bias features per head
    qp = lq_ct * d + lq_ct * nr + lq_ct   # query pack of one core row: q tiles, rel features, sinks
    nwin = 9                    # bias windows per round (distance blocks 8 .. 0)
    win = 160                   # rows per window
    tab_rows = 1280             # bias table rows (reversed distance, zero padded)
    kv_tile = lk * d * 9 // 16  # one 16-key K or V tile as host-packed bfp16 (2304 bytes), in bf16 units

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
            qp_ty = mctx.data_ty('qp_ty', (qp,), q_dtype)
            in_ty = mctx.data_ty('in_ty', (kv_tile,), k_dtype)
            q_mem_ty = mctx.data_ty('q_mem_ty', (4 * qp,), q_dtype)
            t_mem_ty = mctx.data_ty('t_mem_ty', (2 * tab_rows * 8,), q_dtype)
            w_ty = mctx.data_ty('w_ty', (2 * win * 8,), q_dtype)
            p_ty = mctx.data_ty('p_ty', (lq * win,), q_dtype)
            kv_mem_ty = mctx.data_ty('kv_mem_ty', (lk_mt // lk * kv_tile,), k_dtype)

            o_ty = mctx.data_ty('o_ty', (64, ), o_dtype)
            o_mem_ty = mctx.data_ty('o_mem_ty', (lq * 4, d), o_dtype)
            s_ty = mctx.data_ty('s_ty', (lq, lk_mt), v_dtype)   
            m_ty = mctx.data_ty('m_ty', (lq, lk), v_dtype)
            y_ty = mctx.data_ty('y_ty', (lq, d), y_dtype)
            L_ty = mctx.data_ty('L_ty', (8,), np.dtype[np.int32])

            mctx.parameter('lq', lq)
            mctx.parameter('lk', lk)
            mctx.parameter('lk_mt', lk_mt)
            mctx.parameter('lq_mt', lq_mt)
            mctx.parameter('lq_ct', lq_ct)
            mctx.parameter('d', d)
            mctx.parameter('kv_tile', kv_tile)
            mctx.parameter('nr', nr)
            mctx.parameter('qp', qp)
            mctx.parameter('nwin', nwin)
            mctx.parameter('win', win)
            mctx.parameter('tab_rows', tab_rows)
            mctx.func("attn", [o_ty, o_ty, qp_ty, in_ty, in_ty, in_ty, in_ty, s_ty, m_ty, y_ty, w_ty, w_ty, p_ty, L_ty, L_ty, L_ty, T.i32(), T.i32()])

            for j in range(total_cols):
                memref.global_(f"q_in_{j}", T.memref(lq_mt * 2 * d, T.bf16()), sym_visibility = "public")
                memref.global_(f"o_out_{j}", T.memref(lq_mt * 2 * d, T.bf16()), sym_visibility = "public")

            attn_core = [[CT[i][j] for j in range(total_cols)] for i in range(total_rows)]

            L_begin_buffers = []
            L_end_buffers = []
            window_size_buffers = []
            for i in range(total_rows):
                for j in range(total_cols):
                    L_begin_buffers.append(mctx.buffer(f"L_begin_attn_core_{attn_core[i][j].row}_{attn_core[i][j].col}", attn_core[i][j], L_ty, use_write_rtp = True))
                    L_end_buffers.append(mctx.buffer(f"L_end_attn_core_{attn_core[i][j].row}_{attn_core[i][j].col}", attn_core[i][j], L_ty, use_write_rtp = True))
                    window_size_buffers.append(mctx.buffer(f"window_size_attn_core_{attn_core[i][j].row}_{attn_core[i][j].col}", attn_core[i][j], L_ty, use_write_rtp = True))

            in_bufs = [[attn_ctbuf(attn_core[i][j], mctx) for j in range(total_cols)] for i in range(total_rows)]
            for j in range(total_cols):
                for i in range(total_rows):
                    attn_cttile(attn_core[i][j], mctx, in_bufs[i][j], in_bufs[i ^ 1][j])
            
            for j in range(total_cols):
                if j & 0x1 == 0:
                    attn_memtile_even(MT[j], mctx)
                else:
                    attn_memtile_odd(MT[j], mctx)
        
            # query packs; the bias table goes from the free shim 3 to every even memtile
            for j in range(4):
                connect_dma(IT[j * 2], MT[j * 2], 0, 0)
                connect_dma(IT[3], MT[j * 2], 0, 1)
                for i in range(4):
                    connect_dma(MT[j * 2], attn_core[i][j * 2], i, 0)
                    connect_dma(MT[j * 2], attn_core[i][j * 2 + 1], i, 0)

            # sharing k and v between 3 CUs, the last CU receives his own k and v
            # k_in & v_in
            for j in range(3):
                connect_dma(IT[1], MT[2 * j + 1], 0, 0)
                connect_dma(IT[1], MT[2 * j + 1], 1, 1)
                for i in range(total_rows):
                    connect_dma(MT[2 * j + 1], attn_core[i][2 * j], i % 2, 1)
                    connect_dma(MT[2 * j + 1], attn_core[i][2 * j + 1], i % 2, 1)
            
            connect_dma(IT[7], MT[7], 0, 0)
            connect_dma(IT[7], MT[7], 1, 1)
            for i in range(total_rows):
                connect_dma(MT[7], attn_core[i][6], i % 2, 1)
                connect_dma(MT[7], attn_core[i][7], i % 2, 1)
    
            # o_out
            for j in range(4):
                connect_dma(attn_core[0][2 * j], MT[2 * j], 0, 2)
                connect_dma(attn_core[0][2 * j + 1], MT[2 * j], 0, 3)
                connect_dma(attn_core[1][2 * j], MT[2 * j], 0, 4)
                connect_dma(attn_core[1][2 * j + 1], MT[2 * j], 0, 5)

                connect_dma(attn_core[2][2 * j], MT[2 * j + 1], 0, 2)
                connect_dma(attn_core[2][2 * j + 1], MT[2 * j + 1], 0, 3)
                connect_dma(attn_core[3][2 * j], MT[2 * j + 1], 0, 4)
                connect_dma(attn_core[3][2 * j + 1], MT[2 * j + 1], 0, 5)
            for j in range(total_cols):
                connect_dma(MT[j], IT[j], 5, 0)

        print(ctx.module)

if __name__ == "__main__":
    flow_arch()
