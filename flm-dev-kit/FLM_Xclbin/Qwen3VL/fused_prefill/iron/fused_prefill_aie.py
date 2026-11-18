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
from attn_tile import attn_cttile, attn_memtile_even, attn_memtile_odd

# Qwen3-VL prefill, both halves in one array so the runtime never has to swap
# xclbins between a projection and the attention that follows it.  On the
# measured 256x256-image case that swap cost ~5 ms per layer, about a quarter of
# the whole prefill.
#
# dequant_mm keeps 6 of the 8 columns (see the 6-column notes in
# dequant_mm_Q4_1_6col); attention keeps 2, i.e. a single CU, so it serialises
# what used to be 4 head groups.
#
# Columns 0-3 and 4-7 each have their own DDR channel, so the split has to be
# even across that boundary or the busier half throttles the whole array.  Giving
# attention the two outermost columns leaves mm with 1-6: three columns, and so
# half the weight and y traffic, on each channel.
MM_COLS = [1, 2, 3, 4, 5, 6]
# attn.cc derives its `col` argument as `ct_col & 0x1` and uses it to pick which
# 16 q tokens the core owns, so a CU has to be an (even, odd) physical pair --
# which 0 and 7 still are, even though they are no longer neighbours.
ATTN_COLS = [0, 7]
# The four memtiles that host an x row-broadcast.  Any four of the six will do
# (all six already use s2mm 1-4 for y and 5 for qw, leaving 0 free), so pick two
# per DDR channel rather than the first four.
X_COLS = [1, 2, 4, 5]
# Logical attention column that sources the shared k/v broadcast (physical 7).
# The standalone build broadcasts from an odd column too, so attn_memtile_odd is
# the right builder for it.
ATTN_KV_LANE = 1


def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    mm_cols = len(MM_COLS)
    attn_cols = len(ATTN_COLS)

    ##################### dequant_mm parameters #####################
    # m is the rows one core owns, and the mm half is 4 core rows deep, so a
    # host `i` step -- and therefore L_MIN in the runtime -- is 4 * m tokens.
    # m = 32 makes that 128 instead of 256: prompts of ~100 tokens stop paying
    # for 156 rows of padding.  The weight traffic per round is unchanged (it
    # does not depend on m), so this trades MAC work, not DMA.
    m = 32
    k = 512
    n = 64

    m_slice = 16
    k_slice = 256
    n_slice = 32
    y_slice = 64

    x_dtype = np.dtype[bfloat16]
    w_dtype = np.dtype[bfloat16]
    y_dtype = np.dtype[bfloat16]
    y_acc_dtype = np.dtype[np.float32]
    qw_dtype = np.dtype[np.uint32]

    # Q4_1 is 5 bits per weight: half a byte of quant plus a bf16 scale and a
    # bf16 min per 32-weight group.  The buffers are typed uint32, hence // 4.
    q4_1_bpw = 5
    qw_size = k_slice * n_slice * q4_1_bpw // 8 // 4
    qw_mem_size = k * n * q4_1_bpw // 8 // 4

    ##################### attention parameters #####################
    d = 128
    lq = 16
    lk = 16
    lk_mt = 128
    lq_mt = 64
    lq_ct = 32

    q_dtype = np.dtype[bfloat16]
    k_dtype = np.dtype[bfloat16]
    v_dtype = np.dtype[bfloat16]
    o_dtype = np.dtype[bfloat16]
    attn_y_dtype = np.dtype[np.float32]

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()

            # ---- dequant_mm types ------------------------------------------
            x_ty = mctx.data_ty('x_ty', (m_slice, 128), x_dtype)
            x_mem_ty = mctx.data_ty('x_mem_ty', (m, k), x_dtype)
            qw_ty = mctx.data_ty('qw_ty', (qw_size, 1), qw_dtype)
            w_ty = mctx.data_ty('w_ty', (n_slice, k_slice), w_dtype)
            qw_mem_ty = mctx.data_ty('qw_mem_ty', (qw_mem_size, 1), qw_dtype)
            y_ty = mctx.data_ty('y_ty', (y_slice,), y_dtype)
            y_acc_ty = mctx.data_ty('y_acc_ty', (m * n, ), y_acc_dtype)
            y_mem_ty = mctx.data_ty('y_mem_ty', (m, n * 4), y_dtype)
            K_div_k_ty = mctx.data_ty('K_div_k_ty', (16, ), np.dtype[np.int32])

            # ---- attention types -------------------------------------------
            q_ty = mctx.data_ty('q_ty', (lq_ct, d), q_dtype)
            in_ty = mctx.data_ty('in_ty', (lk, d), k_dtype)
            q_mem_ty = mctx.data_ty('q_mem_ty', (lq_mt * 2, d), q_dtype)
            in_mem_ty = mctx.data_ty('in_mem_ty', (lk_mt * 2, d), k_dtype)

            o_ty = mctx.data_ty('o_ty', (64, ), o_dtype)
            o_mem_ty = mctx.data_ty('o_mem_ty', (lq * 4, d), o_dtype)
            s_ty = mctx.data_ty('s_ty', (lq, lk_mt), v_dtype)
            m_ty = mctx.data_ty('m_ty', (lq, lk), v_dtype)
            # dequant_mm already owns the name `y_ty`; attn_tile.py reads this one.
            attn_y_ty = mctx.data_ty('attn_y_ty', (lq, d), attn_y_dtype)
            L_ty = mctx.data_ty('L_ty', (8,), np.dtype[np.int32])

            mctx.parameter('m', m)
            mctx.parameter('k', k)
            mctx.parameter('n', n)
            mctx.parameter('m_slice', m_slice)
            mctx.parameter('k_slice', k_slice)
            mctx.parameter('n_slice', n_slice)
            mctx.parameter('qw_size', qw_size)
            mctx.parameter('qw_mem_size', qw_mem_size)

            mctx.parameter('lq', lq)
            mctx.parameter('lk', lk)
            mctx.parameter('lk_mt', lk_mt)
            mctx.parameter('lq_mt', lq_mt)
            mctx.parameter('lq_ct', lq_ct)
            mctx.parameter('d', d)

            mctx.func("dequant_mm_e", [mctx.y_ty, mctx.y_ty, mctx.x_ty, mctx.x_ty, mctx.qw_ty, mctx.qw_ty, mctx.w_ty, mctx.y_acc_ty, mctx.K_div_k_ty, mctx.K_div_k_ty])
            mctx.func("dequant_mm_o", [mctx.y_ty, mctx.y_ty, mctx.x_ty, mctx.x_ty, mctx.qw_ty, mctx.qw_ty, mctx.w_ty, mctx.y_acc_ty, mctx.K_div_k_ty, mctx.K_div_k_ty])
            mctx.func("attn", [o_ty, o_ty, q_ty, in_ty, in_ty, s_ty, m_ty, attn_y_ty, L_ty, L_ty, L_ty, T.i32(), T.i32()])

            for j in ATTN_COLS:
                memref.global_(f"q_in_{j}", T.memref(lq_mt * 2 * d, T.bf16()), sym_visibility = "public")
                memref.global_(f"o_out_{j}", T.memref(lq_mt * 2 * d, T.bf16()), sym_visibility = "public")

            mm_core = [[CT[i][MM_COLS[j]] for j in range(mm_cols)] for i in range(total_rows)]
            attn_core = [[CT[i][ATTN_COLS[j]] for j in range(attn_cols)] for i in range(total_rows)]

            ##################### dequant_mm #####################
            K_div_k_buffers = []
            rtp_activate_buffers = []

            for i in range(total_rows):
                for j in range(mm_cols):
                    K_div_k_buffers.append(mctx.buffer(f"K_div_k_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp = True))
                    rtp_activate_buffers.append(mctx.buffer(f"rtp_act_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp = True))

            for j in range(mm_cols):
                for i in range(total_rows // 2):
                    dequant_mm_tile(mm_core[2 * i][j], mm_core[2 * i + 1][j], mctx)
                dequant_mm_memtile(MT[MM_COLS[j]], mctx, y_pair_2x2 = True)

            # w_in: column j's shim and memtile feed that column's cores.  Only the
            # even core of each row pair takes the quantised tile; the odd one reads
            # its neighbour's dequantised w_buffer.
            for j in range(mm_cols):
                connect_dma(IT[MM_COLS[j]], MT[MM_COLS[j]], 1, 5)
                for i in range(total_rows // 2):
                    connect_dma(MT[MM_COLS[j]], mm_core[2 * i][j], 2, 1)

            # x_in: one broadcaster per core row, hosted by the X_COLS memtiles.
            # Those four then have s2mm 0 = x, 1-4 = y, 5 = qw, i.e. all six
            # channels; the other two leave s2mm 0 idle.
            for i in range(total_rows):
                connect_dma(IT[X_COLS[i]], MT[X_COLS[i]], 0, 0)
                for j in range(mm_cols):
                    connect_dma(MT[X_COLS[i]], mm_core[i][j], 0, 0)

            # y_out: 24 core lanes over 6 memtiles.  Memtile g = 3a + b collects the
            # 2x2 tile of token-blocks {2a, 2a+1} x N-slabs {2b, 2b+1}, in lane order
            # rb * 2 + s so it matches the y_pair_2x2 drain BD.
            for a in range(2):
                for b in range(3):
                    g = 3 * a + b
                    for rb in range(2):
                        for sl in range(2):
                            connect_dma(mm_core[2 * a + rb][2 * b + sl], MT[MM_COLS[g]], 0, 1 + rb * 2 + sl)

            for j in range(mm_cols):
                connect_dma(MT[MM_COLS[j]], IT[MM_COLS[j]], 1, 0)

            ##################### attention #####################
            L_begin_buffers = []
            L_chunks_buffers = []
            window_size_buffers = []
            for i in range(total_rows):
                for j in range(attn_cols):
                    L_begin_buffers.append(mctx.buffer(f"L_begin_attn_core_{attn_core[i][j].row}_{attn_core[i][j].col}", attn_core[i][j], L_ty, use_write_rtp = True))
                    L_chunks_buffers.append(mctx.buffer(f"L_chunks_attn_core_{attn_core[i][j].row}_{attn_core[i][j].col}", attn_core[i][j], L_ty, use_write_rtp = True))
                    window_size_buffers.append(mctx.buffer(f"window_size_attn_core_{attn_core[i][j].row}_{attn_core[i][j].col}", attn_core[i][j], L_ty, use_write_rtp = True))

            for j in range(attn_cols):
                for i in range(total_rows):
                    attn_cttile(attn_core[i][j], mctx)

            # The even/odd split is on the *logical* lane, which ATTN_COLS keeps in
            # step with the physical column parity.
            for j in range(attn_cols):
                if j & 0x1 == 0:
                    attn_memtile_even(MT[ATTN_COLS[j]], mctx)
                else:
                    attn_memtile_odd(MT[ATTN_COLS[j]], mctx)

            # q_in: the even column of the CU takes q for both its columns.
            connect_dma(IT[ATTN_COLS[0]], MT[ATTN_COLS[0]], 0, 0)
            connect_dma(IT[ATTN_COLS[0]], MT[ATTN_COLS[0]], 1, 1)
            for i in range(total_rows):
                connect_dma(MT[ATTN_COLS[0]], attn_core[i][0], i, 0)
                connect_dma(MT[ATTN_COLS[0]], attn_core[i][1], i, 0)

            # k_in & v_in: one memtile broadcasts to all 8 attention cores.
            kv_col = ATTN_COLS[ATTN_KV_LANE]
            connect_dma(IT[kv_col], MT[kv_col], 0, 0)
            connect_dma(IT[kv_col], MT[kv_col], 1, 1)
            for j in range(attn_cols):
                for i in range(total_rows):
                    connect_dma(MT[kv_col], attn_core[i][j], 0, 1)

            # o_out: rows 0-1 drain into the CU's even memtile, rows 2-3 into its
            # odd one; each memtile then pushes its own column's shim.
            connect_dma(attn_core[0][0], MT[ATTN_COLS[0]], 0, 2)
            connect_dma(attn_core[0][1], MT[ATTN_COLS[0]], 0, 3)
            connect_dma(attn_core[1][0], MT[ATTN_COLS[0]], 0, 4)
            connect_dma(attn_core[1][1], MT[ATTN_COLS[0]], 0, 5)

            connect_dma(attn_core[2][0], MT[ATTN_COLS[1]], 0, 2)
            connect_dma(attn_core[2][1], MT[ATTN_COLS[1]], 0, 3)
            connect_dma(attn_core[3][0], MT[ATTN_COLS[1]], 0, 4)
            connect_dma(attn_core[3][1], MT[ATTN_COLS[1]], 0, 5)

            for j in range(attn_cols):
                connect_dma(MT[ATTN_COLS[j]], IT[ATTN_COLS[j]], 5, 0)

        print(ctx.module)


if __name__ == "__main__":
    flow_arch()
