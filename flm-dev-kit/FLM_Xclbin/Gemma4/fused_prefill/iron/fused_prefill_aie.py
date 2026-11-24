# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.extras import types as T
import aie.dialects.aie as aie
import aie.dialects.aiex as aiex
from aie.helpers.dialects.ext.scf import _for as range_
from aie.dialects import memref

from utils import connect_dma, generate_tile_map, mlir_context
from dequant_mm_tile import dequant_mm_memtile, dequant_mm_tile
from attn_tile import attn_cfg, attn_column

# Gemma4-E2B prefill, whole layer in one overlay.
#
# Gemma4 alternates 4 sliding-window layers to 1 full-attention layer, and the
# two use different head dims (512 full, 256 sliding), so the standalone flow
# needed *three* xclbins per layer -- dequant_mm, attention_DH_512, and
# attention_DH_256 -- and swapped between them constantly.  Fusing all three
# removes every one of those swaps.
#
# Column budget follows Qwen3VL/fused_prefill: dequant_mm keeps the six middle
# columns, attention keeps the two outermost, which also splits the mm weight
# and y traffic evenly across the two DDR channels (columns 0-3 / 4-7).  The
# difference here is that the two attention columns are not one CU: each is a
# self-contained attention with its own head dim, its own kernel object, and its
# own four cores.  That is what forces the solo-column memtile in attn_tile.py.
ATTN_FULL_COL = 0
MM_COLS = [1, 2, 3, 4, 5, 6]
ATTN_SWA_COL = 7
# The four memtiles that host an x row-broadcast; all six mm memtiles already use
# s2mm 1-4 for y and 5 for qw, so any four of them can spare s2mm 0.  Two per DDR
# channel.
X_COLS = [1, 2, 4, 5]
# The two mm columns with no x broadcast.  Their memtiles' s2mm 0 / mm2s 0 and
# locks 0..4 are dead weight, and mm2s 3/4/5 were never used at all, so each can
# host the q staging for one attention column -- the channels attention's own
# memtile no longer has once k and v arrive separately.  Q_COLS[i] serves
# ATTN_COLS[i]; 3 sits next to the full column and 6 next to the swa column, so
# neither q route has to cross the whole array.
Q_COLS = [c for c in MM_COLS if c not in X_COLS]   # -> [3, 6]
ATTN_OF_Q_COL = dict(zip(Q_COLS, ('full', 'swa')))


def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    mm_cols = len(MM_COLS)

    ##################### dequant_mm parameters #####################
    # m is the rows one core owns, and the mm half is 4 core rows deep, so a
    # host `i` step -- and therefore L_MIN in the runtime -- is 4 * m tokens.
    # m = 32 makes that 128 instead of 256, so an 84-token prompt pays for 44
    # rows of padding rather than 172.  The number of `i` steps (and with it the
    # weight traffic, which does not depend on m) is unchanged at these lengths;
    # this only removes MAC work on padding rows.
    m = 32
    # One k_slice per memtile load.  This used to be 512 (two slices), which
    # forced every gemm's K to be a multiple of 512 and made the per-layer-input
    # up projection -- K = PLI_D = 256 -- pad its weights and its x with a zero
    # half, doubling both for nothing.
    k = 256
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
    # (d, lq==lk, lk_mt).  lk_mt is the k block the host streams and the memtile
    # stages; the full-attention column has to run 64 rather than the standalone
    # design's 128 because at d=512 two ping-pong k/v buffers of 128 tokens would
    # need 512 KB on their own, the whole memtile.  Nothing in the microkernel
    # depended on 128 -- every hardcoded chunk count in the two attn_kernel
    # headers was lk_mt/lk, and they now read that from `n_kblk`.
    # pkt_base: the five packet ids the column owns on its shim mm2s 0 --
    # 0..3 are q for compute rows 0..3, 4 is k.  The two columns never share a
    # stream switch (every flow here is column-local), but they get disjoint
    # ranges anyway so a trace or a future cross-column route cannot collide.
    attn_geom = {
        'full': dict(d=512, lq=8,  lk=8,  lk_mt=64,  obj="attn_full.o",  func="attn_full",  pkt_base=0),
        'swa':  dict(d=256, lq=16, lk=16, lk_mt=128, obj="attn_swa.o",   func="attn_swa",   pkt_base=8),
    }

    q_dtype = np.dtype[bfloat16]
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

            mctx.parameter('m', m)
            mctx.parameter('k', k)
            mctx.parameter('n', n)
            mctx.parameter('m_slice', m_slice)
            mctx.parameter('k_slice', k_slice)
            mctx.parameter('n_slice', n_slice)
            mctx.parameter('qw_size', qw_size)
            mctx.parameter('qw_mem_size', qw_mem_size)

            mctx.func("dequant_mm_e", [mctx.y_ty, mctx.y_ty, mctx.x_ty, mctx.x_ty, mctx.qw_ty, mctx.qw_ty, mctx.w_ty, mctx.y_acc_ty, mctx.K_div_k_ty, mctx.K_div_k_ty, mctx.K_div_k_ty])
            mctx.func("dequant_mm_o", [mctx.y_ty, mctx.y_ty, mctx.x_ty, mctx.x_ty, mctx.qw_ty, mctx.qw_ty, mctx.w_ty, mctx.y_acc_ty, mctx.K_div_k_ty, mctx.K_div_k_ty, mctx.K_div_k_ty])

            # ---- attention types -------------------------------------------
            # One shared module, two attention variants, and mlir_context rejects
            # a duplicate name -- so every attention type is prefixed with its
            # variant and travels inside the attn_cfg rather than being looked up
            # off mctx the way the standalone designs did.
            L_ty = mctx.data_ty('L_ty', (8,), np.dtype[np.int32])

            cfgs = {}
            for p, g in attn_geom.items():
                d, lq, lk, lk_mt = g['d'], g['lq'], g['lk'], g['lk_mt']
                # A round is one lq slice per compute row.
                lq_mt = total_rows * lq

                q_ty = mctx.data_ty(f'{p}_q_ty', (lq, d), q_dtype)
                in_ty = mctx.data_ty(f'{p}_in_ty', (lk, d), q_dtype)
                s_ty = mctx.data_ty(f'{p}_s_ty', (lq, lk_mt), q_dtype)
                m_ty = mctx.data_ty(f'{p}_m_ty', (lq, lk), q_dtype)
                ay_ty = mctx.data_ty(f'{p}_attn_y_ty', (lq, d), attn_y_dtype)
                o_ty = mctx.data_ty(f'{p}_o_ty', (64, ), o_dtype)
                q_mem_ty = mctx.data_ty(f'{p}_q_mem_ty', (lq_mt, d), q_dtype)
                in_mem_ty = mctx.data_ty(f'{p}_in_mem_ty', (lk_mt * 2, d), q_dtype)
                o_mem_ty = mctx.data_ty(f'{p}_o_mem_ty', (lq * total_rows, d), o_dtype)

                fn = mctx.func(g['func'], [o_ty, o_ty, q_ty, in_ty, in_ty, s_ty,
                                           m_ty, ay_ty, L_ty, L_ty, L_ty,
                                           T.i32(), T.i32()])

                cfgs[p] = attn_cfg(
                    name=p, kernel_obj=g['obj'], func=fn,
                    d=d, lq=lq, lk=lk, lk_mt=lk_mt,
                    q_ty=q_ty, in_ty=in_ty, s_ty=s_ty, m_ty=m_ty, y_ty=ay_ty,
                    o_ty=o_ty, q_mem_ty=q_mem_ty, in_mem_ty=in_mem_ty,
                    o_mem_ty=o_mem_ty, pkt_base=g['pkt_base'],
                )

            for j in (ATTN_FULL_COL, ATTN_SWA_COL):
                memref.global_(f"q_in_{j}", T.memref(64 * 512, T.bf16()), sym_visibility="public")
                memref.global_(f"o_out_{j}", T.memref(64 * 512, T.bf16()), sym_visibility="public")

            mm_core = [[CT[i][MM_COLS[j]] for j in range(mm_cols)] for i in range(total_rows)]

            ##################### dequant_mm #####################
            K_div_k_buffers = []
            rtp_activate_buffers = []
            q_type_buffers = []

            for i in range(total_rows):
                for j in range(mm_cols):
                    K_div_k_buffers.append(mctx.buffer(f"K_div_k_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp=True))
                    rtp_activate_buffers.append(mctx.buffer(f"rtp_act_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp=True))
                    # 0 = q4_1, otherwise the weights arrive already bf16 and the
                    # core copies them into w_buffer instead of unpacking nibbles.
                    # Lets the per-layer-input projections, which are not quantised,
                    # run on this overlay rather than forcing a context switch to
                    # the image encoder's mm overlay twice per layer.
                    q_type_buffers.append(mctx.buffer(f"q_type_{mm_core[i][j].row}_{mm_core[i][j].col}", mm_core[i][j], K_div_k_ty, use_write_rtp=True))

            for j in range(mm_cols):
                for i in range(total_rows // 2):
                    dequant_mm_tile(mm_core[2 * i][j], mm_core[2 * i + 1][j], mctx)
                col_j = MM_COLS[j]
                dequant_mm_memtile(MT[col_j], mctx, y_pair_2x2=True,
                                   host_x=col_j in X_COLS,
                                   q_cfg=(cfgs[ATTN_OF_Q_COL[col_j]]
                                          if col_j in Q_COLS else None))

            # w_in: column j's shim and memtile feed that column's cores.  Only the
            # even core of each row pair takes the quantised tile; the odd one reads
            # its neighbour's dequantised w_buffer.
            for j in range(mm_cols):
                connect_dma(IT[MM_COLS[j]], MT[MM_COLS[j]], 1, 5)
                for i in range(total_rows // 2):
                    connect_dma(MT[MM_COLS[j]], mm_core[2 * i][j], 2, 1)

            # x_in: one broadcaster per core row, hosted by the X_COLS memtiles.
            for i in range(total_rows):
                connect_dma(IT[X_COLS[i]], MT[X_COLS[i]], 0, 0)
                for j in range(mm_cols):
                    connect_dma(MT[X_COLS[i]], mm_core[i][j], 0, 0)

            # y_out: 24 core lanes over 6 memtiles.  Memtile g = 3a + b collects the
            # 2x2 tile of token-blocks {2a, 2a+1} x N-slabs {2b, 2b+1}, in lane order
            # rb * 2 + sl so it matches the y_pair_2x2 drain BD.
            for a in range(2):
                for b in range(3):
                    g = 3 * a + b
                    for rb in range(2):
                        for sl in range(2):
                            connect_dma(mm_core[2 * a + rb][2 * b + sl], MT[MM_COLS[g]], 0, 1 + rb * 2 + sl)

            for j in range(mm_cols):
                connect_dma(MT[MM_COLS[j]], IT[MM_COLS[j]], 1, 0)

            ##################### attention #####################
            for (p, col), q_col in zip((('full', ATTN_FULL_COL), ('swa', ATTN_SWA_COL)), Q_COLS):
                cores = [CT[i][col] for i in range(total_rows)]
                for c in cores:
                    mctx.buffer(f"L_begin_attn_{p}_core_{c.row}_{c.col}", c, L_ty, use_write_rtp=True)
                    mctx.buffer(f"L_end_attn_{p}_core_{c.row}_{c.col}", c, L_ty, use_write_rtp=True)
                    mctx.buffer(f"window_size_attn_{p}_core_{c.row}_{c.col}", c, L_ty, use_write_rtp=True)
                attn_column(IT[col], MT[col], cores, cfgs[p], mctx,
                            q_shim=IT[q_col], q_memtile=MT[q_col])

        print(ctx.module)


if __name__ == "__main__":
    flow_arch()
