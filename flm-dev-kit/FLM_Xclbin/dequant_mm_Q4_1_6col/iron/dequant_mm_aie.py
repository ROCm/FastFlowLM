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

def flow_arch():
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4
    # Physical columns the mm half occupies.  The remaining two are left free for
    # the attention half of the fused overlay.
    MM_COLS = [0, 1, 2, 3, 4, 5]
    mm_cols = len(MM_COLS)

    m = 64
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
    # bf16 min per 32-weight group, i.e. 5120 B for a 32x256 block.  The buffers
    # are typed uint32, hence the extra // 4.
    q4_1_bpw = 5
    qw_size = k_slice * n_slice * q4_1_bpw // 8 // 4
    qw_mem_size = k * n * q4_1_bpw // 8 // 4
    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(4 + 2, 8)

            mctx = mlir_context()
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

            mctx.func("dequant_mm_e", [mctx.y_ty, mctx.y_ty, mctx.x_ty, mctx.x_ty, mctx.qw_ty, mctx.qw_ty, mctx.w_ty, mctx.y_acc_ty, mctx.K_div_k_ty, mctx.K_div_k_ty])
            mctx.func("dequant_mm_o", [mctx.y_ty, mctx.y_ty, mctx.x_ty, mctx.x_ty, mctx.qw_ty, mctx.qw_ty, mctx.w_ty, mctx.y_acc_ty, mctx.K_div_k_ty, mctx.K_div_k_ty])

            mm_core = [[CT[i][MM_COLS[j]] for j in range(mm_cols)] for i in range(total_rows)]
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

            # x_in: one broadcaster per core row, hosted by the first four mm
            # memtiles.  Those four then have s2mm 0 = x, 1-4 = y, 5 = qw, i.e. all
            # six channels; the other two leave s2mm 0 idle.
            for i in range(total_rows):
                connect_dma(IT[MM_COLS[i]], MT[MM_COLS[i]], 0, 0)
                for j in range(mm_cols):
                    connect_dma(MT[MM_COLS[i]], mm_core[i][j], 0, 0)

            # y_out: 24 core lanes over 6 memtiles.  Memtile g = 3a + b collects the
            # 2x2 tile of token-blocks {2a, 2a+1} x N-slabs {2b, 2b+1}, in lane order
            # rb * 2 + s so it matches the y_pair_2x2 drain BD above.
            for a in range(2):
                for b in range(3):
                    g = 3 * a + b
                    for rb in range(2):
                        for sl in range(2):
                            connect_dma(mm_core[2 * a + rb][2 * b + sl], MT[MM_COLS[g]], 0, 1 + rb * 2 + sl)

            for j in range(mm_cols):
                connect_dma(MT[MM_COLS[j]], IT[MM_COLS[j]], 1, 0)

        print(ctx.module)


if __name__ == "__main__":
    flow_arch()