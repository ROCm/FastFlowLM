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
from q4k_mm import lm_head_column, W_PKT_ID, UNITS_PER_SLOT_A

from enum import IntEnum, auto

class ModelArch(IntEnum):
    QWEN3_8_27B = auto()

ModelArchNames: dict[ModelArch, str] = {
    ModelArch.QWEN3_8_27B: "QWEN3_8_27B",
}

DIM_TABLE = {
    ModelArch.QWEN3_8_27B: 5120,
}

# Token rows per dispatch -- the verify phase of speculative decoding scores 8
# candidates at once. Must match L_TOKENS in kernel/qmm.cc. The L = 1 design is
# ../../lm_head_npu_bin.
L_TOKENS = 8

# Lanes in the x stream: the L token rows plus one carrying model.norm.weight.
# The kernel normalises on the core, and the per-column half of RMSNorm has to
# meet x elementwise, so the weight travels with x rather than living in a
# buffer of its own -- a one-shot fill BD is what broke the second dispatch
# (see README.md). Must match X_LANES in kernel/qmm.cc.
X_LANES = L_TOKENS + 1

VOCAB_SIZE  = 248320
M_PER_ROUND = 32 * 32


def lm_head_top(model_name: str):
    
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4
    
    Q4NX_ROW_BLOCK_SIZE: int = 32
    Q4NX_COL_BLOCK_SIZE: int = 256
    
    MODEL_DIM: int = DIM_TABLE[ModelArch[model_name]]

    W_BLOCK_SIZE = int(Q4NX_COL_BLOCK_SIZE * Q4NX_ROW_BLOCK_SIZE * 8.5 // 8 // 2)
    VOCAB_SIZE_PADDED = (VOCAB_SIZE + M_PER_ROUND - 1) // M_PER_ROUND * M_PER_ROUND

    w_dtype = np.dtype[bfloat16]
    x_dtype = np.dtype[bfloat16]
    y_dtype = np.dtype[bfloat16]

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            # x on a core is now one column block for all L tokens, not a whole
            # resident row: 8 * 5120 * 2 B would be 80 KB against a 64 KB core
            # memory. See the header comment in kernel/qmm.cc.
            #
            # X_LANES, not L_TOKENS: lane L carries this column block's slice of
            # model.norm.weight, which the kernel folds into x as it loads it.
            # 4.5 KB per ping/pong buffer instead of 4.
            x_ty = mctx.data_ty('x_ty', (X_LANES * Q4NX_COL_BLOCK_SIZE, 1), x_dtype)
            w_ty = mctx.data_ty('w_ty', (W_BLOCK_SIZE, 1), w_dtype)
            y_ty = mctx.data_ty('y_ty', (L_TOKENS * Q4NX_ROW_BLOCK_SIZE, 1), y_dtype)
            # A memtile weight slot holds one k-block for each of the two cores its
            # group feeds, not one for every core in the column: the column's weights
            # arrive on two shim channels now. See lm_head_memtile.
            w_mem_ty = mctx.data_ty('w_mem_ty', (W_BLOCK_SIZE * 2, 1), w_dtype)
            # Group a stages several units per slot so that its S2MM ring, which
            # also carries x and so must tile exactly one round, stays short
            # enough for the memtile's per-parity BD bank. See UNITS_PER_SLOT_A.
            wa_mem_ty = mctx.data_ty('wa_mem_ty', (W_BLOCK_SIZE * 2 * UNITS_PER_SLOT_A, 1), w_dtype)
            y_mem_ty = mctx.data_ty('y_mem_ty', (L_TOKENS * Q4NX_ROW_BLOCK_SIZE * total_rows, 1), y_dtype)
            # The column's whole x, [k_block][lane][256], replayed per round.
            # 9 * 5120 * 2 B = 90 KB of the memtile's 512 KB.
            x_mem_ty = mctx.data_ty('x_mem_ty', (X_LANES * MODEL_DIM, 1), x_dtype)

            mctx.parameter('W_BLOCK_SIZE', W_BLOCK_SIZE)
            mctx.parameter('Q4NX_ROW_BLOCK_SIZE', Q4NX_ROW_BLOCK_SIZE)
            mctx.parameter('L_TOKENS', L_TOKENS)
            mctx.parameter('X_LANES', X_LANES)
            mctx.parameter('X_BLOCK_SIZE', X_LANES * Q4NX_COL_BLOCK_SIZE)
            mctx.parameter('K_BLOCKS', MODEL_DIM // Q4NX_COL_BLOCK_SIZE)
            mctx.parameter('ROUNDS', VOCAB_SIZE_PADDED // M_PER_ROUND)

            mctx.func("qmm", [mctx.y_ty, mctx.y_ty, mctx.w_ty, mctx.w_ty, mctx.x_ty, mctx.x_ty])

            lm_head_core_0 = [CT[i][0] for i in range(total_rows)] 
            lm_head_core_1 = [CT[i][1] for i in range(total_rows)] 
            lm_head_core_2 = [CT[i][2] for i in range(total_rows)] 
            lm_head_core_3 = [CT[i][3] for i in range(total_rows)] 
            lm_head_core_4 = [CT[i][4] for i in range(total_rows)] 
            lm_head_core_5 = [CT[i][5] for i in range(total_rows)] 
            lm_head_core_6 = [CT[i][6] for i in range(total_rows)] 
            lm_head_core_7 = [CT[i][7] for i in range(total_rows)] 

            # Column 0's shim MM2S 0 keeps its packet id; the rest get a plain flow.
            lm_head_column(lm_head_core_0, MT[0], IT[0], mctx, w_pkt_id = W_PKT_ID)
            lm_head_column(lm_head_core_1, MT[1], IT[1], mctx)
            lm_head_column(lm_head_core_2, MT[2], IT[2], mctx)
            lm_head_column(lm_head_core_3, MT[3], IT[3], mctx)
            lm_head_column(lm_head_core_4, MT[4], IT[4], mctx)
            lm_head_column(lm_head_core_5, MT[5], IT[5], mctx)
            lm_head_column(lm_head_core_6, MT[6], IT[6], mctx)
            lm_head_column(lm_head_core_7, MT[7], IT[7], mctx)

        print(ctx.module)

if __name__ == '__main__':
    model_type = "QWEN3_8_27B"
    if len(sys.argv) != 2:
        print("Usage: python lm_head_aie.py <model_type>, Options: QWEN3_8_27B")
        print("Use default model type: QWEN3_8_27B")
    else:
        model_type = sys.argv[1]
    lm_head_top(model_type)