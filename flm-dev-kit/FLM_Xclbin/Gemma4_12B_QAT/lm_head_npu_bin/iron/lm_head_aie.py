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
from q4k_mm import lm_head_column, W_PKT_ID

from enum import IntEnum, auto

class ModelArch(IntEnum):
    GEMMA4_E2B = auto()
    GEMMA4_E4B = auto()
    GEMMA4_12B = auto()

ModelArchNames: dict[ModelArch, str] = {
    ModelArch.GEMMA4_E2B: "GEMMA4_E2B",
    ModelArch.GEMMA4_E4B: "GEMMA4_E4B",
    ModelArch.GEMMA4_12B: "GEMMA4_12B"
}

DIM_TABLE = {
    ModelArch.GEMMA4_E2B: 1536,
    ModelArch.GEMMA4_E4B: 2560,
    ModelArch.GEMMA4_12B: 3840
}

def lm_head_top(model_name: str):

    # kernel/qmm.cc here is Q8_0 with a 16x256 block; E2B/E4B use Q4_K with 32x256
    # and live in ../../Gemma4/lm_head_npu_bin.
    assert ModelArch[model_name] == ModelArch.GEMMA4_12B, \
        "this lm_head is Q8_0 / Gemma4-12B only, got " + model_name

    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    Q8NX_ROW_BLOCK_SIZE: int = 16
    Q8NX_COL_BLOCK_SIZE: int = 256

    MODEL_DIM: int = DIM_TABLE[ModelArch[model_name]]
    assert MODEL_DIM % Q8NX_COL_BLOCK_SIZE == 0, "MODEL_DIM must be a whole number of Q8NX column blocks"

    # Q8_0 is 8.5 bits/weight (8 bit quant + one bf16 scale per 32-wide group),
    # i.e. 17/16 byte per weight -> a 16x256 block is 4352 B. The trailing // 2
    # converts bytes to bf16 elements, which is how the w buffers are typed.
    W_BLOCK_SIZE = Q8NX_ROW_BLOCK_SIZE * Q8NX_COL_BLOCK_SIZE * 17 // 16 // 2

    w_dtype = np.dtype[bfloat16]
    x_dtype = np.dtype[bfloat16]
    y_dtype = np.dtype[bfloat16]

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            x_ty = mctx.data_ty('x_ty', (MODEL_DIM, 2), x_dtype)
            w_ty = mctx.data_ty('w_ty', (W_BLOCK_SIZE, 1), w_dtype)
            y_ty = mctx.data_ty('y_ty', (Q8NX_ROW_BLOCK_SIZE, 1), y_dtype)
            # A memtile weight slot holds one k-block for each of the two cores its
            # group feeds, not one for every core in the column: the column's weights
            # arrive on two shim channels now. See lm_head_memtile.
            w_mem_ty = mctx.data_ty('w_mem_ty', (W_BLOCK_SIZE * 2, 1), w_dtype)
            y_mem_ty = mctx.data_ty('y_mem_ty', (Q8NX_ROW_BLOCK_SIZE * total_rows, 1), y_dtype)
            rtp_ty = mctx.data_ty('rtp_ty', (32, 1), np.dtype[np.int32])

            mctx.parameter('W_BLOCK_SIZE', W_BLOCK_SIZE)
            mctx.parameter('Q8NX_ROW_BLOCK_SIZE', Q8NX_ROW_BLOCK_SIZE)

            mctx.func("qmm", [mctx.y_ty, mctx.y_ty, mctx.w_ty, mctx.w_ty, mctx.x_ty, mctx.rtp_ty])

            lm_head_core_0 = [CT[i][0] for i in range(total_rows)] 
            lm_head_core_1 = [CT[i][1] for i in range(total_rows)] 
            lm_head_core_2 = [CT[i][2] for i in range(total_rows)] 
            lm_head_core_3 = [CT[i][3] for i in range(total_rows)] 
            lm_head_core_4 = [CT[i][4] for i in range(total_rows)] 
            lm_head_core_5 = [CT[i][5] for i in range(total_rows)] 
            lm_head_core_6 = [CT[i][6] for i in range(total_rows)] 
            lm_head_core_7 = [CT[i][7] for i in range(total_rows)] 

            # Column 0's shim MM2S 0 is shared with the x broadcast, so its second
            # weight stream is packet-switched; the rest get a plain flow.
            lm_head_column(lm_head_core_0, MT[0], IT[0], IT[0], mctx, w_pkt_id = W_PKT_ID)
            lm_head_column(lm_head_core_1, MT[1], IT[1], IT[0], mctx)
            lm_head_column(lm_head_core_2, MT[2], IT[2], IT[0], mctx)
            lm_head_column(lm_head_core_3, MT[3], IT[3], IT[0], mctx)
            lm_head_column(lm_head_core_4, MT[4], IT[4], IT[0], mctx)
            lm_head_column(lm_head_core_5, MT[5], IT[5], IT[0], mctx)
            lm_head_column(lm_head_core_6, MT[6], IT[6], IT[0], mctx)
            lm_head_column(lm_head_core_7, MT[7], IT[7], IT[0], mctx)

        print(ctx.module)

if __name__ == '__main__':
    model_type = "GEMMA4_12B"
    if len(sys.argv) != 2:
        print("Usage: python flowArch_aie.py <model_type>, Options: GEMMA4_12B")
        print("Use default model type: GEMMA4_12B")
    else:
        model_type = sys.argv[1]
    lm_head_top(model_type)