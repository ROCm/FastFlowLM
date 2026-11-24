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
    QWEN3_5_0_8B = auto()
    QWEN3_5_2B = auto()
    QWEN3_5_4B = auto()
    QWEN3_5_9B = auto()
    HY_MEMORY_30B_A3B = auto()

ModelArchNames: dict[ModelArch, str] = {
    ModelArch.QWEN3_5_0_8B: "QWEN3_5_0_8B",
    ModelArch.QWEN3_5_2B  : "QWEN3_5_2B",
    ModelArch.QWEN3_5_4B  : "QWEN3_5_4B",
    ModelArch.QWEN3_5_9B  : "QWEN3_5_9B",
    ModelArch.HY_MEMORY_30B_A3B: "HY_MEMORY_30B_A3B"
}

# The dataflow only depends on D; the vocab size lives in the kernel and the
# host sequence (kernel/qmm.cc).
DIM_TABLE = {
    ModelArch.QWEN3_5_0_8B: 1024,
    ModelArch.QWEN3_5_2B:   2048,
    ModelArch.QWEN3_5_4B:   2560,
    ModelArch.QWEN3_5_9B:   4096,
    ModelArch.HY_MEMORY_30B_A3B: 2048
}


def lm_head_top(model_name: str):
    
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4
    
    Q4NX_ROW_BLOCK_SIZE: int = 32
    Q4NX_COL_BLOCK_SIZE: int = 256
    
    MODEL_DIM: int = DIM_TABLE[ModelArch[model_name]]

    W_BLOCK_SIZE = int(Q4NX_COL_BLOCK_SIZE * Q4NX_ROW_BLOCK_SIZE * 8.5 // 8 // 2)

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
            y_ty = mctx.data_ty('y_ty', (Q4NX_ROW_BLOCK_SIZE, 1), y_dtype)
            # A memtile weight slot holds one k-block for each of the two cores its
            # group feeds, not one for every core in the column: the column's weights
            # arrive on two shim channels now. See lm_head_memtile.
            w_mem_ty = mctx.data_ty('w_mem_ty', (W_BLOCK_SIZE * 2, 1), w_dtype)
            y_mem_ty = mctx.data_ty('y_mem_ty', (Q4NX_ROW_BLOCK_SIZE * total_rows, 1), y_dtype)

            mctx.parameter('W_BLOCK_SIZE', W_BLOCK_SIZE)  
            mctx.parameter('Q4NX_ROW_BLOCK_SIZE', Q4NX_ROW_BLOCK_SIZE)

            mctx.func("qmm", [mctx.y_ty, mctx.y_ty, mctx.w_ty, mctx.w_ty, mctx.x_ty])

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
    model_type = "QWEN3_5_0_8B"
    if len(sys.argv) != 2:
        print("Usage: python flowArch_aie.py <model_type>, Options: QWEN3_5_0_8B, QWEN3_5_2B, QWEN3_5_4B, QWEN3_5_9B, HY_MEMORY_30B_A3B")
        print("Use default model type: QWEN3_5_0_8B")
    else:
        model_type = sys.argv[1]
    lm_head_top(model_type)