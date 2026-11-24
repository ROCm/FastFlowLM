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
from q4k_mm import lmhead_column

def flow_arch(model_type: str = "GEMMA3_1B"):
    
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    if model_type == "GEMMA3_1B":
        K = 1152
    elif model_type == "GEMMA3_270M":
        K = 640
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    M = (262144 + 4096 - 1) // 4096 * 4096 // 4
    m = 16
    k = 128
    assert M % (m * total_rows) == 0

    w_dtype = np.dtype[np.uint32]
    x_dtype = np.dtype[bfloat16]
    y_dtype = np.dtype[bfloat16]

    w_tile_size = int(m * k * 5 / 8 / 4) # unit is 4 byte
    w_mem_size = int(M * K * 5 / 8 / 4)

    K_div_k = K // k
    M_div_m = M // m // total_rows // total_cols

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            x_ty = mctx.data_ty('x_ty', (K, 1), x_dtype)
            w_ty = mctx.data_ty('w_ty', (w_tile_size, 1), w_dtype)
            w_mem_ty = mctx.data_ty('w_mem_ty', (w_tile_size * total_rows, 1), w_dtype)
            y_ty = mctx.data_ty('y_ty', (m, 1), y_dtype)
            y_mem_ty = mctx.data_ty('y_mem_ty', (m * total_rows, 1), y_dtype)

            mctx.parameter('K_div_k', K_div_k)
            mctx.parameter('M_div_m', M_div_m)
            mctx.parameter('w_tile_size', w_tile_size)  
            mctx.parameter('m', m)

            mctx.func("qmm", [mctx.x_ty, mctx.x_ty, mctx.w_ty, mctx.w_ty, mctx.y_ty, mctx.y_ty, T.i32(), T.i32()])

            # lmhead_core_0 = [CT[i][0] for i in range(total_rows)] 
            # # lmhead_core_1 = [CT[i][1] for i in range(total_rows)] 
            # lmhead_column(lmhead_core_0, MT[0], IT[0], mctx)
            # # lmhead_column(lmhead_core_1, MT[1], IT[1], mctx)

            lmhead_core_0 = [CT[i][0] for i in range(total_rows)] 
            lmhead_core_1 = [CT[i][1] for i in range(total_rows)] 
            lmhead_core_2 = [CT[i][2] for i in range(total_rows)] 
            lmhead_core_3 = [CT[i][3] for i in range(total_rows)] 
            lmhead_core_4 = [CT[i][4] for i in range(total_rows)] 
            lmhead_core_5 = [CT[i][5] for i in range(total_rows)] 
            lmhead_core_6 = [CT[i][6] for i in range(total_rows)] 
            lmhead_core_7 = [CT[i][7] for i in range(total_rows)] 
            # # lmhead_memtile(MT[0], mctx)
            # for i in range(total_rows):
            #     lmhead_tile(lmhead_core_0[i], mctx)

            # connect_dma(IT[0], MT[0], 0, 0)
            # connect_dma(MT[0], lmhead_core_0[0], 0, 0)
            # connect_dma(MT[0], lmhead_core_0[1], 0, 0)
            # connect_dma(MT[0], lmhead_core_0[2], 0, 0)
            # connect_dma(MT[0], lmhead_core_0[3], 0, 0)

            # connect_dma(IT[0], MT[0], 1, 1)
            # connect_dma(MT[0], lmhead_core_0[0], 1, 1)
            # connect_dma(MT[0], lmhead_core_0[1], 2, 1)
            # connect_dma(MT[0], lmhead_core_0[2], 3, 1)
            # connect_dma(MT[0], lmhead_core_0[3], 4, 1)

            # connect_dma(lmhead_core_0[0], MT[0], 0, 2)
            # connect_dma(lmhead_core_0[1], MT[0], 0, 3)
            # connect_dma(lmhead_core_0[2], MT[0], 0, 4)
            # connect_dma(lmhead_core_0[3], MT[0], 0, 5)

            # connect_dma(MT[0], IT[0], 5, 0)

            lmhead_column(lmhead_core_0, MT[0], IT[0], IT[0], mctx)
            lmhead_column(lmhead_core_1, MT[1], IT[1], IT[0], mctx)
            lmhead_column(lmhead_core_2, MT[2], IT[2], IT[0], mctx)
            lmhead_column(lmhead_core_3, MT[3], IT[3], IT[0], mctx)
            lmhead_column(lmhead_core_4, MT[4], IT[4], IT[0], mctx)
            lmhead_column(lmhead_core_5, MT[5], IT[5], IT[0], mctx)
            lmhead_column(lmhead_core_6, MT[6], IT[6], IT[0], mctx)
            lmhead_column(lmhead_core_7, MT[7], IT[7], IT[0], mctx)

        print(ctx.module)

if __name__ == '__main__':

    model_type = "LLAMA_3B"
    if len(sys.argv) < 2:
        print("Usage: python qmm_aie.py <model_type>")
        print("model_type: GEMMA3_4B")
        print("default model_type: GEMMA3_4B")
    else:
        model_type = sys.argv[1]
    flow_arch(model_type)