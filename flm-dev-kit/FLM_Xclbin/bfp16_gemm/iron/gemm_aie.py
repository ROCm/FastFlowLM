# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""bfp16 x bfp16 GEMM with fp32 accumulation, bias and optional GELU, bf16 out (32 cores).

Y[M, N] = act(X[M, K] W[N, K]^T + b). X and W arrive pre-packed as bfp16 (bfp16ebs8, 9 B per 8 k).
The array works on rounds of 256 rows x 1024 columns: column c owns 128 columns, core row r owns 64 rows.
The two cores of a vertical pair (rows 2/3 and 4/5) hold the two 64-column halves (lo in the even core,
hi in the odd core) of the current 128 k chunk of their column's weights and both read both halves.

  weights  shim c MM2S0 -> memtile c S2MM0 (a whole N block panel, resident while all rows stream past)
           memtile c MM2S0 (lo) -> cores (2, c), (4, c) S2MM1; MM2S1 (hi) -> cores (3, c), (5, c) S2MM1
           The memtile weight path (buffer descriptors, locks, repeat counts) is programmed by the host
           sequence for each call; the design only reserves the buffers, locks and routes.
  x        shim r MM2S1 -> the 8 cores of core row 2 + r, S2MM0 (16 rows x 128 k per buffer)
  y        core (2 + r, c) MM2S0 -> memtile c S2MM(1 + r) -> memtile c MM2S2 -> shim c S2MM0

See host/seq_gen.hpp for the DDR layouts and the run-time part.
"""

import numpy as np
import sys

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
import aie.dialects.aie as aie

from utils import connect_dma, generate_tile_map, mlir_context
from gemm_tile import gemm_memtile, gemm_pair


def flow_arch():
    total_cols = 8
    total_rows = 4

    with mlir_mod_ctx() as ctx:
        @aie.device(aie.AIEDevice.npu2)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            mctx = mlir_context()
            mctx.data_ty('acc_ty', (8192,), np.dtype[np.float32])     # 64 rows x 128 columns fp32, bf16 y in place
            mctx.data_ty('x_ty', (2304,), np.dtype[np.uint8])         # 16 rows x 128 k bfp16
            mctx.data_ty('w_ty', (9344,), np.dtype[np.uint8])         # 64 columns x 128 k bfp16 + 64 bf16 bias
            mctx.data_ty('rtp_ty', (16,), np.dtype[np.int32])
            mctx.data_ty('y_mem_ty', (32768,), np.dtype[bfloat16])    # 4 core rows x 64 rows x 128 columns
            mctx.data_ty('panel_ty', (57344,), np.dtype[np.int32])    # 224 KB weight panel

            args = [mctx.acc_ty, mctx.x_ty, mctx.x_ty, mctx.w_ty, mctx.w_ty, mctx.w_ty, mctx.w_ty, mctx.rtp_ty]
            mctx.func("gemm_e", args)
            mctx.func("gemm_o", args)

            for i in range(total_rows):
                for j in range(total_cols):
                    t = CT[i][j]
                    mctx.buffer(f"rtp_{t.row}_{t.col}", t, mctx.rtp_ty, use_write_rtp=True)

            for j in range(total_cols):
                for i in range(total_rows // 2):
                    gemm_pair(CT[2 * i][j], CT[2 * i + 1][j], mctx)
                gemm_memtile(MT[j], mctx)

            for j in range(total_cols):
                # weights
                connect_dma(IT[j], MT[j], 0, 0)
                connect_dma(MT[j], CT[0][j], 0, 1)
                connect_dma(MT[j], CT[2][j], 0, 1)
                connect_dma(MT[j], CT[1][j], 1, 1)
                connect_dma(MT[j], CT[3][j], 1, 1)
                # y
                for i in range(total_rows):
                    connect_dma(CT[i][j], MT[j], 0, 1 + i)
                connect_dma(MT[j], IT[j], 2, 0)

            # x: one shim per core row, broadcast along the row
            for i in range(total_rows):
                for j in range(total_cols):
                    connect_dma(IT[i], CT[i][j], 1, 0)

        print(ctx.module)


if __name__ == "__main__":
    flow_arch()
