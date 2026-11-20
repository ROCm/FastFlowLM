# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""MiniCPM-OCR vision overlay, base design: the core program and memory of all 32 cores. See ../README.md.

No routes, locks (other than the go lock) or DMA chains: the host writes those for each op,
from the standalone designs' own configuration (host/op_configs.hpp).

    python3 overlay_aie.py base | pm  LAYOUT_JSON      (pm: one core only, to check program memory)
"""
import json
import sys

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.extras import types as T
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_

COLS = 8
ROWS = 4


def flow_arch(variant, layout):
    with mlir_mod_ctx() as ctx:
        @aie.device(aie.AIEDevice.npu2)
        def device_body():
            arena_ty = np.ndarray[(layout["arena_bytes"],), np.dtype[np.int8]]
            fused = aie.external_func("fused", [arena_ty, T.i32()])
            for col in range(COLS):
                aie.tile(col, 0)
                aie.tile(col, 1)

            def make_core(col, row):
                tile = aie.tile(col, row)
                arena = aie.buffer(tile, arena_ty, name=f"arena_{row}_{col}")
                aie.lock(tile, lock_id=layout["op_lock"], init=0)

                @aie.core(tile, link_with="fused.o", stack_size=layout["stack"])
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        fused(arena, col & 1)

            for col in range(COLS):
                for row in range(2, 2 + ROWS):
                    if variant == "base" or (col, row) == (0, 2):
                        make_core(col, row)

        print(ctx.module)


if __name__ == "__main__":
    flow_arch(sys.argv[1], json.load(open(sys.argv[2])))
