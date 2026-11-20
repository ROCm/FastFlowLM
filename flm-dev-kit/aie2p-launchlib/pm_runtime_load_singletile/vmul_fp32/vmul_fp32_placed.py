#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# vmul_fp32 for AIE2P (npu2): the AIE2P port of aie4-launchlib's vmul_fp32.
#
# COLS columns of TILES = 4 cores, one vector of N = 256 fp32 per core and
# launch, C = A * B. Per column c (slice c of A, B and C):
#
#   DDR A --shim MM2S 0--> mem tile --+
#   DDR B --shim MM2S 1--> mem tile --+--join--> [A block | B block] --broadcast--> core r
#   core r --> mem tile --join--> shim S2MM --> DDR C
#
# On AIE4 the mem tile sends each core its own slices of A and B; an AIE2P mem
# tile has 6 MM2S channels, too few for 2 * 4 + 1, so this design joins A and B
# into one block and broadcasts it (2 MM2S per column); core r computes slice r.
# The cores loop forever; the launch library's next kernel stops them.

import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

p = argparse.ArgumentParser()
p.add_argument("--cols", type=int, default=4, choices=range(1, 9))
args = p.parse_args()

COLS, TILES, N = args.cols, 4, 256
COL_ELEMS = TILES * N


def core_body(t, kernel, fpair, fc, r):
    @core(t, link_with="vmul_fp32.o")
    def _():
        for _ in range_(sys.maxsize):
            ep = fpair.acquire(ObjectFifoPort.Consume, 1)
            ec = fc.acquire(ObjectFifoPort.Produce, 1)
            kernel(ep, ec, r)
            fpair.release(ObjectFifoPort.Consume, 1)
            fc.release(ObjectFifoPort.Produce, 1)


with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu2)
    def _():
        vec_ty = np.ndarray[(N,), np.dtype[np.float32]]
        col_ty = np.ndarray[(COL_ELEMS,), np.dtype[np.float32]]
        pair_ty = np.ndarray[(2 * COL_ELEMS,), np.dtype[np.float32]]
        kernel = external_func("vmul_fp32", inputs=[pair_ty, vec_ty, np.int32])

        shims = [tile(c, 0) for c in range(COLS)]
        mems = [tile(c, 1) for c in range(COLS)]
        cores = [[tile(c, 2 + r) for r in range(TILES)] for c in range(COLS)]
        offsets = [r * N for r in range(TILES)]

        as_, bs, cs = [], [], []
        for c in range(COLS):
            as_.append(object_fifo(f"A_{c}", shims[c], mems[c], 2, col_ty))
            bs.append(object_fifo(f"B_{c}", shims[c], mems[c], 2, col_ty))
            cs.append(object_fifo(f"C_{c}", mems[c], shims[c], 2, col_ty))
            pair = object_fifo(f"P_{c}", mems[c], cores[c], 2, pair_ty)
            object_fifo_link([as_[c], bs[c]], pair, [0, COL_ELEMS], [])
            core_c = [object_fifo(f"C_{c}_{r}", cores[c][r], mems[c], 2, vec_ty) for r in range(TILES)]
            object_fifo_link(core_c, cs[c], offsets, [])
            for r in range(TILES):
                core_body(cores[c][r], kernel, pair, core_c[r], r)

        abc_ty = np.ndarray[(COLS * COL_ELEMS,), np.dtype[np.float32]]

        # The reference sequence; the host runs its C++ copy (host_test/sequence.hpp).
        @runtime_sequence(abc_ty, abc_ty, abc_ty)
        def _(A, B, C):
            tasks, outs = [], []
            for c in range(COLS):
                off = c * COL_ELEMS
                tasks.append(shim_dma_single_bd_task(as_[c], A, offset=off, sizes=[1, 1, 1, COL_ELEMS]))
                tasks.append(shim_dma_single_bd_task(bs[c], B, offset=off, sizes=[1, 1, 1, COL_ELEMS]))
                outs.append(shim_dma_single_bd_task(cs[c], C, offset=off, sizes=[1, 1, 1, COL_ELEMS],
                                                    issue_token=True))
                tasks.append(outs[-1])
            dma_start_task(*tasks)
            dma_await_task(*outs)

    res = ctx.module.operation.verify()
    if res is not True:
        print(res, file=sys.stderr)
        sys.exit(1)
    print(ctx.module)
