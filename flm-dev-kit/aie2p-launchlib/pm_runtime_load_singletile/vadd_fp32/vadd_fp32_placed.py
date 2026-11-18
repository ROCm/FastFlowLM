#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# vadd_fp32 for AIE2P (npu2): the AIE2P port of aie4-launchlib's vadd_fp32.
#
# COLS columns of TILES = 4 cores (rows 2..5), one vector of N = 256 fp32 per
# core and launch. C = A + B, where A (N fp32) is ONE vector for every core:
#
#   DDR A[N] --shim (0,0) MM2S--> broadcast to every core of every column
#   per column c (slice c of B and C, COL_ELEMS = TILES * N elements):
#     DDR B --shim MM2S--> mem tile --split--> core r
#     core r --> mem tile --join--> shim S2MM --> DDR C
#
# The routing differs from vmul_fp32 (pair block) and vminus_fp32 (A on the core
# stream port), so a switch between them changes routes, BDs and locks.
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


def core_body(t, kernel, fa, fb, fc):
    @core(t, link_with="vadd_fp32.o")
    def _():
        # Never ends; the next kernel stops the core.
        for _ in range_(sys.maxsize):
            ea = fa.acquire(ObjectFifoPort.Consume, 1)
            eb = fb.acquire(ObjectFifoPort.Consume, 1)
            ec = fc.acquire(ObjectFifoPort.Produce, 1)
            kernel(ea, eb, ec)
            fa.release(ObjectFifoPort.Consume, 1)
            fb.release(ObjectFifoPort.Consume, 1)
            fc.release(ObjectFifoPort.Produce, 1)

with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu2)
    def _():
        vec_ty = np.ndarray[(N,), np.dtype[np.float32]]
        col_ty = np.ndarray[(COL_ELEMS,), np.dtype[np.float32]]
        kernel = external_func("vadd_fp32", inputs=[vec_ty, vec_ty, vec_ty])

        shims = [tile(c, 0) for c in range(COLS)]
        mems = [tile(c, 1) for c in range(COLS)]
        cores = [[tile(c, 2 + r) for r in range(TILES)] for c in range(COLS)]
        offsets = [r * N for r in range(TILES)]

        # A: one vector from column 0's shim to every core.
        a = object_fifo("A_0", shims[0], [t for col in cores for t in col], 2, vec_ty)
        bs, cs = [], []
        for c in range(COLS):
            bs.append(object_fifo(f"B_{c}", shims[c], mems[c], 2, col_ty))
            cs.append(object_fifo(f"C_{c}", mems[c], shims[c], 2, col_ty))
            core_b = [object_fifo(f"B_{c}_{r}", mems[c], cores[c][r], 2, vec_ty) for r in range(TILES)]
            core_c = [object_fifo(f"C_{c}_{r}", cores[c][r], mems[c], 2, vec_ty) for r in range(TILES)]
            object_fifo_link(bs[c], core_b, [], offsets)
            object_fifo_link(core_c, cs[c], offsets, [])
            for r in range(TILES):
                core_body(cores[c][r], kernel, a, core_b[r], core_c[r])

        a_ty = np.ndarray[(N,), np.dtype[np.float32]]
        bc_ty = np.ndarray[(COLS * COL_ELEMS,), np.dtype[np.float32]]

        # The reference sequence; the host runs its C++ copy (host_test/sequence.hpp).
        @runtime_sequence(a_ty, bc_ty, bc_ty)
        def _(A, B, C):
            tasks = [shim_dma_single_bd_task(a, A, sizes=[1, 1, 1, N])]
            outs = []
            for c in range(COLS):
                tasks.append(shim_dma_single_bd_task(bs[c], B, offset=c * COL_ELEMS, sizes=[1, 1, 1, COL_ELEMS]))
                outs.append(shim_dma_single_bd_task(cs[c], C, offset=c * COL_ELEMS, sizes=[1, 1, 1, COL_ELEMS],
                                                    issue_token=True))
                tasks.append(outs[-1])
            dma_start_task(*tasks)
            dma_await_task(*outs)

    res = ctx.module.operation.verify()
    if res is not True:
        print(res, file=sys.stderr)
        sys.exit(1)
    print(ctx.module)
