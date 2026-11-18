#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# vminus_fp32 for AIE2P (npu2): the AIE2P port of aie4-launchlib's vminus_fp32.
#
# COLS columns of TILES = 4 cores, one vector of N = 256 fp32 per core and
# launch, C = A - B. Per column c (slice c of A, B and C):
#
#   DDR A --shim MM2S 1--> (stream, no mem tile, no buffer) --> stream port of
#                          every core of the column; core r keeps slice r
#   DDR B --shim MM2S 0--> mem tile --split--> core r (core DMA)
#   core r --> mem tile --join--> shim S2MM --> DDR C
#
# Each core also has an RTP buffer ct_rtp_<c>_<r> (word 0: the elements per core)
# and an RTP lock ct_rtplock_<c>_<r>: the runtime sequence writes the RTP and
# sets the lock to 1 before the transfers, the core acquires it once per vector.
# So the sequence uses npu_rtp_write and set_lock_value (by name, from the
# manifest), as AIE4's vminus.
#
# The routing differs on purpose from vadd_fp32 and vmul_fp32 (A goes to the core
# stream port, not to a core DMA), so a switch to or from this design changes the
# core stream switch, the core DMA channels and the locks.

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
RTP_LOCK_ID = 15  # the object FIFOs take the low lock ids


def core_body(t, kernel, rtp, rtplock, fb, fc, r):
    @core(t, link_with="vminus_fp32.o")
    def _():
        for _ in range_(sys.maxsize):
            use_lock(rtplock, LockAction.AcquireGreaterEqual, value=1)
            count = rtp[0]
            eb = fb.acquire(ObjectFifoPort.Consume, 1)
            ec = fc.acquire(ObjectFifoPort.Produce, 1)
            kernel(eb, ec, count, r)
            fb.release(ObjectFifoPort.Consume, 1)
            fc.release(ObjectFifoPort.Produce, 1)


with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu2)
    def _():
        vec_ty = np.ndarray[(N,), np.dtype[np.float32]]
        col_ty = np.ndarray[(COL_ELEMS,), np.dtype[np.float32]]
        rtp_ty = np.ndarray[(16,), np.dtype[np.int32]]
        kernel = external_func("vminus_fp32", inputs=[vec_ty, vec_ty, np.int32, np.int32])

        shims = [tile(c, 0) for c in range(COLS)]
        mems = [tile(c, 1) for c in range(COLS)]
        cores = [[tile(c, 2 + r) for r in range(TILES)] for c in range(COLS)]
        offsets = [r * N for r in range(TILES)]

        bs, cs, rtps, rtplocks = [], [], [], []
        for c in range(COLS):
            # A: the shim's MM2S 1 straight to the stream port of each core.
            shim_dma_allocation(f"A_{c}", DMAChannelDir.MM2S, 1, c)
            for r in range(TILES):
                flow(shims[c], WireBundle.DMA, 1, cores[c][r], WireBundle.Core, 0)
            bs.append(object_fifo(f"B_{c}", shims[c], mems[c], 2, col_ty))
            cs.append(object_fifo(f"C_{c}", mems[c], shims[c], 2, col_ty))
            core_b = [object_fifo(f"B_{c}_{r}", mems[c], cores[c][r], 2, vec_ty) for r in range(TILES)]
            core_c = [object_fifo(f"C_{c}_{r}", cores[c][r], mems[c], 2, vec_ty) for r in range(TILES)]
            object_fifo_link(bs[c], core_b, [], offsets)
            object_fifo_link(core_c, cs[c], offsets, [])
            rtps.append([])
            rtplocks.append([])
            for r in range(TILES):
                rtp = buffer(cores[c][r], rtp_ty, name=f"ct_rtp_{c}_{r}")
                lk = lock(cores[c][r], lock_id=RTP_LOCK_ID, init=0, sym_name=f"ct_rtplock_{c}_{r}")
                rtps[c].append(rtp)
                rtplocks[c].append(lk)
                core_body(cores[c][r], kernel, rtp, lk, core_b[r], core_c[r], r)

        abc_ty = np.ndarray[(COLS * COL_ELEMS,), np.dtype[np.float32]]

        # The reference sequence; the host runs its C++ copy (host_test/sequence.hpp).
        @runtime_sequence(abc_ty, abc_ty, abc_ty)
        def _(A, B, C):
            tasks, outs = [], []
            for c in range(COLS):
                for r in range(TILES):
                    npu_rtp_write(f"ct_rtp_{c}_{r}", 0, N)
                    set_lock_value(rtplocks[c][r], 1)
                off = c * COL_ELEMS
                tasks.append(shim_dma_single_bd_task(f"A_{c}", A, offset=off, sizes=[1, 1, 1, COL_ELEMS]))
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
