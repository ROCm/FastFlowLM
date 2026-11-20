#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# One array image: out = lhs (op) rhs, op = add or mul, on COLS columns of
# CORES cores each, for AIE2P (npu2). The Makefile runs this twice (--op add,
# --op mul); aiecc builds each run and tools/make_package.py (launchlib) turns
# the two into the package the launch library reads. vadd and vmul differ only
# in the kernel, so a wrong answer reads as "the wrong program ran", and the
# library can switch between them by the program alone.
#
# The cores loop forever, as the AIE4 designs: the next kernel either runs the
# sequence again (rerun) or stops them (warm, cold).
#
# lhs and rhs are the two inputs of out = lhs (op) rhs. Per column c, which
# owns slice c of every buffer, two layouts (--layout):
#
#   split (CORES 1-2)   each core gets its own lhs and rhs slices
#     DDR --(shim MM2S ch0)--> mem tile lhs --split--> core r (r = 0..CORES-1)
#     DDR --(shim MM2S ch1)--> mem tile rhs --split--> core r
#   pair  (CORES 1-4)   the mem tile joins lhs and rhs into one block
#                       [lhs of every core | rhs of every core] and broadcasts
#                       it to every core of the column; core r computes slice r
#     DDR --(shim MM2S ch0)--> mem tile lhs --+
#     DDR --(shim MM2S ch1)--> mem tile rhs --+--join--> pair --broadcast--> core r
#   both
#     core r --> mem tile out --join--> (shim S2MM ch0) --> DDR
#
# Why two: an AIE2P mem tile has 6 MM2S channels. split needs 2 * CORES + 1 of
# them (lhs and rhs to each core, out to the shim), so 2 cores at most. pair
# needs 2 (pair, out), so 4 cores fit (32 on the array); each core then gets the
# whole column block on its stream (CORES times the data it uses). IRON cannot
# join and split in one link, so pair broadcasts. The shim side is the same in
# both: the same runtime sequence and shim channels (lhs<c>, rhs<c>, out<c>).
#
# The runtime sequence below is the reference only (two blocks per core): the
# host runs the C++ copy (include/elementwise_sequence.hpp), any length, and
# make check compares the two.
#

import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

parser = argparse.ArgumentParser(description="vadd / vmul bf16 for npu2 (AIE2P)")
parser.add_argument("--op", required=True, choices=["add", "mul"])
parser.add_argument("--n", type=int, default=256,
                    help="block length per core in bf16 elements")
parser.add_argument("--cols", type=int, default=1, choices=range(1, 9))
parser.add_argument("--cores", type=int, default=1, choices=[1, 2, 3, 4],
                    help="cores per column")
parser.add_argument("--layout", choices=["split", "pair"], default=None,
                    help="split (default for 1-2 cores) or pair (default for 3-4)")
args = parser.parse_args()

N = args.n
if N % 16 != 0:
    raise ValueError(f"n={N} must be a multiple of 16, the bf16 vector width")
COLS, CORES = args.cols, args.cores
LAYOUT = args.layout or ("split" if CORES <= 2 else "pair")
if LAYOUT == "split" and CORES > 2:
    raise ValueError(f"split needs 2 * {CORES} + 1 mem-tile MM2S channels, there are 6: use --layout pair")
COMPUTE_ROW = 2
REFERENCE_BLOCKS = 2  # blocks per core of the reference runtime sequence

# Two bf16 per i32 word. This is what the shim BDs count.
N_WORDS = N // 2


def split_core(tile_, kernel, compute_fn, in_lhs, in_rhs, res):
    @core(tile_, link_with=f"{kernel}.o")
    def _():
        # Never ends; the next kernel stops the core.
        for _ in range_(sys.maxsize):
            elem_lhs = in_lhs.acquire(ObjectFifoPort.Consume, 1)
            elem_rhs = in_rhs.acquire(ObjectFifoPort.Consume, 1)
            elem_out = res.acquire(ObjectFifoPort.Produce, 1)
            compute_fn(elem_lhs, elem_rhs, elem_out)
            in_lhs.release(ObjectFifoPort.Consume, 1)
            in_rhs.release(ObjectFifoPort.Consume, 1)
            res.release(ObjectFifoPort.Produce, 1)


def pair_core(tile_, kernel, compute_fn, in_pair, res, slice_):
    @core(tile_, link_with=f"{kernel}.o")
    def _():
        # Never ends; the next kernel stops the core.
        for _ in range_(sys.maxsize):
            elem_pair = in_pair.acquire(ObjectFifoPort.Consume, 1)
            elem_out = res.acquire(ObjectFifoPort.Produce, 1)
            compute_fn(elem_pair, elem_out, slice_)
            in_pair.release(ObjectFifoPort.Consume, 1)
            res.release(ObjectFifoPort.Produce, 1)

with mlir_mod_ctx() as ctx:

    # The whole array (8 columns), like the FLM xclbins.
    @device(AIEDevice.npu2)
    def _():
        # i32, not bf16: the DMA moves bytes and the kernel casts.
        core_ty = np.ndarray[(N_WORDS,), np.dtype[np.int32]]
        mem_ty = np.ndarray[(CORES * N_WORDS,), np.dtype[np.int32]]

        pair_ty = np.ndarray[(2 * CORES * N_WORDS,), np.dtype[np.int32]]

        # A bare filename: aiecc resolves it against its working directory.
        kernel = f"v{args.op}_bf16"
        if LAYOUT == "split":
            compute_fn = external_func(kernel, inputs=[core_ty, core_ty, core_ty])
        else:
            compute_fn = external_func(f"v{args.op}_pair_bf16", inputs=[pair_ty, core_ty, np.int32])

        shims = [tile(c, 0) for c in range(COLS)]
        mems = [tile(c, 1) for c in range(COLS)]
        computes = [[tile(c, COMPUTE_ROW + r) for r in range(CORES)]
                    for c in range(COLS)]

        offsets = [r * N_WORDS for r in range(CORES)]
        lhs, rhs, out = [], [], []
        for c in range(COLS):
            # Shim <-> mem tile, one block of the column.
            lhs.append(object_fifo(f"lhs{c}", shims[c], mems[c], 2, mem_ty))
            rhs.append(object_fifo(f"rhs{c}", shims[c], mems[c], 2, mem_ty))
            out.append(object_fifo(f"out{c}", mems[c], shims[c], 2, mem_ty))
            # Each core -> mem tile, its slice of the out block.
            core_out = [object_fifo(f"out{c}_{r}", computes[c][r], mems[c], 2, core_ty)
                        for r in range(CORES)]
            object_fifo_link(core_out, out[c], offsets, [])

            if LAYOUT == "split":
                # Mem tile -> each core, its slices of lhs and rhs.
                core_lhs = [object_fifo(f"lhs{c}_{r}", mems[c], computes[c][r], 2, core_ty)
                            for r in range(CORES)]
                core_rhs = [object_fifo(f"rhs{c}_{r}", mems[c], computes[c][r], 2, core_ty)
                            for r in range(CORES)]
                object_fifo_link(lhs[c], core_lhs, [], offsets)
                object_fifo_link(rhs[c], core_rhs, [], offsets)
                for r in range(CORES):
                    split_core(computes[c][r], kernel, compute_fn, core_lhs[r], core_rhs[r], core_out[r])
            else:
                # Mem tile -> every core of the column: [lhs block | rhs block].
                pair = object_fifo(f"pair{c}", mems[c], computes[c], 2, pair_ty)
                object_fifo_link([lhs[c], rhs[c]], pair, [0, CORES * N_WORDS], [])
                for r in range(CORES):
                    pair_core(computes[c][r], kernel, compute_fn, pair, core_out[r], r)

        # One buffer over all columns; column c moves slice c of it.
        words = COLS * CORES * N_WORDS * REFERENCE_BLOCKS
        buf_ty = np.ndarray[(words,), np.dtype[np.int32]]
        col_words = words // COLS

        @runtime_sequence(buf_ty, buf_ty, buf_ty)
        def _(a, b, dst):
            tasks, outs = [], []
            for c in range(COLS):
                off = c * col_words
                tasks.append(shim_dma_single_bd_task(
                    lhs[c], a, offset=off, sizes=[1, 1, 1, col_words]))
                tasks.append(shim_dma_single_bd_task(
                    rhs[c], b, offset=off, sizes=[1, 1, 1, col_words]))
                outs.append(shim_dma_single_bd_task(
                    out[c], dst, offset=off, sizes=[1, 1, 1, col_words],
                    issue_token=True))
                tasks.append(outs[-1])
            dma_start_task(*tasks)
            # A column's output comes from its inputs: its token covers them.
            dma_await_task(*outs)

    res = ctx.module.operation.verify()
    if res is not True:
        print(res, file=sys.stderr)
        sys.exit(1)
    print(ctx.module)
