# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Hy-MT2 prefill overlay: dequant_mm and attention take turns on the whole array.

    variant base : arenas, RTP blocks, op locks and the core programs (both
                   kernels + a dispatcher). No routes, no DMA chains. This is
                   the xclbin that runs.
    variant dq   : base + dequant_mm's locks, chains and routes on all 8 columns
                   (two 4-column halves, as the original standalone design)
    variant attn : base + attention's locks, chains and routes on columns 0..3
    variant dq32 / dq64 : dequant_mm for 32 / 64 L rows ("split"): the two core
                   pairs of a column take different 128-row weight lanes, each
                   core 16 / 32 tokens. Only the mem tile's weight relay and the
                   x / y lengths differ from dq. See README.md "Small-row plans".

The dq / attn variants are never run: aiecc --aie-generate-txn dumps their
load-time configuration, tools/extract_txn2.py turns it into host/op_configs.hpp,
and the host writes one op's configuration into the base xclbin when the op
changes. See README.md.

    python3 overlay_aie.py base | dq | dq32 | dq64 | attn
"""
import sys
import numpy as np
from ml_dtypes import bfloat16

from aie.extras.context import mlir_mod_ctx
from aie.extras import types as T
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_

from utils import generate_tile_map, connect_dma
from layout import CT_ARENA_BYTES, MT_ARENA_BYTES, RTP_WORDS, OP_LOCK, STACK
from dq_tiles import dq_core_pair, dq_memtile
from attn_tiles import attn_core, attn_memtile_even, attn_memtile_odd

COLS = 8
ROWS = 4
HALF = 4          # dequant_mm columns per half
ATTN_COLS = 4
# dequant_mm variants: (tokens per core row per round, split weights)
DQ_VARIANTS = dict(dq=(32, False), dq32=(16, True), dq64=(32, True))


def flow_arch(variant):
    dev = aie.AIEDevice.npu2
    bf16 = np.dtype[bfloat16]

    with mlir_mod_ctx() as ctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(ROWS + 2, COLS)

            ct_arena_ty = np.ndarray[(CT_ARENA_BYTES // 2,), bf16]
            mt_arena_ty = np.ndarray[(MT_ARENA_BYTES // 2,), bf16]
            rtp_ty = np.ndarray[(RTP_WORDS,), np.dtype[np.int32]]
            # fused(arena, neighbour arena, rtp, row, col): even rows get the
            # north (odd) arena -- the dequantized w tile -- and odd rows the
            # south (even) arena -- the q4_k blocks
            fused = aie.external_func("fused", [ct_arena_ty, ct_arena_ty, rtp_ty, T.i32(), T.i32()])

            # ---- arenas, RTP blocks, op locks: identical in every variant --------
            ct_arena = [[None] * COLS for _ in range(ROWS)]
            ct_rtp = [[None] * COLS for _ in range(ROWS)]
            for r in range(ROWS):
                for c in range(COLS):
                    t = CT[r][c].tile
                    ct_arena[r][c] = aie.buffer(t, ct_arena_ty, name=f"arena_{r + 2}_{c}")
                    ct_rtp[r][c] = aie.buffer(t, rtp_ty, name=f"rtp_{r + 2}_{c}")
                    aie.lock(t, lock_id=OP_LOCK, init=0)
            mt_arena = [aie.buffer(MT[c].tile, mt_arena_ty, name=f"mt_arena_{c}") for c in range(COLS)]

            def make_core(r, c):
                # a function per core so the body captures r, c (the decorator
                # would read default arguments as MLIR block arguments)
                even = r % 2 == 0
                other = ct_arena[r + 1][c] if even else ct_arena[r - 1][c]
                obj = "fused_e.o" if even else "fused_o.o"

                @aie.core(CT[r][c].tile, link_with=obj, stack_size=STACK)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        fused(ct_arena[r][c], other, ct_rtp[r][c], r, c)

            for r in range(ROWS):
                for c in range(COLS):
                    make_core(r, c)

            if variant == "base":
                return

            if variant in DQ_VARIANTS:
                m, split = DQ_VARIANTS[variant]
                for c in range(COLS):
                    for r in range(0, ROWS, 2):
                        dq_core_pair(ct_arena[r][c], ct_arena[r + 1][c], CT[r][c].tile, CT[r + 1][c].tile)
                    dq_memtile(mt_arena[c], MT[c].tile, m, split)
                # w: shim c -> mem c S2MM 5 -> MM2S 2 -> the even cores of column c;
                # split: MM2S 2 -> core row 0 only, MM2S 3 -> core row 2
                for c in range(COLS):
                    connect_dma(IT[c], MT[c], 1, 5)
                    if split:
                        connect_dma(MT[c], CT[0][c], 2, 1)
                        connect_dma(MT[c], CT[2][c], 3, 1)
                    else:
                        for r in range(0, ROWS, 2):
                            connect_dma(MT[c], CT[r][c], 2, 1)
                # x: core row r from mem 2r, broadcast along the whole row (both halves)
                for r in range(ROWS):
                    connect_dma(IT[2 * r], MT[2 * r], 0, 0)
                    for c in range(COLS):
                        connect_dma(MT[2 * r], CT[r][c], 0, 0)
                # y: the 4 cores of a row in one half into one mem tile, then out its shim
                for r in range(ROWS):
                    for c in range(COLS):
                        connect_dma(CT[r][c], MT[r + HALF * (c // HALF)], 0, 1 + c % HALF)
                for c in range(COLS):
                    connect_dma(MT[c], IT[c], 1, 0)

            elif variant == "attn":
                for c in range(ATTN_COLS):
                    for r in range(ROWS):
                        attn_core(ct_arena[r][c], CT[r][c].tile)
                    if c % 2 == 0:
                        attn_memtile_even(mt_arena[c], MT[c].tile)
                    else:
                        attn_memtile_odd(mt_arena[c], MT[c].tile)
                # q: shim 2j -> mem 2j (two halves) -> core rows of columns 2j, 2j + 1
                for j in range(ATTN_COLS // 2):
                    connect_dma(IT[2 * j], MT[2 * j], 0, 0)
                    connect_dma(IT[2 * j], MT[2 * j], 1, 1)
                    for r in range(ROWS):
                        connect_dma(MT[2 * j], CT[r][2 * j], r, 0)
                        connect_dma(MT[2 * j], CT[r][2 * j + 1], r, 0)
                # k / v: shim 3 -> mem 3 -> every attention core (both CUs share a KV head)
                connect_dma(IT[3], MT[3], 0, 0)
                connect_dma(IT[3], MT[3], 1, 1)
                for c in range(ATTN_COLS):
                    for r in range(ROWS):
                        connect_dma(MT[3], CT[r][c], 0, 1)
                # o: rows 0, 1 into mem 2j, rows 2, 3 into mem 2j + 1, then to the shim
                for j in range(ATTN_COLS // 2):
                    connect_dma(CT[0][2 * j], MT[2 * j], 0, 2)
                    connect_dma(CT[0][2 * j + 1], MT[2 * j], 0, 3)
                    connect_dma(CT[1][2 * j], MT[2 * j], 0, 4)
                    connect_dma(CT[1][2 * j + 1], MT[2 * j], 0, 5)
                    connect_dma(CT[2][2 * j], MT[2 * j + 1], 0, 2)
                    connect_dma(CT[2][2 * j + 1], MT[2 * j + 1], 0, 3)
                    connect_dma(CT[3][2 * j], MT[2 * j + 1], 0, 4)
                    connect_dma(CT[3][2 * j + 1], MT[2 * j + 1], 0, 5)
                for c in range(ATTN_COLS):
                    connect_dma(MT[c], IT[c], 5, 0)
            else:
                raise ValueError(variant)

        print(ctx.module)


if __name__ == "__main__":
    flow_arch(sys.argv[1] if len(sys.argv) > 1 else "base")
