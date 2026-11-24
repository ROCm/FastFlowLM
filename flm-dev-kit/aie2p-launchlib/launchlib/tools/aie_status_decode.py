#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# aie_status_decode.py SNAPSHOT.txt [--all]
#
# Decodes the output of `aie_status --raw` (DRM_AMDXDNA_QUERY_AIE_STATUS) for
# npu2 (8 columns; per column 504 B). Layout, found by matching the current BD
# numbers with the designs' chain starts:
#   core tile (rows 2-5), 20 words: DMA status S2MM0 MM2S0 S2MM1 MM2S1, memory
#     events (4), core events (4), core status, 3 core words, locks (16 bytes)
#   mem tile (row 1), 34 words: DMA status S2MMk/MM2Sk interleaved (12), events
#     (6), locks (64 bytes)
#   shim (row 0), 12 words: DMA status S2MM0 MM2S0 S2MM1 MM2S1, events (4),
#     locks (16 bytes)
# Prints, per tile, the DMA channels that are not idle, the core status and the
# locks that are not 0 (--all: every channel).

import re
import sys

CORE_STATUS_BITS = [(0, "enabled"), (1, "reset"), (2, "mem-stall-S"), (3, "mem-stall-W"), (4, "mem-stall-N"),
                    (5, "mem-stall-E"), (6, "lock-stall-S"), (7, "lock-stall-W"), (8, "lock-stall-N"),
                    (9, "lock-stall-E"), (10, "stream-stall-SS0"), (12, "stream-stall-MS0"),
                    (14, "cascade-stall-SCD"), (15, "cascade-stall-MCD"), (16, "debug-halt"),
                    (19, "error-halt"), (20, "done")]


def dma(v):
    """A DMA channel status word, in words."""
    state = ["idle", "starting", "running", "?"][v & 3]
    flags = [name for bit, name in [(2, "lock-acq-stall"), (3, "lock-rel-stall"), (4, "stream-starved"),
                                    (5, "token-fifo-full"), (10, "BD-unavailable"), (11, "BD-invalid"),
                                    (18, "queue-overflow")] if v >> bit & 1]
    return f"{state} bd{(v >> 24) & 63} queue{(v >> 20) & 7}" + (" " + " ".join(flags) if flags else "")


def locks(words, n):
    vals = [(w >> (8 * b)) & 0xFF for w in words for b in range(4)][:n]
    return {i: v for i, v in enumerate(vals) if v}


def main():
    path = sys.argv[1]
    every = "--all" in sys.argv
    text = open(path).read()
    parts = re.split(r"column (\d+):", text)[1:]
    for i in range(0, len(parts), 2):
        col = int(parts[i])
        w = [int(x, 16) for x in re.findall(r"\b[0-9a-f]{8}\b", parts[i + 1])]
        print(f"column {col}")
        shim, mem = w[114:126], w[80:114]
        names = ["S2MM0", "MM2S0", "S2MM1", "MM2S1"]
        chans = [f"{names[k]} {dma(shim[k])}" for k in range(4) if every or shim[k] & 3]
        print(f"  shim   {'; '.join(chans) or '-'}   locks {locks(shim[8:12], 16) or '-'}")
        chans = [f"{'S2MM' if k % 2 == 0 else 'MM2S'}{k // 2} {dma(mem[k])}" for k in range(12) if every or mem[k] & 3]
        print(f"  mem    {'; '.join(chans) or '-'}")
        print(f"         locks {locks(mem[18:34], 64) or '-'}")
        for t in range(4):
            c = w[t * 20:(t + 1) * 20]
            chans = [f"{names[k]} {dma(c[k])}" for k in range(4) if every or c[k] & 3]
            status = [name for bit, name in CORE_STATUS_BITS if c[12] >> bit & 1]
            print(f"  core{t + 2}  {'; '.join(chans) or '-'} | core {' '.join(status) or '-'} | locks {locks(c[16:20], 16) or '-'}")


main()
