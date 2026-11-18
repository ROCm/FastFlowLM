#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# aie_status_diff.py A.txt B.txt -- the status words that differ between two
# `aie_status --raw` snapshots, named by tile and field (aie_status_decode.py
# has the layout).

import re
import sys


def words(path):
    text = open(path).read()
    parts = re.split(r"column (\d+):", text)[1:]
    return {int(parts[i]): [int(x, 16) for x in re.findall(r"\b[0-9a-f]{8}\b", parts[i + 1])]
            for i in range(0, len(parts), 2)}


def field(i):
    """Tile and field of word i of a column."""
    if i < 80:
        t, k = divmod(i, 20)
        names = (["DMA S2MM0", "DMA MM2S0", "DMA S2MM1", "DMA MM2S1"] + [f"mem event {j}" for j in range(4)]
                 + [f"core event {j}" for j in range(4)] + ["core status", "core word 1", "core word 2", "core word 3"]
                 + [f"locks {4 * j}-{4 * j + 3}" for j in range(4)])
        return f"core row {t + 2}", names[k]
    if i < 114:
        k = i - 80
        if k < 12:
            return "mem tile", f"DMA {'S2MM' if k % 2 == 0 else 'MM2S'}{k // 2}"
        if k < 18:
            return "mem tile", f"event {k - 12}"
        return "mem tile", f"locks {4 * (k - 18)}-{4 * (k - 18) + 3}"
    k = i - 114
    names = ["DMA S2MM0", "DMA MM2S0", "DMA S2MM1", "DMA MM2S1"] + [f"event {j}" for j in range(4)] + \
        [f"locks {4 * j}-{4 * j + 3}" for j in range(4)]
    return "shim", names[k]


a, b = words(sys.argv[1]), words(sys.argv[2])
for col in sorted(a):
    for i, (x, y) in enumerate(zip(a[col], b.get(col, []))):
        if x != y:
            tile, name = field(i)
            print(f"column {col} {tile:12} {name:14} {x:08x} -> {y:08x}")
