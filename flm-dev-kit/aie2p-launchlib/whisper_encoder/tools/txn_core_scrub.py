#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# txn_core_scrub.py IN.txn OUT.txn [GROUPS] -- an experiment: after the 32 core
# holds at the front of a switch stream, writes the reset value of every core
# register of GROUPS (default all) in all 32 cores. Registers and reset values:
# aie-rt xaie2pgbl_params.h, XAIE2PGBL_CORE_MODULE_CORE_*. The accumulator (BM)
# and vector (W) parts are 128-bit windows: a block write of 4 words each.

import struct
import sys

WIDE = {
    "bm": [0x30000 + 0x100 * r + 0x40 * q + 0x10 * p for r in range(5) for q in range(4) for p in range(4)],
    "w": [0x30800 + 0x20 * k + 0x10 * p for k in range(24) for p in range(2)],
}
NARROW = {
    "q": [(0x30C00 + 0x10 * i, 0) for i in range(4)],
    "e": [(0x30D00 + 0x10 * i, 0) for i in range(12)],
    "ctrl": [(0x30E10, 0), (0x30E40, 0), (0x30E50, 0xFFFFF), (0x30E60, 0),
             (0x30E70, 0x1800), (0x30E80, 0x2AA), (0x30E90, 0)],      # FC LS LE LC CR1 CR2 SR
    "r": [(0x31000 + 0x10 * i, 0) for i in range(32)],
    "mdp": [(0x31200 + 0x10 * i, 0) for i in range(40)],              # M DN DJ DC P
    "s": [(0x31480 + 0x10 * i, 0) for i in range(4)],
}

data = open(sys.argv[1], "rb").read()
groups = (sys.argv[3] if len(sys.argv) > 3 else ",".join(list(WIDE) + list(NARROW))).split(",")
w = list(struct.unpack(f"<{len(data) // 4}I", data))
holds = 4 + 32 * 7  # the 32 maskwrite32 (7 words) core holds
assert all((w[4 + 7 * k + 2] & 0xFFFFF) == 0x32000 for k in range(32)), "no 32 core holds at the front"
ops, n = [], 0
for col in range(8):
    for row in range(2, 6):
        base = (col << 25) | (row << 20)
        for g in groups:
            for off in WIDE.get(g, []):
                ops += [1, 0, base | off, 4 * 8, 0, 0, 0, 0]  # blockwrite of 4 words
                n += 1
            for off, value in NARROW.get(g, []):
                ops += [0, 0, base | off, 0, value, 24]       # write32
                n += 1
out = w[:holds] + ops + w[holds:]
out[2] = w[2] + n
out[3] = len(out) * 4
open(sys.argv[2], "wb").write(struct.pack(f"<{len(out)}I", *out))
print(f"{sys.argv[2]}: core registers {','.join(groups)} reset after the core holds ({n} writes)")
