#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# txn_split_switch.py IN.txn OUT.txn -- an experiment: every block write into
# stream-switch register space (core and shim 0x3F000-0x3FFFF, mem tile
# 0xB0000-0xBFFFF) becomes one write32 per register, as aiecc writes them.

import struct
import sys


def is_switch(addr):
    row, off = (addr >> 20) & 0x1F, addr & 0xFFFFF
    return 0xB0000 <= off < 0xC0000 if row == 1 else 0x3F000 <= off < 0x40000


data = open(sys.argv[1], "rb").read()
w = list(struct.unpack(f"<{len(data) // 4}I", data))
out, i, ops, split = w[:4], 4, 0, 0
for _ in range(w[2]):
    op = w[i] & 0xFF
    size = {0: 6, 3: 7, 0x80: 4, 0x81: 12, 6: 1}.get(op)
    if op == 1:
        size = w[i + 3] // 4
        if is_switch(w[i + 2]):
            for k, v in enumerate(w[i + 4:i + size]):
                out += [0, 0, w[i + 2] + 4 * k, 0, v, 24]   # write32: op, -, addr, -, value, size in bytes
                ops += 1
            split += 1
            i += size
            continue
    out += w[i:i + size]
    ops += 1
    i += size
out[2], out[3] = ops, len(out) * 4
open(sys.argv[2], "wb").write(struct.pack(f"<{len(out)}I", *out))
print(f"{sys.argv[2]}: {split} block writes into switch registers split, {ops} ops")
