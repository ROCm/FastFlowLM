#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# txn_enable_only.py IN.txn OUT.txn TILES -- an experiment: like txn_no_enable.py,
# but keeps the core enable writes of TILES ("col,row;col,row", or "" for none).

import struct
import sys

keep = {tuple(map(int, t.split(","))) for t in sys.argv[3].split(";") if t}
data = open(sys.argv[1], "rb").read()
w = list(struct.unpack(f"<{len(data) // 4}I", data))
out, i, ops, dropped = w[:4], 4, 0, 0
for _ in range(w[2]):
    op = w[i] & 0xFF
    size = {0: 6, 3: 7, 0x80: 4, 0x81: 12, 6: 1}.get(op) or w[i + 3] // 4
    is_write = op in (0, 3)  # write32, maskwrite32; other ops (a 1-word preempt too) have no core enable
    addr = w[i + 2] if is_write else 0
    tile = ((addr >> 25) & 0x7F, (addr >> 20) & 0x1F)
    is_ctrl = is_write and (addr & 0xFFFFF) == 0x32000 and tile[1] >= 2
    enables = is_ctrl and (w[i + 4] & 1) and (op == 0 or w[i + 5] & 1)
    if enables and tile not in keep:
        dropped += 1
    else:
        out += w[i:i + size]
        ops += 1
    i += size
out[2], out[3] = ops, len(out) * 4
open(sys.argv[2], "wb").write(struct.pack(f"<{len(out)}I", *out))
print(f"{sys.argv[2]}: {dropped} core enables dropped, {len(keep)} kept")
