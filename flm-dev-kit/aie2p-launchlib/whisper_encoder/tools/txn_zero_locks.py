#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# txn_zero_locks.py IN.txn OUT.txn [ROWS] -- an experiment: puts a write of 0
# to every lock value register of the 8 columns (core 16, mem tile 64, shim 16)
# in front of the stream. ROWS (default "shim,mem,core") picks the tile kinds.

import struct
import sys

LOCKS = {"shim": ([0], 0x14000, 16), "mem": ([1], 0xC0000, 64), "core": ([2, 3, 4, 5], 0x1F000, 16)}

data = open(sys.argv[1], "rb").read()
kinds = (sys.argv[3] if len(sys.argv) > 3 else "shim,mem,core").split(",")
w = list(struct.unpack(f"<{len(data) // 4}I", data))
zero = []
for kind in kinds:
    rows, base, count = LOCKS[kind]
    for col in range(8):
        for row in rows:
            for i in range(count):
                zero += [0, 0, (col << 25) | (row << 20) | (base + 0x10 * i), 0, 0, 24]  # write32
out = w[:4] + zero + w[4:]
out[2] = w[2] + len(zero) // 6
out[3] = len(out) * 4
open(sys.argv[2], "wb").write(struct.pack(f"<{len(out)}I", *out))
print(f"{sys.argv[2]}: {len(zero) // 6} lock zero writes ({','.join(kinds)}) in front")
