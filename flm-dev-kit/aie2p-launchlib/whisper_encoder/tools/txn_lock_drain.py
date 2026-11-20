#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# txn_lock_drain.py IN.txn OUT.txn [VALUE] [DELAY] [holds|load] [KINDS] -- an
# experiment (KINDS: "mem,core" by default, the tiles whose locks it writes): after
# the 32 core holds at the front of a switch stream (with "load": before the
# first program write, after the whole reset part), writes VALUE (default 63) to
# every core and mem-tile lock, DELAY (default 200) harmless writes, then 0 to
# every one of them; the load part writes the app's lock values after that. A
# lock acquire that a stopped core left pending gets its lock there, not from
# the next app's lock values.

import struct
import sys

LOCKS = {"mem": ([1], 0xC0000, 64), "core": ([2, 3, 4, 5], 0x1F000, 16)}

data = open(sys.argv[1], "rb").read()
value = int(sys.argv[3], 0) if len(sys.argv) > 3 else 63
delay = int(sys.argv[4]) if len(sys.argv) > 4 else 200
w = list(struct.unpack(f"<{len(data) // 4}I", data))
holds = 4 + 32 * 7  # the 32 maskwrite32 (7 words) core holds
assert all((w[4 + 7 * k + 2] & 0xFFFFF) == 0x32000 for k in range(32)), "no 32 core holds at the front"
if len(sys.argv) > 5 and sys.argv[5] == "load":
    # before the first program write instead: after the whole reset part (DMA channels reset too)
    i = 4
    while not (w[i] & 0xFF == 1 and 0x20000 <= (w[i + 2] & 0xFFFFF) < 0x24000):
        i += {0: 6, 3: 7, 0x80: 4, 0x81: 12, 6: 1}.get(w[i] & 0xFF) or w[i + 3] // 4
    holds = i


kinds = (sys.argv[6] if len(sys.argv) > 6 else "mem,core").split(",")


def all_locks(v):
    ops = []
    for rows, base, count in (LOCKS[k] for k in kinds):
        for col in range(8):
            for row in rows:
                for i in range(count):
                    ops += [0, 0, (col << 25) | (row << 20) | (base + 0x10 * i), 0, v, 24]  # write32
    return ops


ops = all_locks(value)
for _ in range(delay):
    ops += [0, 0, (0 << 25) | (1 << 20) | 0xC0000, 0, value, 24]
ops += all_locks(0)
out = w[:holds] + ops + w[holds:]
out[2] = w[2] + len(ops) // 6
out[3] = len(out) * 4
open(sys.argv[2], "wb").write(struct.pack(f"<{len(out)}I", *out))
print(f"{sys.argv[2]}: locks drained with {value} after the core holds")
