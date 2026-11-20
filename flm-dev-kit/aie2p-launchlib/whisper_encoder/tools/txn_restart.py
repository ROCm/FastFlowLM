#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# txn_restart.py IN.txn OUT.txn [DELAY] -- an experiment: appends, for all 32
# cores, a second start: Core_Control reset on and enable off, reset off, enable
# on (after DELAY copies of a harmless write, default 0: time for the cores to
# run their start code from the new program first).

import struct
import sys

data = open(sys.argv[1], "rb").read()
delay = int(sys.argv[3]) if len(sys.argv) > 3 else 0
w = list(struct.unpack(f"<{len(data) // 4}I", data))
ctrl = lambda col, row: (col << 25) | (row << 20) | 0x32000
tiles = [(c, r) for c in range(8) for r in range(2, 6)]
ops = []
for _ in range(delay):
    ops += [3, 0, ctrl(0, 2), 0, 1, 1, 28]                 # enable stays on: no change
for c, r in tiles:
    ops += [3, 0, ctrl(c, r), 0, 2, 3, 28]                 # reset on, enable off
for c, r in tiles:
    ops += [3, 0, ctrl(c, r), 0, 0, 2, 28]                 # reset off
for c, r in tiles:
    ops += [3, 0, ctrl(c, r), 0, 1, 1, 28]                 # enable on
out = w + ops
out[2] = w[2] + len(ops) // 7
out[3] = len(out) * 4
open(sys.argv[2], "wb").write(struct.pack(f"<{len(out)}I", *out))
print(f"{sys.argv[2]}: second start appended (delay {delay})")
