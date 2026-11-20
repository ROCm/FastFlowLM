#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# txn_no_enable.py IN.txn OUT.txn -- an experiment: drops every write that sets
# the enable bit of a Core_Control register (0x32000), so the cores of the
# loaded app never start.

import struct
import sys

data = open(sys.argv[1], "rb").read()
w = list(struct.unpack(f"<{len(data) // 4}I", data))
out, i, ops, dropped = w[:4], 4, 0, 0
for _ in range(w[2]):
    op = w[i] & 0xFF
    size = {0: 6, 3: 7, 0x80: 4, 0x81: 12, 6: 1}.get(op) or w[i + 3] // 4
    # only write32 and maskwrite32 can enable a core (a 1-word preempt has no address to read)
    is_ctrl = op in (0, 3) and (w[i + 2] & 0xFFFFF) == 0x32000 and ((w[i + 2] >> 20) & 0x1F) >= 2
    enables = is_ctrl and (w[i + 4] & 1) and (op == 0 or w[i + 5] & 1)
    if enables:
        dropped += 1
    else:
        out += w[i:i + size]
        ops += 1
    i += size
out[2], out[3] = ops, len(out) * 4
open(sys.argv[2], "wb").write(struct.pack(f"<{len(out)}I", *out))
print(f"{sys.argv[2]}: {dropped} core enables dropped")
