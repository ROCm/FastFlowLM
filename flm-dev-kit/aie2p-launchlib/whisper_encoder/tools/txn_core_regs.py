#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# txn_core_regs.py IN.txn OUT.txn [REGS] -- an experiment: after the 32 core
# holds at the front of a switch stream, writes the reset value of core
# registers REGS (default "cr1,cr2,sr,s") in all 32 cores (aie-rt
# xaie2pgbl_params.h: CORE_CR1 0x30E70 = 0x1800, CORE_CR2 0x30E80 = 0x2AA,
# CORE_SR 0x30E90 = 0, CORE_S0..S3 0x31480.. = 0); an entry OFFSET=VALUE
# writes that instead.

import struct
import sys

RESET = {"cr1": [(0x30E70, 0x1800)], "cr2": [(0x30E80, 0x2AA)], "sr": [(0x30E90, 0)],
         "s": [(0x31480 + 0x10 * i, 0) for i in range(4)]}

data = open(sys.argv[1], "rb").read()
regs = (sys.argv[3] if len(sys.argv) > 3 else "cr1,cr2,sr,s").split(",")
w = list(struct.unpack(f"<{len(data) // 4}I", data))
holds = 4 + 32 * 7  # the 32 maskwrite32 (7 words) core holds
assert all((w[4 + 7 * k + 2] & 0xFFFFF) == 0x32000 for k in range(32)), "no 32 core holds at the front"
ops = []
for col in range(8):
    for row in range(2, 6):
        for reg in regs:
            raw = [tuple(int(x, 0) for x in reg.split("="))] if "=" in reg else None
            for off, value in raw or RESET[reg]:
                ops += [0, 0, (col << 25) | (row << 20) | off, 0, value, 24]  # write32
out = w[:holds] + ops + w[holds:]
out[2] = w[2] + len(ops) // 6
out[3] = len(out) * 4
open(sys.argv[2], "wb").write(struct.pack(f"<{len(out)}I", *out))
print(f"{sys.argv[2]}: core registers {','.join(regs)} reset after the core holds")
