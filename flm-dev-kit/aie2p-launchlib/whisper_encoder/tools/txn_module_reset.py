#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# txn_module_reset.py IN.txn OUT.txn BITS -- an experiment: puts a pulse of the
# core tiles' Module_Reset_Control (0x60010; bit 2 core module, bit 1 memory
# module, bit 0 stream switch) = BITS, then 0, in all 32 core tiles, in front of
# the stream.

import struct
import sys

bits = int(sys.argv[3], 0)
data = open(sys.argv[1], "rb").read()
w = list(struct.unpack(f"<{len(data) // 4}I", data))
pulse = []
for value in (bits, 0):
    for col in range(8):
        for row in range(2, 6):
            addr = (col << 25) | (row << 20) | 0x60010
            pulse += [0, 0, addr, 0, value, 24]  # write32
out = w[:4] + pulse + w[4:]
out[2] = w[2] + len(pulse) // 6
out[3] = len(out) * 4
open(sys.argv[2], "wb").write(struct.pack(f"<{len(out)}I", *out))
print(f"{sys.argv[2]}: module reset {bits:#x} pulse in front")
