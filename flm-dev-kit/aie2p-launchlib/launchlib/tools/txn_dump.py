#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# txn_dump.py -- print an AIE2/AIE2P instruction stream (TXN binary) as text.
#
#   txn_dump.py FILE.bin [--data]
#
# The format is the one mlir-aie writes (lib/Targets/AIETargetNPU.cpp,
# AIETranslateNpuToBinary) and the C++ side reads and writes
# (include/launch/txn/encode.hpp). Addresses are shown as column, row, offset
# (bits 31:25, 24:20, 19:0).

import struct
import sys

WRITE, BLOCKWRITE, MASKWRITE, MASKPOLL, PREEMPT, TCT, DDR_PATCH = 0, 1, 3, 4, 6, 0x80, 0x81


def where(addr):
    return f"({addr >> 25},{(addr >> 20) & 0x1F}) {addr & 0xFFFFF:#07x}"


def main():
    data = open(sys.argv[1], "rb").read()
    show_data = "--data" in sys.argv
    w = struct.unpack(f"<{len(data) // 4}I", data)
    major, minor, gen, rows = w[0] & 0xFF, (w[0] >> 8) & 0xFF, (w[0] >> 16) & 0xFF, w[0] >> 24
    print(f"header: version {major}.{minor}, device gen {gen}, {rows} rows, {w[1] & 0xFF} columns, "
          f"{(w[1] >> 8) & 0xFF} mem-tile rows, {w[2]} ops, {w[3]} bytes")
    i = 4
    for n in range(w[2]):
        op = w[i] & 0xFF
        if op == WRITE:
            print(f"{n:4} write32     {where(w[i + 2])} = {w[i + 4]:#010x}")
            i += 6
        elif op == MASKWRITE:
            print(f"{n:4} maskwrite32 {where(w[i + 2])} = {w[i + 4]:#010x} mask {w[i + 5]:#010x}")
            i += 7
        elif op == MASKPOLL:
            print(f"{n:4} maskpoll    {where(w[i + 2])} == {w[i + 4]:#010x} mask {w[i + 5]:#010x}")
            i += 8
        elif op == BLOCKWRITE:
            words = (w[i + 3] // 4) - 4
            payload = w[i + 4:i + 4 + words]
            text = " ".join(f"{x:08x}" for x in payload) if show_data or words <= 8 else f"{words} words"
            print(f"{n:4} blockwrite  {where(w[i + 2])} [{text}]")
            i += 4 + words
        elif op == TCT:
            d, r, c = w[i + 2] & 0xFF, (w[i + 2] >> 8) & 0xFF, (w[i + 2] >> 16) & 0xFF
            rn, cn, ch = (w[i + 3] >> 8) & 0xFF, (w[i + 3] >> 16) & 0xFF, w[i + 3] >> 24
            print(f"{n:4} sync        col {c} row {r} dir {d} chan {ch} (cols {cn}, rows {rn})")
            i += 4
        elif op == DDR_PATCH:
            print(f"{n:4} ddr_patch   {where(w[i + 6])} <- arg {w[i + 8]} + {w[i + 10]} B")
            i += 12
        elif op == PREEMPT:
            print(f"{n:4} preempt     level {w[i] >> 8}")
            i += 1
        else:
            sys.exit(f"op {n}: unknown opcode {op:#x} at word {i}")
    if i != len(w):
        sys.exit(f"{len(w) - i} words after the last op")


if __name__ == "__main__":
    main()
