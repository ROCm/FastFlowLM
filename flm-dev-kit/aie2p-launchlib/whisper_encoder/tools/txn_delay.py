#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# txn_delay.py JOINED.txn PREFIX.txn COUNT OUT.txn
#
# A probe for timing: JOINED is PREFIX's ops followed by an op stream (the
# engine's joined stream). Writes OUT = the same, with COUNT copies of the
# first op after the prefix inserted right after the prefix (a write32 of a
# value the stream writes there anyway), so the command processor spends time
# between the switch and the op's own instructions.

import struct
import sys

# op sizes in words by opcode (launchlib/include/launch/txn/txn.hpp)
WRITE32, BLOCKWRITE, MASKWRITE32 = 0, 1, 3


def op_words(words, i):
    """Length in words of the op starting at words[i]."""
    opcode = words[i] & 0xFF
    if opcode == WRITE32:
        return 6
    if opcode == MASKWRITE32:
        return 7
    if opcode == BLOCKWRITE:
        return words[i + 3] // 4  # the op's size in bytes, header included
    raise SystemExit(f"op {opcode:#x} at word {i}: not handled here")


def main():
    joined_path, prefix_path, count, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    joined = list(struct.unpack(f"<{len(open(joined_path, 'rb').read()) // 4}I", open(joined_path, "rb").read()))
    prefix = open(prefix_path, "rb").read()
    prefix_ops = struct.unpack("<I", prefix[8:12])[0]
    i = 4
    for _ in range(prefix_ops):
        i += op_words(joined, i)
    first = joined[i:i + op_words(joined, i)]
    if first[0] & 0xFF != WRITE32:
        raise SystemExit("the first op after the prefix is not a write32")
    out = joined[:i] + first * count + joined[i:]
    out[2] += count
    out[3] = len(out) * 4
    open(out_path, "wb").write(struct.pack(f"<{len(out)}I", *out))
    print(f"{out_path}: {count} x write32 inserted after {prefix_ops} prefix ops")


main()
