#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# txn_join.py FIRST.txn SECOND.txn OUT.txn -- one TXN stream with the ops of
# FIRST, then those of SECOND (header of FIRST, op count and size fixed): what
# the engine's prepend_switch launches, e.g. a switch and the app's own stream.

import struct
import sys


def words(path):
    data = open(path, "rb").read()
    return list(struct.unpack(f"<{len(data) // 4}I", data))


a, b = words(sys.argv[1]), words(sys.argv[2])
out = a + b[4:]
out[2] = a[2] + b[2]
out[3] = len(out) * 4
open(sys.argv[3], "wb").write(struct.pack(f"<{len(out)}I", *out))
print(f"{sys.argv[3]}: {out[2]} ops")
