#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# cdo_dump.py FILE.bin [FILE.bin ...] [--diff-txn INIT.txn | --diff-values INIT.txn]
#
# Prints the commands of aiecc's CDO files (<design>.prj/aie_cdo_{elfs,init,
# enable}.bin, what the xclbin's PDI loads) as (col,row) offset = value. With
# --diff-txn, prints instead the register writes of the CDO whose address the
# TXN configure stream (make_package.py's init.txn) never writes, by tile kind
# and offset: what a hw_context load does that the launch kernels do not. With
# --diff-values, the addresses both write whose final values differ (each file
# applied in order to a register model that starts at 0).
#
# CDO v2: 5 header words, then commands; word 0 of a command = id | length << 16
# (length 255: the next word holds it). Ids used here: 0x107 mask write
# (addr hi, addr lo, mask, value), 0x108 write (addr hi, addr lo, value), 0x109
# and 0x105 DMA write (addr hi, addr lo, words...), 0x111 (no operation) and others printed raw.

import collections
import struct
import subprocess
import sys
import os

COL_SHIFT, ROW_SHIFT = 25, 20


def where(addr):
    return (addr >> COL_SHIFT) & 0x7F, (addr >> ROW_SHIFT) & 0x1F, addr & 0xFFFFF


def cdo_writes(path):
    data = open(path, "rb").read()
    w = struct.unpack(f"<{len(data) // 4}I", data)
    i, out = 5, []
    while i < len(w):
        cmd, length = w[i] & 0xFFFF, (w[i] >> 16) & 0xFF
        i += 1
        if length == 255:
            length, i = w[i], i + 1
        body = w[i:i + length]
        i += length
        if cmd == 0x107:
            out.append(("mask", body[1], body[3], body[2]))
        elif cmd == 0x108:
            out.append(("write", body[1], body[2], 0xFFFFFFFF))
        elif cmd in (0x105, 0x109):
            for k, v in enumerate(body[2:]):
                out.append(("block", body[1] + 4 * k, v, 0xFFFFFFFF))
        else:
            out.append((f"cmd{cmd:#x}", 0, length, 0))
    return out


def txn_addresses(path):
    tool = os.path.join(os.path.dirname(__file__), "txn_dump.py")
    text = subprocess.run(["python3", tool, path], capture_output=True, text=True).stdout
    addrs = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[1] in ("write32", "maskwrite32", "blockwrite"):
            col, row = map(int, parts[2].strip("()").split(","))
            off = int(parts[3], 16)
            base = (col << COL_SHIFT) | (row << ROW_SHIFT) | off
            n = 1
            if parts[1] == "blockwrite":  # "[N words]", or the words themselves in brackets
                inside = line[line.index("[") + 1:line.rindex("]")].split()
                n = int(inside[0]) if inside[-1] == "words" else len(inside)
            for k in range(n):
                addrs.add(base + 4 * k)
    return addrs


def txn_model(path):
    """Final value of every address the TXN binary writes."""
    data = open(path, "rb").read()
    w = struct.unpack(f"<{len(data) // 4}I", data)
    m, i = {}, 4
    for _ in range(w[2]):
        op = w[i] & 0xFF
        if op == 0:      # write32
            m[w[i + 2]] = w[i + 4]
            i += 6
        elif op == 3:    # maskwrite32
            a, v, mask = w[i + 2], w[i + 4], w[i + 5]
            m[a] = (m.get(a, 0) & ~mask) | (v & mask)
            i += 7
        elif op == 1:    # blockwrite
            words = w[i + 3] // 4 - 4
            for k in range(words):
                m[w[i + 2] + 4 * k] = w[i + 4 + k]
            i += 4 + words
        elif op == 0x80:  # sync
            i += 4
        elif op == 0x81:  # ddr patch
            i += 12
        elif op == 6:     # preempt
            i += 1
        else:
            sys.exit(f"{path}: unknown op {op:#x}")
    return m


def cdo_model(writes):
    m = {}
    for kind, addr, val, mask in writes:
        if kind.startswith("cmd"):
            continue
        m[addr] = (m.get(addr, 0) & ~mask) | (val & mask)
    return m


def main():
    args = sys.argv[1:]
    if "--diff-values" in args:
        k = args.index("--diff-values")
        txn = txn_model(args[k + 1])
        del args[k:k + 2]
        cdo = cdo_model([x for f in args for x in cdo_writes(f)])
        both = [a for a in cdo if a in txn]
        diff = [a for a in sorted(both) if cdo[a] != txn[a]]
        print(f"{len(both)} addresses in both, {len(diff)} with other values")
        for a in diff[:int(os.environ.get("CDO_DIFF_LIMIT", "40"))]:
            col, row, off = where(a)
            print(f"  ({col},{row}) {off:#07x}: cdo {cdo[a]:#010x}  txn {txn[a]:#010x}")
        return
    txn = None
    if "--diff-txn" in args:
        k = args.index("--diff-txn")
        txn = args[k + 1]
        del args[k:k + 2]
    writes = [x for f in args for x in cdo_writes(f)]
    if txn is None:
        for kind, addr, val, mask in writes:
            col, row, off = where(addr)
            print(f"{kind:6} ({col},{row}) {off:#07x} = {val:#010x}" + (f" mask {mask:#010x}" if kind == "mask" else ""))
        return
    have = txn_addresses(txn)
    only = collections.defaultdict(list)
    for kind, addr, val, mask in writes:
        if kind.startswith("cmd"):
            only[("command", kind)].append(val)
            continue
        if addr not in have:
            col, row, off = where(addr)
            tile = "shim" if row == 0 else "mem" if row == 1 else "core"
            only[(tile, f"{off:#07x}")].append((col, row, val, kind))
    for key, items in sorted(only.items()):
        print(key, len(items), items[:4])


main()
