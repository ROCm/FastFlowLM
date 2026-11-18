#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""L1/L2 occupancy of a built design from build/<root>/bitstream/from_iron/<design>.mlir.prj/input_with_addresses.mlir.
usage: occupancy.py <input_with_addresses.mlir>   prints one row per tile: stack, static buffers, highest buffer end (L1 64 KiB, memtile 512 KiB)"""
import re, sys, collections
txt = open(sys.argv[1]).read()
size = {"bf16": 2, "i32": 4, "f32": 4, "i8": 1, "ui8": 1, "i16": 2}
bufs = collections.defaultdict(list)
for m in re.finditer(r'aie\.buffer\(%(?:mem_)?tile_(\d+)_(\d+)\)\s*\{address = (\d+) : i32[^}]*sym_name = "([^"]+)"\}\s*:\s*memref<([0-9x]+)x?(\w+)>', txt):
    c, r, addr, name, shape, ty = m.groups()
    n = 1
    for d in shape.split("x"):
        if d: n *= int(d)
    bufs[(int(c), int(r))].append((int(addr), n * size.get(ty, 2), name))
stacks = {}
for m in re.finditer(r'aie\.core\(%tile_(\d+)_(\d+)\)', txt):
    c, r = int(m.group(1)), int(m.group(2))
    j = txt.find('stack_size', m.end())
    stacks[(c, r)] = int(re.match(r'stack_size = (\d+)', txt[j:j + 30]).group(1)) if j >= 0 else 0
print(f"{'tile (col,row)':16s} {'kind':8s} {'stack':>7s} {'buffers':>8s} {'top of buffers':>15s} {'limit':>7s}  largest")
for (c, r), l in sorted(bufs.items(), key=lambda kv: (kv[0][1] == 1, kv[0][1], kv[0][0])):
    tot = sum(s for _, s, _ in l)
    top = max(a + s for a, s, _ in l)
    kind = "memtile" if r == 1 else ("core" if (c, r) in stacks else "lender")
    lim = 512 * 1024 if r == 1 else 64 * 1024
    big = sorted(l, key=lambda x: -x[1])[:2]
    print(f"({c},{r}){'':11s} {kind:8s} {stacks.get((c, r), 0):7d} {tot:8d} {top:15d} {lim:7d}  " + ", ".join(f"{n}={s}" for _, s, n in big))
