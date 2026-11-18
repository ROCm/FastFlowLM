#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# make_package.py -- the package of each app, from its aiecc folder.
#
#   make_package.py -o PKG NAME=DESIGN_FOLDER [NAME=DESIGN_FOLDER ...]
#
# DESIGN_FOLDER is where aiecc ran with
#   --aie-generate-xclbin --aie-generate-npu-insts --aie-generate-txn
# so it holds one *.xclbin, at most one *insts*.bin, one MLIR with
# `runtime_sequence @configure` (the --txn-name file) and one *.prj folder.
# The tool writes PKG/NAME/:
#
#   init.txn       aiecc's configure sequence as a TXN binary (aie-translate
#                  -aie-npu-to-binary): every init write of the design. The C++
#                  loader sorts them per tile and register class.
#   design.xclbin  aiecc's xclbin. The host makes its hw_context from one of
#                  them (launch::Result::base_xclbin).
#   manifest.txt   facts of the MLIR that are not register writes (below)
#   insts.bin      aiecc's runtime sequence, only to check the C++ copy (if
#                  the design has one)
#
# manifest.txt, one fact per line (from <prj>/input_with_addresses.mlir):
#   design NAME
#   tile COL ROW PKT_ID              controller id of a tile (task-complete tokens)
#   core COL ROW                     a core with a program
#   buffer NAME COL ROW ADDRESS BYTES
#   declare COL ROW ID               every aie.lock of the design (named or not)
#   lock NAME COL ROW ID INIT        the named ones
#   alloc NAME DIR CHANNEL COL       shim_dma_allocation (DIR mm2s or s2mm)
#
# and from the cores' ELFs (<prj>/core_COL_ROW.elf, disassembled), the locks
# each core can wait on (kernel/reset.hpp drains only those):
#   census objdump                   the lines below are complete
#   acquire COL ROW LCOL LROW ID     core (COL, ROW) acquires lock ID of tile (LCOL, LROW)
#   acquire_any COL ROW              an acquire whose lock is in a register: any
#                                    lock of the core's tile and its neighbours
# An acquire names its lock by an immediate: 0-15 the south tile's, 16-31 the
# west tile's, 32-47 the north tile's, 48-63 the core's own (mlir-aie
# getLockLocalBaseIndex). Without llvm-objdump for aie2p (PEANO_INSTALL_DIR,
# MLIR_AIE_DIR's llvm-aie, or PATH) there is no census, and the library drains
# every lock of the core tiles, as before.
#
# The tool needs aie-translate on PATH, and runs llvm-objdump.

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

ELEMENT_BITS = {"i4": 4, "ui4": 4, "i8": 8, "ui8": 8, "i16": 16, "ui16": 16, "bf16": 16, "f16": 16, "i32": 32,
                "ui32": 32, "f32": 32, "i64": 64}
# A memref type, its layout (strided<[16, 1]>) and memory space (, 2) included.
MEMREF = r"memref<(?:[^<>]|<[^<>]*>)*>"


def one(pattern, what, folder):
    found = sorted(glob.glob(os.path.join(folder, pattern)))
    if len(found) != 1:
        sys.exit(f"{folder}: expected one {what} ({pattern}), found {len(found)}")
    return found[0]


def txn_mlir(folder):
    found = [p for p in glob.glob(os.path.join(folder, "*.mlir"))
             if "runtime_sequence @configure" in open(p).read()]
    if len(found) != 1:
        sys.exit(f"{folder}: expected one MLIR with runtime_sequence @configure "
                 f"(aiecc --aie-generate-txn), found {len(found)}")
    return found[0]


def memref_bytes(ty):
    """Bytes of a static memref: its shape and element type (the first field);
    a layout or memory space after it does not change the size."""
    m = re.match(r"memref<((?:\d+x)+)(\w+)\s*(?:,|>)", ty)
    if not m:
        raise ValueError(f"memref type {ty}: a static shape and an element type expected")
    if m.group(2) not in ELEMENT_BITS:
        raise ValueError(f"memref type {ty}: element type {m.group(2)} not known")
    n = 1
    for d in m.group(1).rstrip("x").split("x"):
        n *= int(d)
    return (n * ELEMENT_BITS[m.group(2)] + 7) // 8


def manifest(name, mlir):
    text = open(mlir).read()
    tiles = {}  # SSA name -> (col, row)
    lines = [f"design {name}"]
    for ssa, col, row, attrs in re.findall(r"(%\w+) = aie\.tile\((\d+), (\d+)\)(.*)", text):
        tiles[ssa] = (int(col), int(row))
        pkt = re.search(r"controller_id = #aie\.packet_info<pkt_type = \d+, pkt_id = (\d+)>", attrs)
        if pkt:
            lines.append(f"tile {col} {row} {pkt.group(1)}")
    for ssa in re.findall(r"= aie\.core\((%\w+)\)", text):
        lines.append("core {} {}".format(*tiles[ssa]))
    for ssa, attrs, ty in re.findall(r"= aie\.buffer\((%\w+)\) \{([^}]*)\} : (" + MEMREF + ")", text):
        addr = re.search(r"address = (\d+)", attrs)
        sym = re.search(r'sym_name = "(\w+)"', attrs)
        if addr and sym:
            lines.append("buffer {} {} {} {} {}".format(sym.group(1), *tiles[ssa], addr.group(1), memref_bytes(ty)))
    for ssa, lock_id, attrs in re.findall(r"= aie\.lock\((%\w+), (\d+)\)(?: \{([^}]*)\})?", text):
        init = re.search(r"init = (-?\d+)", attrs)
        sym = re.search(r'sym_name = "(\w+)"', attrs)
        lines.append("declare {} {} {}".format(*tiles[ssa], lock_id))
        if sym:
            lines.append("lock {} {} {} {} {}".format(sym.group(1), *tiles[ssa], lock_id,
                                                       init.group(1) if init else 0))
    for sym, direction, channel, col in re.findall(r"aie\.shim_dma_allocation @(\w+)\((MM2S|S2MM), (\d+), (\d+)\)", text):
        lines.append(f"alloc {sym} {direction.lower()} {channel} {col}")
    return "\n".join(lines) + "\n"


ACQUIRE = re.compile(r"\bacq\s+(#0x[0-9a-f]+|\w+)")
NEIGHBOUR = {0: (0, -1), 1: (-1, 0), 2: (0, 1), 3: (0, 0)}  # (immediate // 16: south, west, north, own)


def objdump():
    """llvm-objdump of Peano (it knows aie2p), or None."""
    for root in (os.environ.get("PEANO_INSTALL_DIR"),
                 os.environ.get("MLIR_AIE_DIR") and
                 os.path.join(os.environ["MLIR_AIE_DIR"], "ironenv/lib/python3.12/site-packages/llvm-aie")):
        if root and os.path.exists(os.path.join(root, "bin", "llvm-objdump")):
            return os.path.join(root, "bin", "llvm-objdump")
    return shutil.which("llvm-objdump")


def census(prj, cores):
    """The manifest's census lines: per core, the locks its code acquires (see the top of this file)."""
    tool = objdump()
    if tool is None:
        print(f"{prj}: no llvm-objdump for aie2p: no lock census (the library drains every core lock)",
              file=sys.stderr)
        return []
    lines = ["census objdump"]
    for col, row in cores:
        elf = os.path.join(prj, f"core_{col}_{row}.elf")
        if not os.path.exists(elf):
            sys.exit(f"{prj}: no {os.path.basename(elf)} for the core ({col}, {row})")
        text = subprocess.run([tool, "-d", "--triple=aie2p", elf], capture_output=True, text=True, check=True).stdout
        ids = set()
        for arg in ACQUIRE.findall(text):
            if not arg.startswith("#"):
                lines.append(f"acquire_any {col} {row}")
                break
            ids.add(int(arg[1:], 16))
        for i in sorted(ids):
            if i >= 64:
                sys.exit(f"{elf}: acquire of lock {i:#x}: not a lock of the core or a neighbour")
            dc, dr = NEIGHBOUR[i // 16]
            lines.append(f"acquire {col} {row} {col + dc} {row + dr} {i % 16}")
    return lines


def package(name, folder, out):
    folder = os.path.abspath(folder)
    prj = one("*.prj", "aiecc project folder", folder)
    dest = os.path.join(out, name)
    os.makedirs(dest, exist_ok=True)
    subprocess.run(["aie-translate", "-aie-npu-to-binary", "-aie-sequence-name=configure", txn_mlir(folder),
                    "-o", os.path.join(dest, "init.txn")], check=True)
    shutil.copyfile(one("*.xclbin", "xclbin", folder), os.path.join(dest, "design.xclbin"))
    # insts.bin only checks a C++ runtime sequence against aiecc's; a design
    # without an IRON runtime_sequence (FLM's: the host writes its instructions)
    # has none.
    insts = sorted(glob.glob(os.path.join(folder, "*insts*.bin")))
    if len(insts) > 1:
        sys.exit(f"{folder}: expected at most one instruction stream (*insts*.bin), found {len(insts)}")
    if insts:
        shutil.copyfile(insts[0], os.path.join(dest, "insts.bin"))
    text = manifest(name, os.path.join(prj, "input_with_addresses.mlir"))
    cores = [tuple(map(int, line.split()[1:])) for line in text.splitlines() if line.startswith("core ")]
    with open(os.path.join(dest, "manifest.txt"), "w") as f:
        f.write(text + "".join(line + "\n" for line in census(prj, cores)))
    print(f"{name}: {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("apps", nargs="+", metavar="NAME=DESIGN_FOLDER")
    a = ap.parse_args()
    for spec in a.apps:
        if "=" not in spec:
            sys.exit(f"{spec}: expected NAME=DESIGN_FOLDER")
        name, folder = spec.split("=", 1)
        package(name, folder, a.out)


if __name__ == "__main__":
    main()
