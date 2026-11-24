#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Post-build check: FLM_DLL writes K_div_k / rtp_act at fixed core data memory addresses
(54784 / 54848 on the even rows 2 and 4, 61696 / 61760 on the odd rows 3 and 5).  The
design pins its buffers so they cannot move; this verifies the built ELFs say so.

usage: check_rtp.py <aiecc project dir or its parent>
Reads the symbol tables of the per-core ELFs (llvm-nm from PEANO_INSTALL_DIR); falls back to
the legacy input_with_addresses.mlir grep when a toolchain keeps no ELFs around.
"""
import glob, os, re, subprocess, sys

EXPECTED = {2: (54784, 54848), 4: (54784, 54848), 3: (61696, 61760), 5: (61696, 61760)}


def nm_tool():
    peano = os.environ.get("PEANO_INSTALL_DIR")
    for cand in ([os.path.join(peano, "bin", "llvm-nm")] if peano else []) + ["llvm-nm"]:
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            return cand
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def check_elfs(root, nm):
    bad, seen = 0, 0
    for elf in sorted(glob.glob(os.path.join(root, "**", "*.elf"), recursive=True)):
        m = re.search(r"core_(\d+)_(\d+)", elf)
        if not m:
            continue
        col, row = int(m.group(1)), int(m.group(2))
        if row not in EXPECTED:
            continue
        out = subprocess.run([nm, elf], capture_output=True, text=True, check=True).stdout
        for name, want in zip(("K_div_k", "rtp_act"), EXPECTED[row]):
            hit = re.search(rf"^([0-9a-fA-F]+)\s+\w\s+{name}_{row}_{col}$", out, re.M)
            if not hit:
                print(f"error: {name}_{row}_{col} not in {elf}")
                bad += 1
                continue
            seen += 1
            got = int(hit.group(1), 16) & 0xFFFF
            if got != want:
                print(f"error: {name} on row {row} col {col} is at {got}, FLM_DLL expects {want}")
                bad += 1
    return bad, seen


def check_legacy(root):
    bad, seen = 0, 0
    for f in glob.glob(os.path.join(root, "**", "input_with_addresses.mlir"), recursive=True):
        for addr, name, row in re.findall(r'address = (\d+) : i32, sym_name = "(K_div_k|rtp_act)_(\d+)_\d+"', open(f).read()):
            seen += 1
            want = EXPECTED[int(row)][0 if name == "K_div_k" else 1]
            if int(addr) != want:
                print(f"error: {name} on row {row} is at {addr}, FLM_DLL expects {want}")
                bad += 1
    return bad, seen


if __name__ == "__main__":
    root = sys.argv[1]
    nm = nm_tool()
    bad, seen = check_elfs(root, nm) if nm else (0, 0)
    if seen == 0:
        bad, seen = check_legacy(root)
    if seen == 0:
        print("error: no K_div_k / rtp_act symbols found to check")
        sys.exit(1)
    if bad:
        sys.exit(1)
    print(f"check_rtp: {seen} RTP buffers at the FLM_DLL addresses")
