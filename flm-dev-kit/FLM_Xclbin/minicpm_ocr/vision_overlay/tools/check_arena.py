# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Check the base design's memory: every core's arena where the layout says, and the program's own
data (.data / .bss / .rodata) outside it.

    python3 tools/check_arena.py BASE_ADDRESSES LAYOUT_JSON
"""
import glob
import json
import os
import re
import subprocess
import sys

DM_VIEW = 0x70000   # a core sees its own data memory at 0x70000 + address


def main():
    text = open(sys.argv[1]).read()
    layout = json.load(open(sys.argv[2]))
    lo = layout["arena_base"]
    hi = lo + layout["arena_bytes"]

    arenas = re.findall(r'aie\.buffer\(%tile_(\d+)_(\d+)\) \{address = (\d+) : i32[^}]*sym_name = "arena_', text)
    assert len(arenas) == 32, f"{len(arenas)} arenas"
    bad = [(c, r, int(a)) for c, r, a in arenas if int(a) != lo]
    assert not bad, f"arenas not at 0x{lo:04X}: {bad[:4]}"

    readelf = os.path.join(os.environ["PEANO_INSTALL_DIR"], "bin", "llvm-readelf")
    prj = os.path.dirname(sys.argv[1])
    worst = 0
    for elf in sorted(glob.glob(os.path.join(prj, "core_*.elf"))):
        out = subprocess.run([readelf, "-S", "-W", elf], capture_output=True, text=True, check=True).stdout
        for name, addr, size in re.findall(r'\]\s+(\.\S+)\s+\S+\s+([0-9a-f]{8})\s+[0-9a-f]+\s+([0-9a-f]+)', out):
            a, n = int(addr, 16), int(size, 16)
            if n == 0 or not (DM_VIEW <= a < DM_VIEW + 0x10000):
                continue
            a -= DM_VIEW
            assert a + n <= lo or a >= hi, f"{os.path.basename(elf)}: {name} 0x{a:04X}+{n} is inside the arena"
            worst = max(worst, a + n)
    print(f"arenas at 0x{lo:04X}-0x{hi:04X} on all 32 cores; program data ends at 0x{worst:04X}")


if __name__ == "__main__":
    main()
