#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Post-build check for the host-programmed memtile weight path (host/seq_gen.hpp).

The design leaves memtile S2MM0 (weight panel in), MM2S0 (lo halves out) and MM2S1 (hi halves out)
without descriptors; the host sequence writes their descriptors, panel locks and start queues on every
call. This verifies, in the CDO the build produced, that the static y path did not take the descriptor
ids the host writes, that the three channels get no start descriptor, that the panel locks start at 0,
and that the static y descriptors use the base-address and lock-id encoding the host code assumes.

usage: check_memtile_bds.py <aiecc project dir or its parent> [--dump]
"""
import glob, os, struct, sys

BD_BASE, BD_STRIDE = 0xA0000, 0x20
LOCK_BASE, LOCK_STRIDE = 0xC0000, 0x10
# keep in sync with host/seq_gen.hpp
W_BDS = (18, 19, 20, 21, 22, 23, 44, 45, 46, 47)  # S2MM0 panel 0 / 1, MM2S0 + MM2S1 panel 0 / 1 (+ hybrid)
PANEL_LOCKS = (8, 9, 10, 11)
W_START_QUEUES = (0xA0604, 0xA0634, 0xA063C)  # S2MM0, MM2S0, MM2S1
MT_OWN_BASE_WORDS = 0x80000 // 4              # memtile own memory in the descriptor address space
MT_OWN_LOCK = 64                              # memtile own locks in the descriptor lock-id space
Y_ADDR = 0                                    # iron/gemm_tile.py MT_Y_ADDR


def commands(path):
    w = struct.unpack("<%dI" % (os.path.getsize(path) // 4), open(path, "rb").read())
    i = w[0] + 1
    while i < len(w):
        h = w[i]
        cid, ln = h & 0xFFFF, (h >> 16) & 0xFF
        extra = 0
        if ln == 0xFF:
            ln, extra = w[i + 1], 1
        yield cid, w[i + 1 + extra:i + 1 + extra + ln]
        i += 1 + extra + ln


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    dump = "--dump" in sys.argv
    hits = glob.glob(os.path.join(root, "**", "main_aie_cdo_init.bin"), recursive=True) or \
           glob.glob(os.path.join(root, "**", "aie_cdo_init.bin"), recursive=True)
    if not hits:
        print("error: no (main_)aie_cdo_init.bin under", root)
        return 1
    bds, queues, locks = {}, {}, {}
    for path in hits[:1]:
        for cid, pay in commands(path):
            if cid == 0x105 and len(pay) >= 3:
                a, data = pay[1], pay[2:]
                col, row, off = (a >> 25) & 0x7F, (a >> 20) & 0x1F, a & 0xFFFFF
                if row == 1 and BD_BASE <= off < BD_BASE + 48 * BD_STRIDE and (off - BD_BASE) % BD_STRIDE == 0:
                    bds[(col, (off - BD_BASE) // BD_STRIDE)] = data[:8]
            elif cid == 0x108 and len(pay) == 3:
                a, val = pay[1], pay[2]
                col, row, off = (a >> 25) & 0x7F, (a >> 20) & 0x1F, a & 0xFFFFF
                if row != 1:
                    continue
                if 0xA0600 <= off < 0xA0700 and off % 8 == 4:
                    queues[(col, off)] = val
                elif LOCK_BASE <= off < LOCK_BASE + 64 * LOCK_STRIDE and (off - LOCK_BASE) % LOCK_STRIDE == 0:
                    locks[(col, (off - LOCK_BASE) // LOCK_STRIDE)] = val
    if dump:
        for (col, b), d in sorted(bds.items()):
            if col == 0:
                print(f"bd {b:2d}: " + " ".join(f"{x:08x}" for x in d))
        for (col, off), v in sorted(queues.items()):
            if col == 0:
                print(f"start queue {off:#x} = {v:#x}")
        for (col, lk), v in sorted(locks.items()):
            if col == 0:
                print(f"lock {lk} init {v}")
    bad = 0
    for col in range(8):
        for b in W_BDS:
            if (col, b) in bds:
                print(f"error: memtile col {col} BD {b} is used by the design; the host writes it")
                bad += 1
        for q in W_START_QUEUES:
            if (col, q) in queues:
                print(f"error: memtile col {col} start queue {q:#x} is pushed by the design")
                bad += 1
        for lk in PANEL_LOCKS:
            if locks.get((col, lk), 0) != 0:
                print(f"error: memtile col {col} panel lock {lk} init {locks[(col, lk)]}, expected 0")
                bad += 1
        # the y S2MM descriptors: base = MT_OWN_BASE_WORDS + Y_ADDR / 4 + strip offset, locks own 0-7 (+ MT_OWN_LOCK)
        ys = [d for (c, b), d in bds.items() if c == col]
        bases = sorted((d[1] & 0x7FFFF) for d in ys)
        if not bases or bases[0] != MT_OWN_BASE_WORDS + Y_ADDR // 4:
            print(f"error: memtile col {col}: lowest descriptor base {bases[:1]}, expected {MT_OWN_BASE_WORDS + Y_ADDR // 4:#x}")
            bad += 1
        lock_ids = sorted({d[7] & 0xFF for d in ys} | {(d[7] >> 16) & 0xFF for d in ys})
        if lock_ids != [MT_OWN_LOCK + i for i in range(8)]:
            print(f"error: memtile col {col}: descriptor lock ids {lock_ids}, expected {MT_OWN_LOCK}..{MT_OWN_LOCK + 7}")
            bad += 1
    if bad:
        return 1
    print(f"check_memtile_bds: 8 memtiles leave BDs {W_BDS}, the weight start queues and panel locks to the host")
    return 0


if __name__ == "__main__":
    sys.exit(main())
