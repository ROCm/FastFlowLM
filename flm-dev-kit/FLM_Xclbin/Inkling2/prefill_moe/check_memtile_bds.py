#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Post-build check for the runtime-m host sequence (`gen_dequant_mm_rtm` in host/seq_gen.hpp).

Every dispatch sets the memtile x / y paths to its m rows per core row (16 <= m <= 128, multiples of 16),
its Din and its passes P by rewriting a few words of the memtile DMA buffer descriptors that the design
programs for m = 128, Din = 2304, P = 1:
    word 0 (buffer length, 32-bit words)      of every x / y descriptor (x: the two halves of a block)
    word 7 (lock acquire / release values)    of the two x input descriptors (P credits per half)
    word 4 (D2 wrap = number of rows)         of the two y output descriptors
Everything else (strides, base addresses) is left alone.  A channel parked on a descriptor it has already
fetched does not see the rewrite, so the host sequence also resets the x / y channels of every memtile,
puts their locks back to the initial values and pushes each channel's start descriptor again.  This
verifies, in the CDO the build produced, that the descriptors sit at the BD ids / addresses the host code
patches, that they hold the m = 128 values the patch arithmetic assumes, and that the channel start
descriptors and lock initial values are the ones the host re-arm restores.

usage: check_memtile_bds.py <aiecc project dir or its parent>
"""
import glob, os, struct, sys

M = 128
BD_BASE, BD_STRIDE = 0xA0000, 0x20
# (BD ids, word-0 value for m = 128, name); word 0 scales linearly with m
X_HALF = M * 2304 * 9 // 8 // 4 // 2   # words: half of a 128 x 2304 bfp16 block
LEN_TABLE = [
    ((0, 1), X_HALF, "x s2mm (bfp16, half a block)"),
    ((2, 3), X_HALF, "x mm2s (bfp16, half a block)"),
    ((4, 5, 6, 7, 24, 25, 26, 27), M * 24, "y s2mm (one core's rows x 48 n)"),
    ((28, 29), M * 96, "y mm2s (4 cores' rows)"),
]
Y_MM2S = (28, 29)
Y_WRAP_WORD = 4
# x s2mm word 7 for P = 1: (BD, acquire lock, release lock); the host writes it for its P
X_S2MM_W7 = ((0, 0, 1), (1, 2, 3))


def x_s2mm_word7(P, acq, rel):
    """valid, release P on rel, acquire >= P on acq (memtile locks are 0x40 + id)"""
    return 0x80000000 | (P << 24) | ((0x40 + rel) << 16) | ((0x80 | (-P & 0x7F)) << 8) | (0x40 + acq)
Y_S2MM_W4 = (6 << 17) | 3         # y s2mm: D2 wrap 6 (n8 tiles), step 4 words
Y_MM2S_W2 = 24 << 17              # y mm2s: D0 wrap 24 words (48 n)
Y_WRAP_LOW = 0x17      # D2 step - 1 in the low 17 bits; the wrap (rows) sits above


# memtile DMA start-queue register -> start BD id, lock id -> initial value (x / y paths)
START_QUEUE = {0xA0604: 0, 0xA060C: 24, 0xA0614: 4, 0xA061C: 26, 0xA0624: 6, 0xA0634: 2, 0xA063C: 28}
LOCK_INIT = {0: 1, 1: 0, 2: 1, 3: 0, 5: 4, 6: 4, 7: 0, 8: 0}
LOCK_BASE, LOCK_STRIDE = 0xC0000, 0x10


def cdo_writes(path):
    """yield (tile col, tile row, register offset, value) of every plain 64-bit write in a CDO file"""
    w = struct.unpack("<%dI" % (os.path.getsize(path) // 4), open(path, "rb").read())
    i = w[0] + 1
    while i < len(w):
        h = w[i]
        cid, ln = h & 0xFFFF, (h >> 16) & 0xFF
        extra = 0
        if ln == 0xFF:
            ln, extra = w[i + 1], 1
        pay = w[i + 1 + extra:i + 1 + extra + ln]
        if cid == 0x108 and len(pay) == 3:      # write64: addr hi, addr lo, value
            a = pay[1]
            yield (a >> 25) & 0x7F, (a >> 20) & 0x1F, a & 0xFFFFF, pay[2]
        i += 1 + extra + ln


def cdo_blockwrites(path):
    """yield (tile col, tile row, register offset, payload words) of every block write in a CDO file"""
    w = struct.unpack("<%dI" % (os.path.getsize(path) // 4), open(path, "rb").read())
    i = w[0] + 1
    while i < len(w):
        h = w[i]
        cid, ln = h & 0xFFFF, (h >> 16) & 0xFF
        extra = 0
        if ln == 0xFF:
            ln, extra = w[i + 1], 1
        pay = w[i + 1 + extra:i + 1 + extra + ln]
        if cid == 0x105 and len(pay) >= 3:      # block write: addr hi, addr lo, data...
            a = pay[1]
            yield (a >> 25) & 0x7F, (a >> 20) & 0x1F, a & 0xFFFFF, pay[2:]
        i += 1 + extra + ln


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    # the CI mlir-aie names the CDO aie_cdo_init.bin, newer ones main_aie_cdo_init.bin
    hits = glob.glob(os.path.join(root, "**", "main_aie_cdo_init.bin"), recursive=True) or \
           glob.glob(os.path.join(root, "**", "aie_cdo_init.bin"), recursive=True)
    if not hits:
        print("error: no (main_)aie_cdo_init.bin under", root)
        return 1
    bds = {}
    for col, row, off, data in cdo_blockwrites(hits[0]):
        if row == 1 and BD_BASE <= off < BD_BASE + 64 * BD_STRIDE and (off - BD_BASE) % BD_STRIDE == 0 and len(data) >= 8:
            bds[(col, (off - BD_BASE) // BD_STRIDE)] = data[:8]
    bad = 0
    for col in range(8):
        for ids, want, name in LEN_TABLE:
            for b in ids:
                d = bds.get((col, b))
                if d is None or d[0] != want:
                    print(f"error: memtile col {col} BD {b} ({name}): word0 = {None if d is None else hex(d[0])}, expected {hex(want)}")
                    bad += 1
        for b, acq, rel in X_S2MM_W7:
            d = bds.get((col, b))
            if d is not None and d[7] != x_s2mm_word7(1, acq, rel):
                print(f"error: memtile col {col} BD {b} (x s2mm): word7 = {hex(d[7])}, expected {hex(x_s2mm_word7(1, acq, rel))}")
                bad += 1
        for b in (4, 5, 6, 7, 24, 25, 26, 27):
            d = bds.get((col, b))
            if d is not None and d[4] != Y_S2MM_W4:
                print(f"error: memtile col {col} BD {b} (y s2mm): word4 = {hex(d[4])}, expected {hex(Y_S2MM_W4)}")
                bad += 1
        for b in Y_MM2S:
            d = bds.get((col, b))
            if d is not None and d[2] != Y_MM2S_W2:
                print(f"error: memtile col {col} BD {b} (y mm2s): word2 = {hex(d[2])}, expected {hex(Y_MM2S_W2)}")
                bad += 1
        for b in Y_MM2S:
            d = bds.get((col, b))
            if d is not None and d[Y_WRAP_WORD] != ((M << 17) | Y_WRAP_LOW):
                print(f"error: memtile col {col} BD {b}: word4 = {hex(d[Y_WRAP_WORD])}, expected {hex((M << 17) | Y_WRAP_LOW)}")
                bad += 1
    queues, locks = {}, {}
    for col, row, off, val in cdo_writes(hits[0]):
        if row != 1:
            continue
        if off in START_QUEUE:
            queues[(col, off)] = val
        elif LOCK_BASE <= off < LOCK_BASE + 12 * LOCK_STRIDE and (off - LOCK_BASE) % LOCK_STRIDE == 0:
            locks[(col, (off - LOCK_BASE) // LOCK_STRIDE)] = val
    for col in range(8):
        for off, bd in START_QUEUE.items():
            if queues.get((col, off)) != bd:
                print(f"error: memtile col {col} start queue {hex(off)} = {queues.get((col, off))}, expected {bd}")
                bad += 1
        for lk, init in LOCK_INIT.items():
            if locks.get((col, lk)) != init:
                print(f"error: memtile col {col} lock {lk} init = {locks.get((col, lk))}, expected {init}")
                bad += 1
    if bad:
        return 1
    print("check_memtile_bds: 8 memtiles x 14 descriptors, 7 channel start BDs and 8 lock initial values as the host re-arm assumes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
