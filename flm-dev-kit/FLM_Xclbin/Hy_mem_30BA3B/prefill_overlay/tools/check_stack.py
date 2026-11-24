# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Fail the build if a core's deepest call chain can overrun its stack (Peano builds).

Like ../makefiles/check_peano_stack.sh, but it follows the call graph instead of
adding every frame in the ELF: the overlay links two kernels that never run at
the same time, so their frames are never on the stack together. See README.md.

    python3 check_stack.py <llvm-objdump> <stack bytes> core_*.elf
"""
import re
import subprocess
import sys

HEAD = re.compile(r"^([0-9a-f]+) <(.+)>:")
FRAME = re.compile(r"padd[a-z]*\s*\[sp\], #(-?0x[0-9a-f]+|-?\d+)")
CALL = re.compile(r"\bjl\s+(#0x[0-9a-f]+|r\d+|p\d+)")


def parse(objdump, elf):
    text = subprocess.run([objdump, "-d", "--no-show-raw-insn", elf],
                          capture_output=True, text=True, check=True).stdout
    starts, frames, calls = {}, {}, {}
    fn = None
    indirect = False
    for line in text.splitlines():
        m = HEAD.match(line)
        if m and not m.group(2).startswith(".L"):      # .L* are labels inside a function
            fn = m.group(2)
            starts[int(m.group(1), 16)] = fn
            frames[fn], calls[fn] = 0, []
            continue
        if m or fn is None:
            continue
        m = FRAME.search(line)
        if m and int(m.group(1), 0) > 0:
            frames[fn] += int(m.group(1), 0)
        for target in CALL.findall(line):
            if target.startswith("#"):
                calls[fn].append(int(target[1:], 16))
            else:
                indirect = True
    return starts, frames, calls, indirect


def deepest(fn, starts, frames, calls, seen=()):
    if fn in seen:
        raise RuntimeError(f"recursion through {fn}")
    below = [deepest(starts[a], starts, frames, calls, seen + (fn,)) for a in calls[fn] if a in starts]
    return frames[fn] + max(below, default=0)


def main():
    objdump, stack = sys.argv[1], int(sys.argv[2])
    worst = 0
    for elf in sys.argv[3:]:
        starts, frames, calls, indirect = parse(objdump, elf)
        need = sum(frames.values()) if indirect else max(deepest(f, starts, frames, calls) for f in frames)
        worst = max(worst, need)
        if need > stack:
            print(f"error: {elf}: deepest call chain needs {need} B of stack, only {stack} B")
            sys.exit(1)
    print(f"stack OK: deepest call chain {worst} B of {stack} B")


if __name__ == "__main__":
    main()
