# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Print the stack bytes of an AIE ELF's deepest call chain (Peano builds).

    python3 stack_depth.py <llvm-objdump> <elf>

Falls back to the sum of all frames if the ELF has an indirect call.
Same parser as ../../prefill_overlay/tools/check_stack.py.
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
        if m and not m.group(2).startswith(".L"):
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
    starts, frames, calls, indirect = parse(sys.argv[1], sys.argv[2])
    if not frames:
        return
    if indirect:
        print(sum(frames.values()))
    else:
        print(max(deepest(f, starts, frames, calls) for f in frames))


if __name__ == "__main__":
    main()
