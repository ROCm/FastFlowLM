#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Fail the build if any core's stack frames can overrun its stack region.
#
# The AIE stack starts at _sp_start_value_DM_stack and grows upward until it
# reaches the next buffer. The Peano toolchain does not check this (the stack is
# not a section, so ld.lld cannot see it), and an overrun overwrites that buffer
# without any error.
#
# frames is the stack of the deepest call chain (stack_depth.py follows the
# calls; with an indirect call it falls back to the sum of every frame, an upper
# bound). If it fires, raise the stack size in
# iron/layer_aie.py and re-check xclbins/rtp_addresses.txt, since a bigger stack
# can move the RTP buffers that FLM_DLL hardcodes.
set -u

: "${PEANO_INSTALL_DIR:?check_peano_stack.sh needs PEANO_INSTALL_DIR}"
OBJDUMP="${PEANO_INSTALL_DIR}/bin/llvm-objdump"
NM="${PEANO_INSTALL_DIR}/bin/llvm-nm"

rc=0
for f in "$@"; do
    [ -f "$f" ] || continue
    base=$(basename "$f")

    frames=$(python3 "$(dirname "$0")/stack_depth.py" "$OBJDUMP" "$f")
    # A core with no SP adjustment at all uses no stack; nothing to check.
    [ -n "${frames:-}" ] || continue

    start=$("$NM" "$f" 2>/dev/null | awk '/_sp_start_value_DM_stack/{print $1; exit}')
    # llvm-nm -n sorts by address, so the first absolute symbol above the stack
    # start is the buffer that bounds it. Compare as integers: awk would read
    # "00070000" as a number but "0007c000" as a string.
    limit=""
    if [ -n "${start:-}" ]; then
        while read -r addr type _; do
            [ "$type" = "N" ] || continue
            if [ $((0x$addr)) -gt $((0x$start)) ]; then limit=$addr; break; fi
        done < <("$NM" -n "$f" 2>/dev/null)
    fi
    # If the symbols this relies on are missing, warn rather than skip quietly.
    if [ -z "${start:-}" ] || [ -z "$limit" ]; then
        echo "warning: $base: cannot determine the stack bounds, check skipped" >&2
        continue
    fi

    region=$(( 0x$limit - 0x$start ))
    if [ "$frames" -gt "$region" ]; then
        echo "error: $base: stack frames total ${frames} B but only ${region} B of stack" >&2
        echo "       (stack 0x${start} grows up into the buffer at 0x${limit})" >&2
        rc=1
    elif [ $((frames * 100 / region)) -ge 95 ]; then
        echo "warning: $base: stack ${frames}/${region} B ($((frames * 100 / region))% used)" >&2
    fi
done
exit $rc
