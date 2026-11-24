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
#
# Anything the check cannot work out (a missing tool or ELF, a stack_depth.py
# failure, symbols it cannot find) is an error, never a skipped core: a check
# that quietly checks nothing looks the same as one that passes.
set -u

err() { echo "error: check_peano_stack.sh: $*" >&2; }

: "${PEANO_INSTALL_DIR:?check_peano_stack.sh needs PEANO_INSTALL_DIR}"
OBJDUMP="${PEANO_INSTALL_DIR}/bin/llvm-objdump"
NM="${PEANO_INSTALL_DIR}/bin/llvm-nm"
DEPTH="$(dirname "$0")/stack_depth.py"
for tool in "$OBJDUMP" "$NM"; do
    [ -x "$tool" ] || { err "$tool not found or not executable"; exit 1; }
done
command -v python3 >/dev/null 2>&1 || { err "python3 not found"; exit 1; }
command -v awk >/dev/null 2>&1 || { err "awk not found"; exit 1; }
[ -f "$DEPTH" ] || { err "$DEPTH not found"; exit 1; }
[ $# -gt 0 ] || { err "no ELF files given"; exit 1; }

rc=0
for f in "$@"; do
    if [ ! -f "$f" ]; then
        err "$f: no such ELF"
        rc=1
        continue
    fi
    base=$(basename "$f")

    if ! frames=$(python3 "$DEPTH" "$OBJDUMP" "$f"); then
        err "$base: stack_depth.py failed (see above)"
        rc=1
        continue
    fi
    if ! [[ $frames =~ ^[0-9]+$ ]]; then
        err "$base: stack_depth.py printed '${frames}', not a byte count"
        rc=1
        continue
    fi
    # A core with no SP adjustment at all uses no stack; nothing to check.
    [ "$frames" -gt 0 ] || continue

    if ! syms=$("$NM" -n "$f" 2>&1); then
        err "$base: llvm-nm failed:"
        printf '%s\n' "$syms" | head -5 >&2
        rc=1
        continue
    fi
    start=$(awk '$NF == "_sp_start_value_DM_stack" {print $1; exit}' <<<"$syms")
    if ! [[ $start =~ ^[0-9a-fA-F]+$ ]]; then
        err "$base: uses ${frames} B of stack but has no _sp_start_value_DM_stack address"
        rc=1
        continue
    fi
    # llvm-nm -n sorts by address, so the first absolute symbol above the stack
    # start is the buffer that bounds it. Compare as integers: awk would read
    # "00070000" as a number but "0007c000" as a string.
    limit=""
    while read -r addr type _; do
        [ "$type" = "N" ] && [[ $addr =~ ^[0-9a-fA-F]+$ ]] || continue
        if [ $((16#$addr)) -gt $((16#$start)) ]; then limit=$addr; break; fi
    done <<<"$syms"
    if [ -z "$limit" ]; then
        err "$base: uses ${frames} B of stack but no buffer above the stack start 0x${start} bounds it"
        rc=1
        continue
    fi

    region=$(( 16#$limit - 16#$start ))
    if [ "$frames" -gt "$region" ]; then
        echo "error: $base: stack frames total ${frames} B but only ${region} B of stack" >&2
        echo "       (stack 0x${start} grows up into the buffer at 0x${limit})" >&2
        rc=1
    elif [ $((frames * 100 / region)) -ge 95 ]; then
        echo "warning: $base: stack ${frames}/${region} B ($((frames * 100 / region))% used)" >&2
    fi
done
exit $rc
