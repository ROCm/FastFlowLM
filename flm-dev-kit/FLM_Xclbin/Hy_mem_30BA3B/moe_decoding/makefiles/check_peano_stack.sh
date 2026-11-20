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
# frames is the sum of every positive SP adjustment in the ELF, an upper bound
# on the deepest call chain. It over-counts, so the check can reject a build
# that would have fit, but not the reverse. If it fires, raise the stack size in
# iron/layer_aie.py and re-check xclbins/rtp_addresses.txt, since a bigger stack
# can move the RTP buffers that FLM_DLL hardcodes.
#
# Anything the check cannot work out (a missing tool or ELF, disassembly or
# symbols it cannot parse) is an error, never a skipped core: a check that
# quietly checks nothing looks the same as one that passes. Only bash, grep and
# awk are used (no bc), so it runs in the kernel build image as it is.
set -u

err() { echo "error: check_peano_stack.sh: $*" >&2; }

: "${PEANO_INSTALL_DIR:?check_peano_stack.sh needs PEANO_INSTALL_DIR}"
OBJDUMP="${PEANO_INSTALL_DIR}/bin/llvm-objdump"
NM="${PEANO_INSTALL_DIR}/bin/llvm-nm"
for tool in "$OBJDUMP" "$NM"; do
    [ -x "$tool" ] || { err "$tool not found or not executable"; exit 1; }
done
for tool in grep awk; do
    command -v "$tool" >/dev/null 2>&1 || { err "$tool not found"; exit 1; }
done
[ $# -gt 0 ] || { err "no ELF files given"; exit 1; }

# Sum of the positive immediates of every "padd* [sp], #imm" in a disassembly
# on stdin. Negative immediates are the epilogues giving the frame back. Prints
# the sum, or prints each SP adjustment it cannot read and fails.
sum_frames() {
    local line sign num total=0 bad=0
    while IFS= read -r line; do
        if [[ $line =~ padd[a-z]*[[:space:]]*\[sp\],[[:space:]]*#(-?)(0x[0-9a-fA-F]+|[0-9]+)([^0-9a-zA-Z]|$) ]]; then
            sign=${BASH_REMATCH[1]}
            num=${BASH_REMATCH[2]}
            # Decimal goes through 10# so a leading zero is not read as octal.
            if [[ $num == 0x* ]]; then num=$((num)); else num=$((10#$num)); fi
            [ -n "$sign" ] || total=$((total + num))
        else
            echo "cannot parse SP adjustment: $line" >&2
            bad=1
        fi
    done < <(grep -E 'padd[a-z]*[[:space:]]*\[sp\]')
    [ "$bad" -eq 0 ] || return 1
    echo "$total"
}

rc=0
for f in "$@"; do
    if [ ! -f "$f" ]; then
        err "$f: no such ELF"
        rc=1
        continue
    fi
    base=$(basename "$f")

    if ! dis=$("$OBJDUMP" -d "$f" 2>&1); then
        err "$base: llvm-objdump failed:"
        printf '%s\n' "$dis" | head -5 >&2
        rc=1
        continue
    fi
    # A disassembly without a single function is not one this script can read.
    if ! grep -qE '^[0-9a-fA-F]+ <[^>]+>:' <<<"$dis"; then
        err "$base: no functions in the llvm-objdump -d output"
        rc=1
        continue
    fi
    if ! frames=$(sum_frames <<<"$dis"); then
        err "$base: cannot read the SP adjustments above"
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
