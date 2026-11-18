#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# Extract the byte address of the rtp_* buffer from a generated design.
#
# The address is assigned by the compiler and changes whenever the design
# changes. A stale hardcoded address does not fault - the cores read garbage for
# their loop counts and the run hangs in wait(). So the host reads it from here
# rather than from a constant.
#
# Usage: tools/extract_rtp_addr.sh VISION_MM_A
# Writes build/<design>/rtp_addr.txt
set -euo pipefail

cd "$(dirname "$0")/.."

DESIGN="${1:-VISION_MM_A}"

MLIR=$(find "build/${DESIGN}" -name 'input_with_addresses.mlir' 2>/dev/null | head -1 || true)
if [[ -z "${MLIR}" ]]; then
    echo "error: input_with_addresses.mlir not found under build/${DESIGN}; run 'make bitstream MODEL_TYPE=${DESIGN}' first" >&2
    exit 1
fi

# aie.buffer(%tile_0_2) {address = 56832 : i32, sym_name = "rtp_2_0"}
ADDR=$(grep 'sym_name = "rtp_2_0"' "${MLIR}" | grep -o 'address = [0-9]*' \
       | grep -o '[0-9]*' | head -1 || true)

if [[ -z "${ADDR}" ]]; then
    echo "error: could not find an address for rtp_2_0 in ${MLIR}" >&2
    grep 'rtp_2_0' "${MLIR}" >&2 || true
    exit 1
fi

# Every core's rtp buffer must sit at the same offset - the sequence writes one
# address to all 32 tiles. Fail loudly rather than hang if they ever diverge.
BAD=0
while read -r sym other_addr; do
    if [[ "${other_addr}" != "${ADDR}" ]]; then
        echo "error: ${sym} is at ${other_addr}, but rtp_2_0 is at ${ADDR}" >&2
        BAD=1
    fi
done < <(grep -o 'address = [0-9]* : i32, sym_name = "rtp_[0-9_]*"' "${MLIR}" \
         | sed 's/address = \([0-9]*\) : i32, sym_name = "\(rtp_[0-9_]*\)"/\2 \1/' | sort -u)
[[ "${BAD}" == "0" ]] || exit 1

mkdir -p "build/${DESIGN}"
echo "${ADDR}" > "build/${DESIGN}/rtp_addr.txt"
echo "${DESIGN}: RTP address = ${ADDR}  (from ${MLIR}) -> build/${DESIGN}/rtp_addr.txt"
