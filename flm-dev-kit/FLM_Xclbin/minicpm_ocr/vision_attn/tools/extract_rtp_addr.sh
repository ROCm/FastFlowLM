#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# Extract the byte address of the L_attn_core_* RTP buffer from the generated
# design. This is assigned by the compiler and changes whenever the design
# changes - a stale hardcoded address does not fault, it hangs in wait().
#
# Writes the address to build/rtp_addr.txt, which host.cpp reads by default.
set -euo pipefail

cd "$(dirname "$0")/.."

MLIR=$(ls build/bitstream/from_iron/input_with_addresses.mlir 2>/dev/null | head -1 || true)
if [[ -z "${MLIR}" ]]; then
    MLIR=$(find build -name 'input_with_addresses.mlir' 2>/dev/null | head -1 || true)
fi
if [[ -z "${MLIR}" ]]; then
    echo "error: input_with_addresses.mlir not found; run 'make bitstream' first" >&2
    exit 1
fi

# aie.buffer(%tile_0_2) {address = 10496 : i32, sym_name = "L_attn_core_2_0"}
ADDR=$(grep -o 'address = [0-9]* : i32[^\n]*sym_name = "L_attn_core_2_0"' "${MLIR}" \
       | grep -o 'address = [0-9]*' | grep -o '[0-9]*' | head -1 || true)

if [[ -z "${ADDR}" ]]; then
    # attribute order is not guaranteed; fall back to the whole line
    ADDR=$(grep 'L_attn_core_2_0' "${MLIR}" | grep -o 'address = [0-9]*' \
           | grep -o '[0-9]*' | head -1 || true)
fi

if [[ -z "${ADDR}" ]]; then
    echo "error: could not find an address for L_attn_core_2_0 in ${MLIR}" >&2
    grep 'L_attn_core_2_0' "${MLIR}" >&2 || true
    exit 1
fi

mkdir -p build
echo "${ADDR}" > build/rtp_addr.txt
echo "L RTP address = ${ADDR}  (from ${MLIR}) -> build/rtp_addr.txt"
