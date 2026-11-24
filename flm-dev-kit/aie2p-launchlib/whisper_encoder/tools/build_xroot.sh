#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# build_xroot.sh -- the five Whisper xclbins of this tree, where the engine looks for them.
#
#   ../vadd_vmul_bf16/env.sh tools/build_xroot.sh      (after tools/build_packages.sh)
#
# Writes build/xroot/xclbins/Whisper-V3-Turbo-NPU2/ (the engine reads
# <exec_path>/xclbins/<model folder>/*.xclbin): encoder_mm and encoder_attn are
# the xclbins of build/pkg (the switch files are made from those packages, so
# they must be the same binaries); encoder_dequant, decoder_mv and whisper_head
# come from `make bitstream` in their FLM_Xclbin/Whisper folders.

set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
WHISPER=$(cd "$HERE/../../FLM_Xclbin/Whisper" && pwd)
OUT=$HERE/build/xroot/xclbins/Whisper-V3-Turbo-NPU2
mkdir -p "$OUT"

# bitstream FOLDER IRON_NAME XCLBIN_NAME
bitstream() {
    echo "== $1"
    (cd "$WHISPER/$1" && make bitstream > "$HERE/build/bitstream_$1.log" 2>&1) \
        || { echo "make bitstream failed, see build/bitstream_$1.log"; exit 1; }
    # build/bitstream/from_iron, or build/<MODEL_TYPE>/bitstream/from_iron when the Makefile has a default
    local built
    built=$(find "$WHISPER/$1/build" -path "*bitstream/from_iron/$2.xclbin" | head -1)
    [ -n "$built" ] || { echo "no $2.xclbin under $1/build"; exit 1; }
    cp "$built" "$OUT/$3.xclbin"
}

bitstream encoder_dequant dequant encoder_dequant
bitstream decoder_mv_dequant_mv mv decoder_mv
bitstream whisper_lm_head_npu_bin qmm whisper_head
cp "$HERE/build/pkg/mm/design.xclbin" "$OUT/encoder_mm.xclbin"
cp "$HERE/build/pkg/attn/design.xclbin" "$OUT/encoder_attn.xclbin"
ls -la "$OUT"
