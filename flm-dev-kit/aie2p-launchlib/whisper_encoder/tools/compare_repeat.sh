#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# compare_repeat.sh N [WINDOWS] -- run from whisper_encoder after `make compare`:
# runs the engine with the switch (build/switch) N more times and compares each
# output with build/kv_off.bin (today's path). Prints the ms per window and the
# result of each run.

set -u
n=$1 windows=${2:-3}
HERE=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:?set MODEL to the Whisper-V3-Turbo-NPU2 model folder of a FastFlowLM install}
for ((i = 1; i <= n; i++)); do
    out=$HERE/build/kv_on_repeat.bin
    rm -f "$out"
    times=$(FLM_WHISPER_SWITCH_DIR=$HERE/build/switch timeout 300 "$HERE/build/encoder_compare.exe" --model "$MODEL" \
        --xroot "$HERE/build/xroot" --out "$out" --windows "$windows" 2>&1 | awk '/^encode/ {printf "%s ", $3}')
    if cmp -s "$HERE/build/kv_off.bin" "$out"; then result=IDENTICAL; else result=DIFFERENT; fi
    echo "run $i: ms per window $times-> $result"
done
