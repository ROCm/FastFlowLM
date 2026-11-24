#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# sweep.sh -- every array and length of the results table, REPS runs each.
#
#   ./env.sh tools/sweep.sh [OUT_DIR]        (from the vadd_vmul_bf16 folder)
#
# A run is COLSxCORES-BLOCKS[-pair]: the array, the blocks each core computes
# per launch (N = 256 * COLS * CORES * BLOCKS; the cores loop, so N is a run-time
# choice and one build serves every length), and the layout (iron/: split by
# default for 1-2 cores, pair for 3-4; "-pair" forces pair). One log per run in
# OUT_DIR (default build/sweep); tools/summarize.py prints the medians.
#
# TRANSPORT (default ctrlpkt, as the README's table; txn for command-processor writes only).

set -u
OUT=${1:-build/sweep}
RUNS=${RUNS:-"1x1-2 1x2-2 2x1-2 4x1-2 4x2-2 8x1-2 8x2-2 1x1-32 1x1-512 8x2-32 8x2-128 \
1x1-2-pair 8x2-2-pair 1x4-2 2x4-2 4x4-2 8x4-2 8x4-32 8x4-128"}
REPS=${REPS:-3}
ITERS=${ITERS:-50}
PAIRS=${PAIRS:-16}
TRANSPORT=${TRANSPORT:-ctrlpkt}
mkdir -p "$OUT"

for run in $RUNS; do
    cols=${run%%x*}
    rest=${run#*x}
    cores=${rest%%-*}
    rest=${rest#*-}
    blocks=${rest%%-*}
    if [ "$cores" -le 2 ]; then layout=split; else layout=pair; fi
    case $run in *-pair) layout=pair ;; esac
    n=$((256 * cols * cores * blocks))
    cfg=n256-c${cols}x${cores}$([ $layout = pair ] && echo -pair)
    if ! make all COLS=$cols CORES=$cores LAYOUT=$layout > "$OUT/build_$cfg.log" 2>&1; then
        echo "$run: build failed (see $OUT/build_$cfg.log)"
        continue
    fi
    pkg=$(make -s print-config COLS=$cols CORES=$cores LAYOUT=$layout | awk '/^package/ {print $3}')
    exe=build/$cfg/vadd_vmul_bf16.exe
    for rep in $(seq 1 "$REPS"); do
        log=$OUT/run_${run}_$rep.log
        timeout 900 "$exe" --package "$pkg" --transport "$TRANSPORT" --n $n --iters "$ITERS" --pairs "$PAIRS" > "$log" 2>&1
        status=$?
        echo "$run rep $rep: exit $status, $(grep -c 'check FAIL' "$log") failed checks," \
             "switch full $(grep '^RESULT switch_full ' "$log" | cut -d' ' -f3)," \
             "pm $(grep '^RESULT switch_pm ' "$log" | cut -d' ' -f3) us"
    done
done
python3 "$(dirname "$0")/summarize.py" "$OUT"
