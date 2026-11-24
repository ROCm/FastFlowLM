#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# stress_all.sh -- host/stress.cpp on several arrays and seeds.
#
#   ./env.sh tools/stress_all.sh [OUT_DIR]        (from the vadd_vmul_bf16 folder)
#
# For each array (COLSxCORES[-pair]) and seed: mode "all" (S2, S3, S4, S8, S6,
# then the S1 random walk of LAUNCHES launches) and mode "ctx" (S5; it shares the
# NPU, so it runs txn). MODES picks them (default "all ctx"). One log per run in
# OUT_DIR (default build/stress); a summary line per run.

set -u
OUT=${1:-build/stress}
ARRAYS=${ARRAYS:-"1x1 8x2 8x4"}
SEEDS=${SEEDS:-"1 2 3"}
LAUNCHES=${LAUNCHES:-20000}
MODES=${MODES:-"all ctx"}
mkdir -p "$OUT"

for array in $ARRAYS; do
    cols=${array%%x*}
    cores=${array#*x}
    cores=${cores%%-*}
    # The layout as sweep.sh: split for 1-2 cores, pair for 3-4; "-pair" forces pair.
    if [ "$cores" -le 2 ]; then layout=split; else layout=pair; fi
    case $array in *-pair) layout=pair ;; esac
    if ! make all COLS=$cols CORES=$cores LAYOUT=$layout > "$OUT/build_$array.log" 2>&1; then
        echo "$array: build failed"; continue
    fi
    cfg=$(make -s print-config COLS=$cols CORES=$cores LAYOUT=$layout | awk '/^config/ {print $3}')
    pkg=$(make -s print-config COLS=$cols CORES=$cores LAYOUT=$layout | awk '/^package/ {print $3}')
    make build/$cfg/stress.exe COLS=$cols CORES=$cores LAYOUT=$layout >> "$OUT/build_$array.log" 2>&1 || { echo "$array: stress build failed"; continue; }
    for seed in $SEEDS; do
        for mode in $MODES; do
            log=$OUT/${array}_${mode}_seed$seed.log
            start=$(date +%s)
            timeout 3600 build/$cfg/stress.exe --package "$pkg" --mode $mode --seed $seed --launches $LAUNCHES > "$log" 2>&1
            status=$?
            echo "$array seed $seed $mode: exit $status, $(( $(date +%s) - start )) s, $(tail -1 "$log")," \
                 "$(grep -o 'unexpected: [0-9]* wrong, [0-9]* written past the end' "$log")"
        done
    done
done
