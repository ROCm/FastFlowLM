#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# pressure.sh -- long and hard runs of host/stress.cpp on one array (default the
# 32-core 8 x 4, pair layout).
#
#   ./env.sh tools/pressure.sh [OUT_DIR]          (from the vadd_vmul_bf16 folder)
#
#   P1 soak            SEEDS x LAUNCHES random launches (all kinds)
#   P2 long runlists   random walk, runlists of 1..1000 launches
#   P3 big lengths     random walk over 2, 64, 256, 1024 blocks per core
#   P4 latency         one-launch runlists, percentiles per kind
#   P5 two processes   (a) both run robust walks (full switches, cold, reset)
#                      at the same time; (b) a full walk next to a robust one
#   P6 process churn   CHURN times: start, build, short walk, exit
#   P7 context churn   CHURN times in one process: build, hw_context, walk, close
#
# TRANSPORT (default ctrlpkt): how P1-P4 and P6 load their kernels. P5 and P7
# always run txn (another context on the NPU, or ours opened and closed again,
# can take the packets' route: a ctrlpkt kernel that loses it waits on a Sync
# that never ends, and that has not been tested).
#
# One log per part in OUT_DIR (default build/pressure); a summary line each.

set -u
OUT=${1:-build/pressure}
COLS=${COLS:-8}
CORES=${CORES:-4}
SEEDS=${SEEDS:-"11 12 13"}
LAUNCHES=${LAUNCHES:-200000}
CHURN=${CHURN:-50}
TRANSPORT=${TRANSPORT:-ctrlpkt}
mkdir -p "$OUT"

make all COLS=$COLS CORES=$CORES > "$OUT/build.log" 2>&1 || { echo "build failed"; exit 1; }
cfg=$(make -s print-config COLS=$COLS CORES=$CORES | awk '/^config/ {print $3}')
pkg=$(make -s print-config COLS=$COLS CORES=$CORES | awk '/^package/ {print $3}')
make build/$cfg/stress.exe COLS=$COLS CORES=$CORES >> "$OUT/build.log" 2>&1 || { echo "stress build failed"; exit 1; }
exe=build/$cfg/stress.exe

# summary NAME LOG STATUS SECONDS
summary() {
    echo "$1: exit $3, $4 s, $(grep -E '^stress ' "$2" | tail -1)," \
         "$(grep -o 'unexpected: [0-9]* wrong, [0-9]* written past the end' "$2" | tail -1)"
}
timed() { # NAME LOG ARGS...
    local name=$1 log=$2
    shift 2
    local start=$(date +%s)
    "$exe" --package "$pkg" --transport $TRANSPORT "$@" > "$log" 2>&1
    local status=$?
    summary "$name" "$log" $status $(( $(date +%s) - start ))
}

for seed in $SEEDS; do
    timed "P1 soak seed $seed" "$OUT/p1_seed$seed.log" --mode random --seed $seed --launches $LAUNCHES
done
timed "P2 long runlists" "$OUT/p2.log" --mode random --seed 21 --launches 100000 --max-runlist 1000
timed "P3 big lengths" "$OUT/p3.log" --mode random --seed 31 --launches 4000 --max-runlist 16 --lengths 2,64,256,1024
timed "P4 latency" "$OUT/p4.log" --mode latency --count 2000

# P5: two processes at the same time, txn.
start=$(date +%s)
"$exe" --package "$pkg" --transport txn --mode random --robust --seed 51 --launches 50000 > "$OUT/p5a_1.log" 2>&1 &
pid=$!
"$exe" --package "$pkg" --transport txn --mode random --robust --seed 52 --launches 50000 > "$OUT/p5a_2.log" 2>&1
status2=$?
wait $pid
status1=$?
summary "P5a robust + robust, process 1" "$OUT/p5a_1.log" $status1 $(( $(date +%s) - start ))
summary "P5a robust + robust, process 2" "$OUT/p5a_2.log" $status2 $(( $(date +%s) - start ))
start=$(date +%s)
"$exe" --package "$pkg" --transport txn --mode random --robust --seed 53 --launches 50000 > "$OUT/p5b_robust.log" 2>&1 &
pid=$!
"$exe" --package "$pkg" --transport txn --mode random --seed 54 --launches 50000 > "$OUT/p5b_full.log" 2>&1
status2=$?
wait $pid
status1=$?
summary "P5b robust next to a full walk: robust" "$OUT/p5b_robust.log" $status1 $(( $(date +%s) - start ))
summary "P5b robust next to a full walk: full (state-dependent kinds expected to fail)" "$OUT/p5b_full.log" \
    $status2 $(( $(date +%s) - start ))

# P6: process churn.
start=$(date +%s)
failed=0
for i in $(seq 1 "$CHURN"); do
    "$exe" --package "$pkg" --transport $TRANSPORT --mode random --seed $((600 + i)) --launches 500 > "$OUT/p6_last.log" 2>&1 || failed=$((failed + 1))
done
echo "P6 process churn: $CHURN processes, $failed failed, $(( $(date +%s) - start )) s"

TRANSPORT=txn timed "P7 context churn" "$OUT/p7.log" --mode churn --count "$CHURN"
