#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# pressure.sh -- long and hard runs of host_test/stress.cpp with the three apps
# (vadd, vmul, vminus: three floorplans) on COLS columns of 4 cores (default 8:
# 32 cores).
#
#   ./env.sh tools/pressure.sh [OUT_DIR]       (from the pm_runtime_load_singletile folder)
#
# Everything runs through launch::Session (launchlib/include/launch/xrt/session.hpp).
#
#   P0 host test       every row of test.exe, xclbin and ELF flow, 1000 launches per row
#   P1 soak            SEEDS x LAUNCHES random launches (all kinds)
#   P2 long runlists   random walk, runlists of 1..1000 launches
#   P3 lengths + walk  runlists of 1, 23, 24, 25, 48, 200, 1000, the order check, a walk
#   P4 latency         one-launch runs, percentiles per kind (with and without the guard)
#   P5 two processes   (a) both run robust walks (cold and reset only) at the
#                      same time; (b) a full walk next to a robust one: the
#                      Session's sharing guard runs the full walk's warm and rerun
#                      launches as cold kernels, so both must pass
#   P6 process churn   CHURN times: start, build, short walk, exit
#   P7 context churn   CHURN times in one process: Session, walk, close
#   P8 contexts        stress --mode ctx (S5)
#   P9 recovery        stress --mode recover (S4: a wrong prev hangs, the Session recovers)
#   P10 guard          stress --mode guard: a full walk next to a looping context, and
#                      next to contexts that open and close every ~3 ms
#   P11 ELF walk       the walk in the ELF flow (runlists of 1..1000: chunks)
#
# P1-P3 and P11 run with --max-gap 0: alone on the NPU, the walk keeps its warm and
# rerun launches (with the default 2 ms, a gap for the checks makes them cold).
#
# One log per part in OUT_DIR (default build/c<COLS>/pressure); a summary line each.

set -u
COLS=${COLS:-8}
OUT=${1:-build/c$COLS/pressure}
SEEDS=${SEEDS:-"11 12 13"}
LAUNCHES=${LAUNCHES:-200000}
CHURN=${CHURN:-50}
mkdir -p "$OUT"

make all COLS=$COLS > "$OUT/build.log" 2>&1 || { echo "build failed"; exit 1; }
pkg=$(make -s print-config COLS=$COLS | awk '/^package/ {print $3}')
exe=build/c$COLS/stress.exe

# summary NAME LOG STATUS SECONDS
summary() {
    echo "$1: exit $3, $4 s, $(grep -E '^stress ' "$2" | tail -1)," \
         "$(grep -o 'unexpected: [0-9]* wrong, [0-9]* written past the end' "$2" | tail -1)"
}
timed() { # NAME LOG ARGS...
    local name=$1 log=$2
    shift 2
    local start=$(date +%s)
    "$exe" --package "$pkg" "$@" > "$log" 2>&1
    local status=$?
    summary "$name" "$log" $status $(( $(date +%s) - start ))
}

# P0: in the ELF flow the Session runs the 1000 launches of a row as several
# runlists (each ELF run holds a copy of its instructions in the 64 MB device
# heap, and at most ~2,040 runs fit).
for flow in "xclbin 1000" "elf 1000"; do
    set -- $flow
    start=$(date +%s)
    build/c$COLS/test.exe --package "$pkg" --flow $1 --iters $2 > "$OUT/p0_$1.log" 2>&1
    status=$?
    echo "P0 host test, $1 flow, $2 launches per row: exit $status, $(( $(date +%s) - start )) s," \
         "$(grep -c 'PASS$' "$OUT/p0_$1.log") rows PASS, $(grep -c 'FAIL$' "$OUT/p0_$1.log") FAIL"
done

for seed in $SEEDS; do
    timed "P1 soak seed $seed" "$OUT/p1_seed$seed.log" --mode random --seed $seed --launches $LAUNCHES --max-gap 0
done
timed "P2 long runlists" "$OUT/p2.log" --mode random --seed 21 --launches 100000 --max-runlist 1000 --max-gap 0
timed "P3 lengths + walk" "$OUT/p3.log" --mode all --seed 31 --launches 20000 --max-gap 0
timed "P4 latency" "$OUT/p4.log" --mode latency --count 2000
timed "P4 latency, no guard" "$OUT/p4_noguard.log" --mode latency --count 2000 --no-guard

# P5: two processes at the same time, txn (a ctrlpkt kernel next to another
# context is not tested).
start=$(date +%s)
"$exe" --package "$pkg" --transport txn --mode random --robust --seed 51 --launches 20000 > "$OUT/p5a_1.log" 2>&1 &
pid=$!
"$exe" --package "$pkg" --transport txn --mode random --robust --seed 52 --launches 20000 > "$OUT/p5a_2.log" 2>&1
status2=$?
wait $pid
status1=$?
summary "P5a robust + robust, process 1" "$OUT/p5a_1.log" $status1 $(( $(date +%s) - start ))
summary "P5a robust + robust, process 2" "$OUT/p5a_2.log" $status2 $(( $(date +%s) - start ))
start=$(date +%s)
"$exe" --package "$pkg" --transport txn --mode random --robust --seed 53 --launches 20000 > "$OUT/p5b_robust.log" 2>&1 &
pid=$!
"$exe" --package "$pkg" --transport txn --mode random --seed 54 --launches 20000 > "$OUT/p5b_full.log" 2>&1
status2=$?
wait $pid
status1=$?
summary "P5b robust next to a full walk: robust" "$OUT/p5b_robust.log" $status1 $(( $(date +%s) - start ))
summary "P5b robust next to a full walk: full" "$OUT/p5b_full.log" $status2 \
    $(( $(date +%s) - start ))

# P6: process churn.
start=$(date +%s)
failed=0
for i in $(seq 1 "$CHURN"); do
    "$exe" --package "$pkg" --mode random --seed $((600 + i)) --launches 500 > "$OUT/p6_last.log" 2>&1 || failed=$((failed + 1))
done
echo "P6 process churn: $CHURN processes, $failed failed, $(( $(date +%s) - start )) s"

timed "P7 context churn" "$OUT/p7.log" --mode churn --count "$CHURN"
timed "P8 contexts" "$OUT/p8.log" --mode ctx
timed "P9 recovery" "$OUT/p9.log" --mode recover
timed "P10 guard, a looping context" "$OUT/p10_loop.log" --mode guard --child loop --launches 20000
timed "P10 guard, contexts open and close" "$OUT/p10_churn.log" --mode guard --child churn --launches 20000
timed "P11 ELF walk" "$OUT/p11.log" --mode random --flow elf --seed 41 --launches 50000 --max-runlist 1000 --max-gap 0
