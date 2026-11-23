#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
# bench.sh <flm dir> <label> <FLM_HUGEPAGE_BO> <out dir>: one `flm bench minicpm-v-4.7:1b` (1k..8k text
# prompts, per-layer-kind prefill split), sampling the peak AnonHugePages of the flm process.
F=$1; label=$2; hp=$3; out=$4
mkdir -p "$out/run-$label"
o=$out/$label.txt
echo "hugepage $hp" > "$o"
(cd "$out/run-$label" && FLM_HUGEPAGE_BO=$hp FLM_MINICPM_V_PREFILL_PROFILE=1 \
  timeout 900 "$F/flm" bench minicpm-v-4.7:1b -c 8192 --bench-iterations 2 >> "$o" 2>&1) &
pid=$!
max=0
while kill -0 $pid 2>/dev/null; do
  for p in $(pgrep -f "flm-real bench minicpm-v-4.7:1b" -u "$(id -u)"); do
    v=$(awk '/AnonHugePages/{print $2}' /proc/$p/smaps_rollup 2>/dev/null)
    [ -n "$v" ] && [ "$v" -gt "$max" ] && max=$v
  done
  sleep 2
done
wait $pid
echo "exit $?" >> "$o"
echo "max AnonHugePages kB: $max" >> "$o"
grep -E "TTFT|attn .* linear|rror" "$o" | sed 's/\x1b\[[0-9;]*m//g' | tail -12
