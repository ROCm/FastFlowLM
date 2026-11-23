#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
# bench.sh <flm dir> <label> <out dir> [VAR=value ...]: one `flm bench minicpm-v-4.7:1b` (1k..8k text
# prompts, per-layer-kind prefill split) with the given FLM_HUGEPAGE_BO* settings and the buffer log on,
# sampling the flm process's peak Rss, Pss and AnonHugePages.
F=$1; label=$2; out=$3; shift 3
mkdir -p "$out/run-$label"
o=$out/$label.txt
echo "env $*" > "$o"
(cd "$out/run-$label" && env FLM_HUGEPAGE_BO_LOG=1 FLM_MINICPM_V_PREFILL_PROFILE=1 "$@" \
  timeout 900 "$F/flm" bench minicpm-v-4.7:1b -c 8192 --bench-iterations 2 >> "$o" 2>&1) &
pid=$!
rss=0; pss=0; huge=0
while kill -0 $pid 2>/dev/null; do
  for p in $(pgrep -f "flm-real bench minicpm-v-4.7:1b" -u "$(id -u)"); do
    read -r r s h < <(awk '/^Rss:/{r=$2} /^Pss:/{s=$2} /^AnonHugePages:/{h=$2} END{print r+0, s+0, h+0}' /proc/$p/smaps_rollup 2>/dev/null)
    [ "${r:-0}" -gt "$rss" ] && rss=$r
    [ "${s:-0}" -gt "$pss" ] && pss=$s
    [ "${h:-0}" -gt "$huge" ] && huge=$h
  done
  sleep 2
done
wait $pid
echo "exit $?" >> "$o"
echo "max Rss kB: $rss" >> "$o"
echo "max Pss kB: $pss" >> "$o"
echo "max AnonHugePages kB: $huge" >> "$o"
echo "user-ptr bo ok: $(grep -c 'user-ptr bo ok' "$o"), failed: $(grep -c 'user-ptr bo failed' "$o")" >> "$o"
grep -m1 "user-ptr bo failed" "$o"
grep -E "TTFT|attn .* linear|rror|^max|^user-ptr" "$o" | sed 's/\x1b\[[0-9;]*m//g' | tail -15
