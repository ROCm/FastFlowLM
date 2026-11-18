#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# S3 gates, end to end: build the all-host engine at DEBUG_LEVEL=1 (which prints the parsed
# config and the descriptor check), regenerate both references, run the engine on S2's
# prompt, and score it. Exits non-zero if anything fails.
#
#   bash FLM_DLL/detail/minicpm_v_4_7_npu/host_ref/run_gate.sh [model_dir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
MODEL="${1:-/path/to/models/MiniCPM-V-4.7-1B-NPU2}"
VENV=/path/to/FastFlowLM_IRON/FLM_Q4NX_Converter/venv/bin/activate
DUMPS="$REPO/models_simple/dumps"
OUT="$DUMPS/minicpm_v_engine"
SPLIT=12

[ -f "$HOME/setup.sh" ] && { set +u; source "$HOME/setup.sh" > /dev/null 2>&1 || true; set -u; }

echo "== build: all five stages on the host, DEBUG_LEVEL=1"
make -C "$HERE/.." -j16 PREFILL_MM=cpu PREFILL_ATTN=cpu PREFILL_GDN=cpu PREFILL_CONV=cpu DECODE=cpu DEBUG_LEVEL=1 > /tmp/minicpm_v_s3_build.log 2>&1 \
    || { tail -30 /tmp/minicpm_v_s3_build.log; exit 1; }
make -C "$HERE" > /dev/null

set +u; source "$VENV"; set -u
echo "== references"
( cd "$REPO/models_simple" && python verify_minicpm_v.py > /tmp/minicpm_v_s3_ref.log 2>&1 ) \
    || { tail -30 /tmp/minicpm_v_s3_ref.log; exit 1; }
python "$HERE/q4_oracle.py" | tail -4

echo "== engine"
rm -rf "$OUT"; mkdir -p "$OUT"
FLM_DUMP_DIR="$OUT" "$REPO/FLM_DLL/build/minicpm_v_4_7_npu/host_ref/run_engine" \
    "$MODEL" "$DUMPS/minicpm_v/input_ids.npy" "$OUT" --split "$SPLIT" --gen 32

echo "== score"
python "$HERE/compare_engine.py" --engine "$OUT" --split "$SPLIT" 2> /dev/null
