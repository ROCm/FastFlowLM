#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
# Gate for the NPU path: build the engine with every stage on the host at DEBUG_LEVEL=1, run it on
# a fixed prompt, rebuild it with the Makefile's NPU defaults, run it again, and score the NPU run
# against the host run (compare_engine.py). Exits non-zero if anything fails. Needs an NPU, g++-13,
# XRT and python3 with numpy.
#
#   bash flm-dev-kit/FLM_DLL/detail/minicpm_v_4_7_npu/host_ref/run_gate.sh <model_dir> <xclbin_dir>
#
# <xclbin_dir> is the model's xclbin folder, e.g. src/xclbins/MiniCPM-V-4.7-1B-NPU2. Dumps and logs go
# to $FLM_GATE_OUT (default flm-dev-kit/FLM_DLL/build/minicpm_v_4_7_npu/gate); its cpu/ and npu/ are
# replaced on every run. The engine library is left as the DEBUG_LEVEL=1 NPU build: rebuild it
# with a plain make afterwards.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DLL="$(cd "$HERE/../../.." && pwd)"
MODEL="${1:?usage: run_gate.sh <model_dir> <xclbin_dir>}"
XCLBINS="${2:?usage: run_gate.sh <model_dir> <xclbin_dir>}"
OUT="${FLM_GATE_OUT:-$DLL/build/minicpm_v_4_7_npu/gate}"
RUN_ENGINE="$DLL/build/minicpm_v_4_7_npu/host_ref/run_engine"
SPLIT=12

rm -rf "$OUT/cpu" "$OUT/npu"
mkdir -p "$OUT/cpu" "$OUT/npu"

# The prompt: the chat-template tokens of "Give a short introduction to large language models: what
# they are, how they are trained, and one thing they are bad at." (the model's own tokenizer).
python3 -c "import numpy as np, sys; np.save(sys.argv[1], np.array([248045, 846, 198, 33963, 264, 2716, 16308, 310, 3349, 3992, 3983, 25, 1092, 781, 513, 11, 1204, 781, 513, 15700, 11, 321, 799, 3065, 781, 513, 3743, 506, 13, 248046, 198, 248045, 74455, 198], dtype=np.int32))" \
    "$OUT/input_ids.npy"

run_step() {
    local log="$1"
    shift
    "$@" > "$log" 2>&1 || { tail -30 "$log"; exit 1; }
}

echo "== build: all five stages on the host, DEBUG_LEVEL=1"
run_step "$OUT/build_cpu.log" make -C "$HERE/.." -j16 PREFILL_MM=cpu PREFILL_ATTN=cpu PREFILL_GDN=cpu PREFILL_CONV=cpu DECODE=cpu DEBUG_LEVEL=1
run_step "$OUT/build_host_ref.log" make -C "$HERE"
echo "== run: host"
FLM_DUMP_DIR="$OUT/cpu" run_step "$OUT/run_cpu.log" \
    "$RUN_ENGINE" "$MODEL" "$OUT/input_ids.npy" "$OUT/cpu" --split "$SPLIT" --gen 32

echo "== build: the Makefile's defaults (NPU stages), DEBUG_LEVEL=1"
run_step "$OUT/build_npu.log" make -C "$HERE/.." -j16 DEBUG_LEVEL=1
echo "== run: NPU"
FLM_DUMP_DIR="$OUT/npu" run_step "$OUT/run_npu.log" \
    "$RUN_ENGINE" "$MODEL" "$OUT/input_ids.npy" "$OUT/npu" --split "$SPLIT" --gen 32 --xclbins "$XCLBINS"

echo "== score"
python3 "$HERE/compare_engine.py" --npu "$OUT/npu" --cpu "$OUT/cpu" --split "$SPLIT"
