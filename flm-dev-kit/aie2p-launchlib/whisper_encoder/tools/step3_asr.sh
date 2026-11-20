#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# step3_asr.sh VARIANT [TIER] -- step 3: FastFlowLM's ASR test suite (npu_test,
# `flm serve -a 1`: Whisper alone on the NPU) on one FastFlowLM install, the
# FLEURS set of TIER (default full: 84 clips). Run from whisper_encoder.
#
#   stock    build/rt/stock: a FastFlowLM portable install (XRT) as shipped
#   tree     build/rt/tree: the same, with this tree's libwhisper_npu.so and
#            Whisper xclbins (build/xroot); the switch off
#   switch   build/rt/tree with FLM_WHISPER_SWITCH_DIR=build/switch
#
# Needs FLM_TEST_DIR (FastFlowLM's .github/script, which holds npu_test) and
# FLM_MODEL_PATH (a model folder with Whisper-V3-Turbo-NPU2). Output:
# build/step3/VARIANT/ (results.json, summary.md, server logs), or
# build/step3/$STEP3_NAME/.

set -eu
variant=$1 tier=${2:-full}
HERE=$(cd "$(dirname "$0")/.." && pwd)
: "${FLM_TEST_DIR:?set FLM_TEST_DIR to FastFlowLM/.github/script}"
: "${FLM_MODEL_PATH:?set FLM_MODEL_PATH to a model folder with Whisper-V3-Turbo-NPU2}"
export FLM_MODEL_PATH
case $variant in
    stock) rt=$HERE/build/rt/stock ;;
    tree) rt=$HERE/build/rt/tree ;;
    switch) rt=$HERE/build/rt/tree; export FLM_WHISPER_SWITCH_DIR=$HERE/build/switch ;;
    *) echo "unknown variant $variant" >&2; exit 2 ;;
esac
out=$HERE/build/step3/${STEP3_NAME:-$variant}
mkdir -p "$out"
cd "$FLM_TEST_DIR"
python3 -m npu_test --flm-bin "$rt/flm" --label "step3-$variant" --tier "$tier" --suites asr \
    --models whisper-v3:turbo --output-dir "$out"
