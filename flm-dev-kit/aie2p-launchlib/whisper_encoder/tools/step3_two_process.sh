#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# step3_two_process.sh VARIANT [CLIPS] -- step 3 with another process on the
# NPU: checks the NPU (a short stream_probe run), starts tools/chat_load.py (a
# stock chat server answering without pause, its own process), waits 30 s, then
# transcribes the first CLIPS (default 8) English clips once with a Whisper-only
# server of VARIANT (tools/step3_mixed.py VARIANT 1 none), compared with the
# stock transcripts; stops the chat load and waits 60 s. Writes
# build/step3/mixed-two-VARIANT.json. Run from whisper_encoder; needs
# FLM_TEST_DIR and FLM_MODEL_PATH.

set -u
variant=$1 clips=${2:-8}
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
X=build/xroot/xclbins/Whisper-V3-Turbo-NPU2
# A run right after a hang or a killed server fails on a sick NPU: check first.
healthy=0
for try in 1 2 3; do
    good=$(PROBE_FILL=1 timeout 60 build/stream_probe.exe $X/encoder_attn.xclbin build/dump/to_attn_own.txn \
        build/dump/to_attn_own.txn build/dump/to_attn_own.txn 2>&1 | grep -c "buf0 cbfd629396b5b7f5")
    echo "== NPU check $try: $good/3 healthy runs"
    [ "$good" = 3 ] && { healthy=1; break; }
    sleep 30
done
[ $healthy = 1 ] || { echo "the NPU is not healthy: not running"; exit 3; }
python3 tools/chat_load.py 3600 > "build/step3/chat-load-$variant.out" 2>&1 &
load=$!
sleep 30
echo "== $variant, Whisper alone in its server, with the chat load (pid $load) in another process"
STEP3_NAME=two-$variant STEP3_CLIPS=$clips python3 tools/step3_mixed.py "$variant" 1 none
status=$?
kill -TERM $load
wait $load
grep -v "^\[server" "build/step3/chat-load-$variant.out"
sleep 60
exit $status
