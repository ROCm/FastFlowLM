#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# probe_rounds.sh NAME ROUNDS STREAM... -- stream_probe on encoder_mm.xclbin with the
# STREAMs repeated ROUNDS times (other:STREAM runs in a second hw_context on encoder_attn.xclbin) (paths relative to the whisper_encoder folder;
# build/dump/ from FLM_WHISPER_SWITCH_DUMP). Waits 20 s first: after a hang the
# NPU fails the next commands for a while.

set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
name=$1 rounds=$2
shift 2
args=()
for ((r = 0; r < rounds; r++)); do
    for s in "$@"; do
        case $s in
            other:*) args+=("other:$HERE/${s#other:}") ;;
            *) args+=("$HERE/$s") ;;
        esac
    done
done
sleep 20
echo "== $name"
X=$HERE/build/xroot/xclbins/Whisper-V3-Turbo-NPU2
timeout 200 "$HERE/build/stream_probe.exe" "$X/encoder_mm.xclbin" --other "$X/encoder_attn.xclbin" "${args[@]}" 2>&1 | grep -v -E "^Header|^Device|^Cols|^Trans|^NumOps|^UID" | sed "s#$HERE/##"
