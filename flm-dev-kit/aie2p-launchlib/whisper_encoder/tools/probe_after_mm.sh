#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# probe_after_mm.sh TO_MM.txn [TO_MM.txn ...] -- an experiment, run from
# whisper_encoder: for each TO_MM stream, in a fresh matmul context, gets the
# attention to its healthy state (matmul, two switches to it), runs TO_MM (a
# switch to the matmul), switches back and prints the hash of one attention run.
# The healthy attention hash is cbfd629396b5b7f5.

X=build/xroot/xclbins/Whisper-V3-Turbo-NPU2/encoder_mm.xclbin
M=build/dump/to_mm_own.txn
A=build/dump/to_attn_own.txn
SA=${SWITCH_TO_ATTN:-build/switch/to_attn.txn}
for sm in "$@"; do
    sleep 3
    printf '%-40s ' "$sm"
    PROBE_FILL=1 timeout 100 build/stream_probe.exe "$X" $M "$SA" $A "$SA" $A "$sm" "$SA" $A 2>&1 |
        grep -E '^ ?[0-9]+ ' | awk '{print $NF}' | tr '\n' ' '
    echo
done
