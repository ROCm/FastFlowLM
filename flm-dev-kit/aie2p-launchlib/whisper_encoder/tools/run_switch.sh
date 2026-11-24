#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# run_switch.sh SWITCH_DIR OUT [WINDOWS] -- encoder_compare with FLM_WHISPER_SWITCH_DIR=SWITCH_DIR
# (from the whisper_encoder folder; SWITCH_DIR relative to it).

set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:?set MODEL to the Whisper-V3-Turbo-NPU2 model folder of a FastFlowLM install}
FLM_WHISPER_SWITCH_DIR=$HERE/$1 exec "$HERE/build/encoder_compare.exe" --model "$MODEL" --xroot "$HERE/build/xroot" \
    --out "$HERE/$2" --windows "${3:-3}"
