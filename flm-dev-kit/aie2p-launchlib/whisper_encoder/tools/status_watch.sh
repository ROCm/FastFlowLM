#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# status_watch.sh OUT_DIR SECONDS -- an AIE status snapshot every second
# (../launchlib/build/aie_status --raw) into OUT_DIR/NNN.txt, for SECONDS.

HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$1"
for ((i = 0; i < $2; i++)); do
    "$HERE/../launchlib/build/aie_status" --raw > "$1/$(printf %03d $i).txt" 2>&1
    sleep 1
done
