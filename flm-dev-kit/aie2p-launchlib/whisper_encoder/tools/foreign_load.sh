#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# foreign_load.sh N XCLBIN STREAM -- another process on the NPU: stream_probe
# runs STREAM N times in its own hw_context on XCLBIN, with hashes (PROBE_FILL),
# as load next to a test in another process. Run from whisper_encoder.

set -u
n=$1 xclbin=$2 stream=$3
args=()
for ((i = 0; i < n; i++)); do args+=("$stream"); done
PROBE_FILL=1 exec build/stream_probe.exe "$xclbin" "${args[@]}"
