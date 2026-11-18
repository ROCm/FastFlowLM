#!/bin/bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# build_packages.sh -- the launchlib packages of the Whisper encoder kernels.
#
#   ../vadd_vmul_bf16/env.sh tools/build_packages.sh      (from the whisper_encoder folder)
#
# For each kernel (FLM_Xclbin/Whisper/encoder_MM, encoder_attention_888): the
# Peano kernel object and the IRON MLIR as its own Makefile makes them, then
# aiecc with the flags of its bitstream.mk plus --aie-generate-txn (the
# configure stream launchlib needs), then tools/make_package.py into build/pkg.
# Everything goes to build/ here; the kernel folders get a build/ of their own
# (git-ignored), as `make bitstream` would.

set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
WHISPER=$(cd "$HERE/../../FLM_Xclbin/Whisper" && pwd)
LAUNCHLIB=$(cd "$HERE/../launchlib" && pwd)
AIECC_FLAGS="--aie-generate-cdo --no-compile-host --no-xchesscc --no-xbridge --alloc-scheme basic-sequential"

# build_one APP FOLDER IRON_SCRIPT KERNEL_OBJECT
build_one() {
    local app=$1 folder=$WHISPER/$2 script=$3 object=$4
    local out=$HERE/build/$app
    echo "== $app ($2)"
    (cd "$folder" && make kernel > /dev/null)
    mkdir -p "$out"
    python3 "$folder/iron/$script" > "$out/design.mlir"
    cp "$folder/build/kernel/$object" "$out/"
    (cd "$out" && aiecc.py $AIECC_FLAGS --aie-generate-xclbin --xclbin-name="$app.xclbin" \
        --aie-generate-txn --txn-name="${app}_txn.mlir" design.mlir > aiecc.log 2>&1) \
        || { echo "aiecc failed, see $out/aiecc.log"; exit 1; }
}

build_one mm encoder_MM mm_aie.py mm.o
build_one attn encoder_attention_888 attn_aie.py attn.o
rm -rf "$HERE/build/pkg"
python3 "$LAUNCHLIB/tools/make_package.py" -o "$HERE/build/pkg" mm="$HERE/build/mm" attn="$HERE/build/attn"
