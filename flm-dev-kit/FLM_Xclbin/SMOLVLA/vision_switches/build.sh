#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
#
# The warm switches of the SmolVLA vision tower (see ../../../FLM_Pybind/SmolVLA/src/SmolVLA/vlm/vision/seq_gen.hpp):
# the 8-column matmul (vision_mm8) and the 32-core attention (vision_attn32) built as aie2p-launchlib packages,
# and from them the switch mm8 -> attn32 and back, once as instruction streams and once as control packets,
# packed into one file, vision.switches.
#
#   build.sh <out dir>        out/: vision.switches, vision_switches_base.xclbin (the mm8 build's xclbin)
#
# Every app is built as its own unit builds it (the unit's `make kernel`, its IRON design, aiecc with the unit's
# bitstream.mk flags) plus --aie-generate-txn; the toolchain is the one on PATH (the aie-build image, or the
# environment of FLM_Xclbin/README.md). The engine's hardware context is the vision_mm8 unit's xclbin.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
X=$(cd "$HERE/../.." && pwd)
LL=$(cd "$X/../aie2p-launchlib/launchlib" && pwd)
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
WORK=$HERE/build/work
FLAGS="--aie-generate-cdo --no-compile-host --no-xchesscc --no-xbridge"

# build_one APP UNIT IRON_SCRIPT
build_one() {
    local app=$1 unit=$X/SMOLVLA/$2 script=$3
    local out=$WORK/$app
    echo "== $app (SMOLVLA/$2)"
    (cd "$unit" && rm -rf build && make MODEL_TYPE=0 kernel)
    mkdir -p "$out"
    python3 "$unit/iron/$script" 0 > "$out/design.mlir"
    cp "$unit"/build/0/kernel/*.o "$out/"
    (cd "$out" && aiecc.py $FLAGS --aie-generate-xclbin --xclbin-name="$app.xclbin" \
        --aie-generate-txn --txn-name="${app}_txn.mlir" design.mlir)
    # aiecc names the ELFs main_core_C_R.elf; make_package.py wants core_C_R.elf
    for f in "$out"/design.mlir.prj/main_core_*.elf; do ln -sf "$(basename "$f")" "$out/design.mlir.prj/${f##*/main_}"; done
}

rm -rf "$WORK" && mkdir -p "$WORK"
build_one mm8    vision_mm8    vision_mm8_aie.py
build_one attn32 vision_attn32 attn_aie.py

python3 "$LL/tools/make_package.py" -o "$WORK/pkg" mm8="$WORK/mm8" attn32="$WORK/attn32"

# The switch generator needs no NPU and no XRT.
${CXX:-g++} -std=c++20 -O2 -Wall -I "$LL/include" -I "$X/../FLM_DLL/include" "$HERE/switch_files.cpp" -o "$WORK/switch_files"
"$WORK/switch_files" --package "$WORK/pkg" --out "$OUT"

# The base xclbin the switches were built against (the engine's context is the vision_mm8 unit's xclbin).
cp "$WORK/mm8/mm8.xclbin" "$OUT/vision_switches_base.xclbin"
ls -l "$OUT"
