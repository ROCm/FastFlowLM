#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
#
# The control-packet switches of the Inkling2 decode: the decode overlay (Inkling2/decode_layer) and the lm_head
# matrix-vector engine (Inkling2/lm_head) built as aie2p-launchlib packages, and from them the warm switches
# cold_dec, dec_then_lmh, lmh_then_dec in one file (format: ../../../FLM_DLL/detail/inkling2_npu/inkling2_launch.hpp).
#
#   build.sh <out dir>        out/: inkling2_decode.switches, decode.xclbin (the decode build the switches are for)
#
# As ../switches/build.sh: each app from a copy of its unit plus --aie-generate-txn, the toolchain on PATH.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
X=$(cd "$HERE/../.." && pwd)
LL=$(cd "$X/../aie2p-launchlib/launchlib" && pwd)
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
WORK=$HERE/build/work
FLAGS="--aie-generate-cdo --no-compile-host --no-xchesscc --no-xbridge"

# build_one APP UNIT IRON_SCRIPT "EXTRA_AIECC_FLAGS" [IRON_ARGS]
build_one() {
    local app=$1 src=$X/$2 script=$3 extra=${4:-} args=${5:-}
    local unit=$WORK/src_$app out=$WORK/$app
    echo "== $app ($2)"
    mkdir -p "$unit"
    (cd "$src" && tar cf - --exclude=./build --exclude='*.exe' --exclude='*.log' .) | (cd "$unit" && tar xf -)
    (cd "$unit" && make kernel)
    mkdir -p "$out"
    (cd "$unit/iron" && python3 "$script" $args) > "$out/design.mlir"
    cp "$unit"/build/*/kernel/*.o "$out/" 2>/dev/null || cp "$unit"/build/kernel/*.o "$out/"
    (cd "$out" && aiecc.py $FLAGS $extra --aie-generate-xclbin --xclbin-name="$app.xclbin" \
        --aie-generate-txn --txn-name="${app}_txn.mlir" design.mlir)
    for f in "$out"/design.mlir.prj/main_core_*.elf; do ln -sf "$(basename "$f")" "$out/design.mlir.prj/${f##*/main_}"; done
}

rm -rf "$WORK" && mkdir -p "$WORK"
build_one dec Inkling2/decode_layer decode_aie.py "--alloc-scheme=basic-sequential"
build_one lmh Inkling2/lm_head      lm_head_aie.py "" INKLING2

python3 "$LL/tools/make_package.py" -o "$WORK/pkg" dec="$WORK/dec" lmh="$WORK/lmh"

${CXX:-g++} -std=c++20 -O2 -Wall -I "$LL/include" -I "$X/../FLM_DLL/include" "$HERE/../switches/switch_files.cpp" -o "$WORK/switch_files"
"$WORK/switch_files" --package "$WORK/pkg" --out "$WORK" --apps dec,lmh
mv "$WORK/inkling2.switches" "$OUT/inkling2_decode.switches"
cp "$WORK/dec/dec.xclbin" "$OUT/decode.xclbin"
ls -l "$OUT"
