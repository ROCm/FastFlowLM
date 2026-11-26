#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
#
# The control-packet switches of the Qwen3.5-4B vision encoder (see ../../../FLM_DLL/detail/qwen3_5vl_npu/
# qwen3_5vl_image.cpp): the vision matmul (MM/host7_mt_variant_bias, MODEL_TYPE QWEN3_5_4B-VISION-MM) and the vision
# attention (VisionAttention/hostAttention_whoel_array, MODEL_TYPE QWEN3_5_4B-VISION-ATTENTION) kernels built as
# aie2p-launchlib packages, and from them one warm switch per ordered pair, packed into qwen3_5_vision.switches
# (same format as ../prefill_switches).
#
#   build.sh <out dir>        out/: qwen3_5_vision.switches, qwen3_5_vision_switches_base.xclbin (the mm build's xclbin)
#
# Each app is built from a copy of its unit (`make kernel MODEL_TYPE=...`, its iron/ design with the unit's IRON_ARGS
# for that MODEL_TYPE, aiecc with the unit's bitstream.mk flags) plus --aie-generate-txn, so the units' own build trees
# are left alone; the toolchain is the one on PATH (the aie-build image, or the environment of FLM_Xclbin/README.md).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
X=$(cd "$HERE/../.." && pwd)
LL=$(cd "$X/../aie2p-launchlib/launchlib" && pwd)
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
WORK=$HERE/build/work
FLAGS="--aie-generate-cdo --no-compile-host --no-xchesscc --no-xbridge"

# build_one APP COPY_ROOT UNIT_SUBDIR IRON_SCRIPT MODEL_TYPE
#   COPY_ROOT is copied whole (VisionAttention units include ../makefiles and ../kernel)
build_one() {
    local app=$1 root=$X/$2 sub=$3 script=$4 model=$5
    local copy=$WORK/src_$app out=$WORK/$app
    echo "== $app ($root/$sub, $model)"
    mkdir -p "$copy"
    (cd "$root" && tar cf - --exclude='./build' --exclude='./*/build' --exclude='*.exe' --exclude='*.log' .) |
        (cd "$copy" && tar xf -)
    local unit=$copy/$sub
    (cd "$unit" && make -j"$(nproc)" kernel MODEL_TYPE="$model")
    local args
    args=$(cd "$unit" && make -pn MODEL_TYPE="$model" kernel 2>/dev/null | sed -n 's/^IRON_ARGS := //p')
    [ -n "$args" ] || { echo "no IRON_ARGS for $app" >&2; exit 1; }
    mkdir -p "$out"
    # shellcheck disable=SC2086
    (cd "$unit" && python3 "iron/$script" $args) > "$out/design.mlir"
    find "$copy" -path '*build/kernel/*' -name '*.o' -exec cp {} "$out/" \;
    (cd "$out" && aiecc.py $FLAGS --aie-generate-xclbin --xclbin-name="$app.xclbin" \
        --aie-generate-txn --txn-name="${app}_txn.mlir" design.mlir)
    # aiecc names the ELFs main_core_C_R.elf; make_package.py wants core_C_R.elf
    for f in "$out"/design.mlir.prj/main_core_*.elf; do ln -sf "$(basename "$f")" "$out/design.mlir.prj/${f##*/main_}"; done
}

rm -rf "$WORK" && mkdir -p "$WORK"
build_one mm   MM/host7_mt_variant_bias       .                        mm_bf16_accf32_aie.py   QWEN3_5_4B-VISION-MM
build_one attn VisionAttention                hostAttention_whoel_array attention_bf16_aie.py QWEN3_5_4B-VISION-ATTENTION

python3 "$LL/tools/make_package.py" -o "$WORK/pkg" mm="$WORK/mm" attn="$WORK/attn"

# The switch generator (../prefill_switches/switch_files.cpp) needs no NPU and no XRT.
${CXX:-g++} -std=c++20 -O2 -Wall -I "$LL/include" -I "$X/../FLM_DLL/include" "$HERE/../prefill_switches/switch_files.cpp" \
    -o "$WORK/switch_files"
"$WORK/switch_files" --package "$WORK/pkg" --out "$OUT" --apps mm,attn --file qwen3_5_vision.switches

# The base xclbin the switches were built against (the engine's vision mm context is this unit's xclbin).
cp "$WORK/mm/mm.xclbin" "$OUT/qwen3_5_vision_switches_base.xclbin"
ls -l "$OUT"
