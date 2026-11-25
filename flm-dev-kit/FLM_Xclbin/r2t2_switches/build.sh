#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
#
# The control-packet switches of r2t2 (see ../../FLM_DLL/detail/qwen3_asr_npu/prefill/qwen3_asr_launch.hpp): the
# prompt prefill's mm kernel and the prefill attention kernel, built as aie2p-launchlib packages, and from them one
# warm switch per ordered pair (instruction stream + control packets), packed into one file, r2t2.switches.
#
#   build.sh <out dir>        out/: r2t2.switches, r2t2_switches_base.xclbin (the mm build's xclbin)
#
# Every app is built as its own unit builds it (the unit's `make kernel`, its IRON design, aiecc with the unit's
# bitstream.mk flags) plus --aie-generate-txn; the toolchain is the one on PATH (the aie-build image, or the
# environment of FLM_Xclbin/README.md). The mm app is dequant_mm_mix_bfp16 (prefill.xclbin): the switches load it
# cold (cold_mm), so the engine's base hardware context only has to be that unit's xclbin, which has the same
# partition.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
X=$(cd "$HERE/.." && pwd)
LL=$(cd "$X/../aie2p-launchlib/launchlib" && pwd)
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
WORK=$HERE/build/work
FLAGS="--aie-generate-cdo --no-compile-host --no-xchesscc --no-xbridge"

# build_one APP UNIT IRON_SCRIPT EXTRA_AIECC_FLAGS
build_one() {
    local app=$1 unit=$X/$2 script=$3 extra=${4:-}
    local out=$WORK/$app
    echo "== $app ($2)"
    (cd "$unit" && rm -rf build && make kernel)
    mkdir -p "$out"
    python3 "$unit/iron/$script" > "$out/design.mlir"
    cp "$unit"/build/*/kernel/*.o "$out/" 2>/dev/null || cp "$unit"/build/kernel/*.o "$out/"
    (cd "$out" && aiecc.py $FLAGS $extra --aie-generate-xclbin --xclbin-name="$app.xclbin" \
        --aie-generate-txn --txn-name="${app}_txn.mlir" design.mlir)
    # aiecc names the ELFs main_core_C_R.elf; make_package.py wants core_C_R.elf
    for f in "$out"/design.mlir.prj/main_core_*.elf; do ln -sf "$(basename "$f")" "$out/design.mlir.prj/${f##*/main_}"; done
}

rm -rf "$WORK" && mkdir -p "$WORK"
build_one mm   dequant_mm_mix_bfp16 dequant_mm_aie.py
build_one attn generic_prefill_attention/attention_DH_128_GQA_1_2 attn_aie.py

python3 "$LL/tools/make_package.py" -o "$WORK/pkg" mm="$WORK/mm" attn="$WORK/attn"

# The switch generator needs no NPU and no XRT.
${CXX:-g++} -std=c++20 -O2 -Wall -I "$LL/include" -I "$X/../FLM_DLL/include" "$HERE/switch_files.cpp" -o "$WORK/switch_files"
"$WORK/switch_files" --package "$WORK/pkg" --out "$OUT" --apps mm,attn

# The base xclbin the switches were built against (the engine's mm context is this unit's xclbin).
cp "$WORK/mm/mm.xclbin" "$OUT/r2t2_switches_base.xclbin"
ls -l "$OUT"
