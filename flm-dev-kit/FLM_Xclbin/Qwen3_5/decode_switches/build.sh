#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
#
# The control-packet switches of the Qwen3.5-4B decode (see ../../../FLM_DLL/detail/qwen3_5vl_npu/qwen3_5vl_npu.cpp):
# the decode layer (Qwen3_5/qwen3_5_decoding, MODEL_TYPE QWEN3_5_4B) and the q4_k lm_head (Qwen3_5/lm_head_npu_bin,
# MODEL_TYPE QWEN3_5_4B_Q4K) built as aie2p-launchlib packages, and from them cold_layer, layer_then_lm and
# lm_then_layer, packed into qwen3_5_decode.switches (same format as ../prefill_switches).
#
#   build.sh <out dir>        out/: qwen3_5_decode.switches, qwen3_5_decode_switches_base.xclbin (the layer build's xclbin)
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
export ENABLE_CHESSCC=0

# build_one APP UNIT IRON_SCRIPT MODEL_TYPE
build_one() {
    local app=$1 src=$X/$2 script=$3 model=$4
    local unit=$WORK/src_$app out=$WORK/$app
    echo "== $app ($src, $model)"
    mkdir -p "$unit"
    (cd "$src" && tar cf - --exclude=./build --exclude='*.exe' --exclude='*.log' .) | (cd "$unit" && tar xf -)
    (cd "$unit" && make -j"$(nproc)" kernel MODEL_TYPE="$model")
    local args
    args=$(cd "$unit" && make -pn MODEL_TYPE="$model" kernel 2>/dev/null | sed -n 's/^IRON_ARGS := //p')
    [ -n "$args" ] || { echo "no IRON_ARGS for $app" >&2; exit 1; }
    mkdir -p "$out"
    # shellcheck disable=SC2086
    (cd "$unit" && python3 "iron/$script" $args) > "$out/design.mlir"
    find "$unit/build" -name '*.o' -exec cp {} "$out/" \;
    (cd "$out" && aiecc.py $FLAGS --aie-generate-xclbin --xclbin-name="$app.xclbin" \
        --aie-generate-txn --txn-name="${app}_txn.mlir" design.mlir)
    # aiecc names the ELFs main_core_C_R.elf; make_package.py wants core_C_R.elf
    for f in "$out"/design.mlir.prj/main_core_*.elf; do ln -sf "$(basename "$f")" "$out/design.mlir.prj/${f##*/main_}"; done
}

rm -rf "$WORK" && mkdir -p "$WORK"
build_one layer Qwen3_5/qwen3_5_decoding layer_aie.py   QWEN3_5_4B
build_one lm    Qwen3_5/lm_head_npu_bin  lm_head_aie.py QWEN3_5_4B_Q4K

# The engine writes the attention context length to fixed RTP addresses (qwen3_5_decoding/makefiles/bitstream.mk):
# lm_then_layer restores this build's layer, so its RTPs must be where the engine writes them.
grep -hoE 'address = [0-9]+ : i32, mem_bank = [0-9]+ : i32, sym_name = "RTP_L_attn_(qk|kv)_core_[0-9_]+"' \
    "$WORK"/layer/design.mlir.prj/input_with_addresses.mlir |
    sed -E 's/address = ([0-9]+) .*sym_name = "(RTP_L_attn_(qk|kv))_core_([0-9_]+)"/\1 \2 \4/' |
    awk '{ want = ($2 == "RTP_L_attn_qk") ? 53248 : 53536; n++ }
         $1 != want { printf "error: %s_core_%s is at %s, FLM_DLL expects %s\n", $2, $3, $1, want; bad = 1 }
         END { if (n == 0) { print "error: no attention RTP buffers found"; bad = 1 } exit bad }'

python3 "$LL/tools/make_package.py" -o "$WORK/pkg" layer="$WORK/layer" lm="$WORK/lm"

# The switch generator (../prefill_switches/switch_files.cpp) needs no NPU and no XRT.
${CXX:-g++} -std=c++20 -O2 -Wall -I "$LL/include" -I "$X/../FLM_DLL/include" "$HERE/../prefill_switches/switch_files.cpp" \
    -o "$WORK/switch_files"
"$WORK/switch_files" --package "$WORK/pkg" --out "$OUT" --apps layer,lm --file qwen3_5_decode.switches

# The base xclbin the switches were built against (the engine's layer context is this unit's xclbin).
cp "$WORK/layer/layer.xclbin" "$OUT/qwen3_5_decode_switches_base.xclbin"
ls -l "$OUT"
