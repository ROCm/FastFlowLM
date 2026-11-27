#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
#
# The control-packet switches of the Qwen3.5 prefill (see ../../../FLM_DLL/detail/qwen3_5vl_npu/qwen3_5vl_launch.hpp):
# the prompt matmul (dequant_mm_bfp16_rtm2_glu_512x512x512), conv1d (Qwen3_5/conv1d_prefill), gated delta net
# (Qwen3_5/gate_delta_net_prefill/hostDeltaNetPrefill) and gated attention (Qwen3_5/gated_attention_prefill) kernels
# built as aie2p-launchlib packages, and from them one warm switch per ordered pair (instruction stream + control
# packets), packed into one file, qwen3_5.switches.
#
#   build.sh <out dir>        out/: qwen3_5.switches, qwen3_5_switches_base.xclbin (the mm build's xclbin)
#
# Each app is built from a copy of its unit (the unit's `make kernel`, its iron/ design with the unit's IRON_ARGS,
# aiecc with the unit's bitstream.mk flags) plus --aie-generate-txn, so the units' own build trees are left alone; the
# toolchain is the one on PATH (the aie-build image, or the environment of FLM_Xclbin/README.md). The switches load the
# mm app cold (cold_mm), so the engine's base hardware context only has to be an xclbin with the same partition.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
X=$(cd "$HERE/../.." && pwd)
LL=$(cd "$X/../aie2p-launchlib/launchlib" && pwd)
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
WORK=$HERE/build/work
FLAGS="--aie-generate-cdo --no-compile-host --no-xchesscc --no-xbridge"

# build_one APP UNIT IRON_SCRIPT EXTRA_AIECC_FLAGS MAKE_VARS [IRON_ARGS...]
build_one() {
    local app=$1 src=$X/$2 script=$3 extra=${4:-} vars=${5:-}
    shift 5
    local unit=$WORK/src_$app out=$WORK/$app
    echo "== $app ($src) $vars"
    mkdir -p "$unit"
    (cd "$src" && tar cf - --exclude=./build --exclude='*.exe' --exclude='*.log' .) | (cd "$unit" && tar xf -)
    (cd "$unit" && make kernel $vars)
    mkdir -p "$out"
    python3 "$unit/iron/$script" "$@" > "$out/design.mlir"
    cp "$unit"/build/*/kernel/*.o "$out/" 2>/dev/null || cp "$unit"/build/kernel/*.o "$out/"
    (cd "$out" && aiecc.py $FLAGS $extra --aie-generate-xclbin --xclbin-name="$app.xclbin" \
        --aie-generate-txn --txn-name="${app}_txn.mlir" design.mlir)
    # aiecc names the ELFs main_core_C_R.elf; make_package.py wants core_C_R.elf
    for f in "$out"/design.mlir.prj/main_core_*.elf; do ln -sf "$(basename "$f")" "$out/design.mlir.prj/${f##*/main_}"; done
}

rm -rf "$WORK" && mkdir -p "$WORK"
build_one mm   dequant_mm_bfp16_rtm2_glu_512x512x512 dequant_mm_aie.py "--alloc-scheme=basic-sequential" ""
build_one conv Qwen3_5/conv1d_prefill conv_aie.py "" ""
# The delta net twice: gdn with the chunked core (GDN_CHUNK_MODE=1, kernel/gdn_chunked.h), gdn0 with the per-token
# recurrence (the standalone GateDeltaNet_prefill.xclbin's), for QWEN35_GDN_RECURRENT=1. Same buffers and DMA.
# IRON_ARGS of hostDeltaNetPrefill/Makefile: DEVICE TRACE_SIZE L DQ DK DV rows columns MT_L CT_L heads QK_group layout
GDN_ARGS="npu2 0 2048 128 128 128 4 8 128 16 32 2 1"
build_one gdn  Qwen3_5/gate_delta_net_prefill/hostDeltaNetPrefill GateDeltaNet_prefill_aie.py "" GDN_CHUNK_MODE=1 $GDN_ARGS
build_one gdn0 Qwen3_5/gate_delta_net_prefill/hostDeltaNetPrefill GateDeltaNet_prefill_aie.py "" GDN_CHUNK_MODE=0 $GDN_ARGS
build_one attn Qwen3_5/gated_attention_prefill attn_aie.py "" ""

python3 "$LL/tools/make_package.py" -o "$WORK/pkg" mm="$WORK/mm" conv="$WORK/conv" gdn="$WORK/gdn" gdn0="$WORK/gdn0" \
    attn="$WORK/attn"

# The switch generator needs no NPU and no XRT. Only the switches the engine takes (qwen3_5vl_launch.hpp names).
${CXX:-g++} -std=c++20 -O2 -Wall -I "$LL/include" -I "$X/../FLM_DLL/include" "$HERE/switch_files.cpp" -o "$WORK/switch_files"
"$WORK/switch_files" --package "$WORK/pkg" --out "$OUT" --apps mm,conv,gdn,gdn0,attn \
    --pairs mm:conv,conv:gdn,gdn:mm,mm:attn,attn:mm,conv:gdn0,gdn0:mm

# The base xclbin the switches were built against (the engine's mm context is this unit's xclbin).
cp "$WORK/mm/mm.xclbin" "$OUT/qwen3_5_switches_base.xclbin"
ls -l "$OUT"
