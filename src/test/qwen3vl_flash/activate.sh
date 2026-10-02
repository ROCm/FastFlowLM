#!/usr/bin/bash

# set up model loading path
export FLM_MODEL_PATH="/scratch/$USER"

IRON=/scratch/$USER/FastFlowLM_IRON
MODEL_XCLBINS=../../xclbins/Qwen3-VL-4B-Instruct-NPU2

# copy src to dst dir only if the contents differ
copy_if_different() {
    local src="$1" dst="$2/$(basename "$1")"
    if cmp -s "$src" "$dst"; then
        echo "unchanged: $dst"
    else
        cp "$src" "$dst" && echo "updated:   $dst"
    fi
}

# copy lib
copy_if_different $IRON/FLM_DLL/build/lib/libqwen3vl_flash.so ../../lib/xrt
# libqwen3vl_flash.so leaves the MHA symbols undefined and picks them up from
# libmha.so at load time, so a change to the attention sequence generator only
# takes effect if this one is refreshed too.
copy_if_different $IRON/FLM_DLL/build/lib/libmha.so ../../lib/xrt

# copy xclbins
# Prefill is now a single fused overlay: columns 0-5 are the 6-column dequant+mm
# array, columns 6-7 a single attention CU. Splitting them across mm.xclbin and
# attn.xclbin meant reconfiguring the array twice per layer, ~5 ms each layer.
# Both halves are geometry-locked to the engine (m=64 / 256-row mm rounds, GQA
# 1:4 attention) -- do not substitute another dequant_mm or attention build.
cp $IRON/FLM_Xclbin/Qwen3VL/fused_prefill/build/xclbins/fused_prefill.xclbin /tmp/fused_prefill.xclbin
copy_if_different /tmp/fused_prefill.xclbin $MODEL_XCLBINS
