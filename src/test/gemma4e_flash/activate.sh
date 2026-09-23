#!/usr/bin/bash

# set up model loading path
export FLM_MODEL_PATH="/scratch/$USER"

IRON=/scratch/$USER/FastFlowLM_IRON

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
copy_if_different $IRON/FLM_DLL/build/lib/libgemma4e_npu.so ../../lib/xrt
# libqwen3vl_npu.so leaves the MHA symbols undefined and picks them up from
# libmha.so at load time, so a change to the attention sequence generator only
# takes effect if this one is refreshed too.
copy_if_different $IRON/FLM_DLL/build/lib/libmha.so ../../lib/xrt

# xclbins: the classic prefill path uses the committed mm.xclbin / attn.xclbin /
# /scratch/alfxu/FastFlowLM_IRON/FLM_Xclbin/Gemma4/fused_prefill/build/xclbins/fused_prefill.xclbin
copy_if_different $IRON/FLM_Xclbin/Gemma4/fused_prefill/build/xclbins/fused_prefill.xclbin ../../xclbins/Gemma4-E2B-IT-NPU2/
copy_if_different $IRON/FLM_Xclbin/Gemma4/fused_prefill/build/xclbins/fused_prefill.xclbin ../../xclbins/Gemma4-E4B-IT-NPU2/
