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
copy_if_different $IRON/FLM_DLL/build/lib/libqwen3vl_npu.so ../../lib/xrt
# libqwen3vl_npu.so leaves the MHA symbols undefined and picks them up from
# libmha.so at load time, so a change to the attention sequence generator only
# takes effect if this one is refreshed too.
copy_if_different $IRON/FLM_DLL/build/lib/libmha.so ../../lib/xrt

# xclbins: the classic prefill path uses the committed mm.xclbin / attn.xclbin /
# dequant.xclbin in ../../xclbins/Qwen3-VL-4B-Instruct-NPU2. Those are shipped
# artifacts -- none of the FLM_Xclbin trees on this host reproduces them
# byte-for-byte -- so nothing is staged here. See ../qwen3vl_flash/activate.sh
# for the fused overlay the flash engine uses instead.
