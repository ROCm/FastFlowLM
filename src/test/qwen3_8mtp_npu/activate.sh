#!/usr/bin/bash

# Model weights live at $FLM_MODEL_PATH/models/Qwen3.8-27B-NPU2.
# model_list.json's "model_path" is "models", so this must point at the PARENT
# of that directory, not at it.
export FLM_MODEL_PATH="/scratch/$USER"

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
copy_if_different /scratch/$USER/Projects/FastFlowLM_IRON/FLM_DLL/build/lib/libqwen3_8mtp_npu.so ../../lib/xrt

# Phase 1 is CPU-only -- no xclbins to stage. The NPU kernel binaries land here
# in phase 2 (dequant_mm / lm_head / layer / GateDeltaNet_prefill), at which
# point add the matching copy_if_different lines for
# ../../xclbins/Qwen3.8-27B-NPU2/.
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/MTP_decoding/build/QWEN3_8_27B/xclbins/MTP_layer.xclbin /scratch/michyu/FastFlowLM/src/xclbins/Qwen3.8-27B-NPU2/

