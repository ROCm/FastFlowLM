#!/usr/bin/bash

# Model weights live at $FLM_MODEL_PATH/models/Qwen3.8-27B-NPU2.
# model_list.json's "model_path" is "models", so this must point at the PARENT
# of that directory, not at it.
export FLM_MODEL_PATH="/scratch/$USER"

# copy src to dst only if the contents differ; dst may be a directory or a
# destination filename (used to rename layer.xclbin -> attention_layer.xclbin)
copy_if_different() {
    local src="$1" dst="$2"
    [ -d "$dst" ] && dst="$dst/$(basename "$src")"
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
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/gate_delta_net_verify/build/QWEN3_8_27B/xclbins/deltanet_layer.xclbin /scratch/michyu/FastFlowLM/src/xclbins/Qwen3.8-27B-NPU2/
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/gated_attention_verify/build/QWEN3_8_27B/xclbins/layer.xclbin /scratch/michyu/FastFlowLM/src/xclbins/Qwen3.8-27B-NPU2/attention_layer.xclbin
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/gated_attention_prefill/build/xclbins/attn.xclbin /scratch/michyu/FastFlowLM/src/xclbins/Qwen3.8-27B-NPU2/attention_prefill.xclbin
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/lm_head_npu_bin/build/QWEN3_8_27B/xclbins/lm_head.xclbin /scratch/michyu/FastFlowLM/src/xclbins/Qwen3.8-27B-NPU2/
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/lm_head_8/build/QWEN3_8_27B/xclbins/lm_head.xclbin /scratch/michyu/FastFlowLM/src/xclbins/Qwen3.8-27B-NPU2/lm_head_8.xclbin
# The fused Q4_K dequant + GEMM the MTP head's step-0 projections run on. Its
# design is not named layer.xclbin, so it stages under its own name unrenamed.
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/dequant_mm/build/xclbins/dequant_mm.xclbin /scratch/michyu/FastFlowLM/src/xclbins/Qwen3.8-27B-NPU2/

# N10's two: the depthwise conv1d (+ SiLU + the q/k l2norms) and the gated
# delta-rule fold, which together are the 48 linear layers' whole recurrence.
# Both keep their own names; nothing else stages a conv.xclbin.
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/conv1d_prefill/build/xclbins/conv.xclbin /scratch/michyu/FastFlowLM/src/xclbins/Qwen3.8-27B-NPU2/
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/gate_delta_net_prefill/build/xclbins/GateDeltaNet_prefill.xclbin /scratch/michyu/FastFlowLM/src/xclbins/Qwen3.8-27B-NPU2/

# The vision tower's attention (N15). Its source is also attn.xclbin, same as
# gated_attention_prefill above, so it MUST be renamed on the way in or the two
# designs land on the same file and the tower silently drives the language
# model's causal core. The engine looks for it under this name and falls back
# to the host tower if it is absent.
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/vision_attn/build/xclbins/attn.xclbin /scratch/michyu/FastFlowLM/src/xclbins/Qwen3.8-27B-NPU2/vision_attn.xclbin

# The vision tower's matmuls (N16), the other ~7 s of a 1200-patch image. Two
# designs: A (m=64,k=384,n=48) serves every shape, B (m=32,k=384,n=128) exists
# for merger.fc2 alone, whose N of 5120 is not a multiple of A's tile_N of 384.
# BOTH sources are build/<design>/xclbins/mm.xclbin, so both must be renamed or
# the second copy overwrites the first and the tower drives the wrong tile shape.
# The engine wants both or it leaves the matmuls on the host; it never runs one.
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/vision_mm/build/VISION_MM_A/xclbins/mm.xclbin /scratch/michyu/FastFlowLM/src/xclbins/Qwen3.8-27B-NPU2/vision_mm_a.xclbin
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/vision_mm/build/VISION_MM_B/xclbins/mm.xclbin /scratch/michyu/FastFlowLM/src/xclbins/Qwen3.8-27B-NPU2/vision_mm_b.xclbin
