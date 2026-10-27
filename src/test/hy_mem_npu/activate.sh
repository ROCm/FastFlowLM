#!/usr/bin/bash
# source this from src/test/hy_mem_npu before make / make test.

# Weights live at $FLM_MODEL_PATH/models/Hy-Memory-30B-A3B-NPU2
export FLM_MODEL_PATH="/scratch/$USER"

FLM_IRON="/scratch/$USER/Projects/FastFlowLM_IRON"
FLM_DLL_BUILD="$FLM_IRON/FLM_DLL/build"
XCLBIN_SRC="$FLM_IRON/FLM_Xclbin/Hy_mem_30BA3B"
XCLBIN_DST="../../xclbins/Hy-Memory-30B-A3B-NPU2"

# copy src to dst only if the contents differ
copy_if_different() {
    local src="$1" dst="$2"
    if cmp -s "$src" "$dst"; then
        echo "unchanged: $dst"
    else
        cp "$src" "$dst" && echo "updated:   $dst"
    fi
}

# the engine and its public header (a stale copy makes experiments lie)
copy_if_different $FLM_DLL_BUILD/hy_mem_npu/lib/libhy_mem_npu.so ../../lib/xrt/libhy_mem_npu.so
mkdir -p ../../include/models/hy_mem/flm/aie2p
copy_if_different $FLM_IRON/FLM_DLL/include/models/hy_mem/hy_mem_npu.hpp ../../include/models/hy_mem/flm/aie2p/hy_mem_npu.hpp

# The two layer overlays were both built as layer.xclbin; the engine asks for them
# by these names. Their RTP addresses are constants in hy_mem_npu_sequence.hpp, so
# nothing but the xclbins is installed -- re-check those constants against the new
# xclbins/rtp_addresses.txt after rebuilding either decoding kernel.
mkdir -p $XCLBIN_DST
copy_if_different $XCLBIN_SRC/dense_decoding/build/HY_MEMORY_30B_A3B_DENSE/xclbins/layer.xclbin    $XCLBIN_DST/dense_layer.xclbin
copy_if_different $XCLBIN_SRC/moe_decoding/build/HY_MEMORY_30B_A3B/xclbins/layer.xclbin            $XCLBIN_DST/moe_layer.xclbin
copy_if_different $XCLBIN_SRC/lm_head_npu_bin/build/HY_MEMORY_30B_A3B/xclbins/lm_head.xclbin       $XCLBIN_DST/lm_head.xclbin
# the batched prefill's dequant_mm + attention, one xclbin (prefill_overlay/README.md)
copy_if_different $XCLBIN_SRC/prefill_overlay/build/xclbins/overlay.xclbin                          $XCLBIN_DST/prefill_overlay.xclbin
