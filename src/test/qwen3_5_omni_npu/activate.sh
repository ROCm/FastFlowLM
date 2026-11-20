#!/usr/bin/bash

# set up model loading path
export FLM_MODEL_PATH="/scratch/$USER"

# The dev kit in this repository; FLM_DEVKIT_DIR overrides it.
FLM_DEVKIT_DIR="${FLM_DEVKIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/flm-dev-kit}"

# copy lib
cp $FLM_DEVKIT_DIR/FLM_DLL/build/lib/libqwen3_5_omni_npu.so ../../lib

# copy xclbins
# cp $FLM_DEVKIT_DIR/FLM_Xclbin/Qwen3_5/qwen3_5_decoding/build/QWEN3_5_OMNI/xclbins/layer.xclbin ../../xclbins/Qwen3.5-OMNI-NPU2/
# cp /scratch/$USER/FastFlowLM/FLM_Xclbin/Dequant_mix_w64x512_chunk_reorder/build/xclbins/dequant.xclbin ../../xclbins/Qwen3.6-35B-A3B-NPU2/
# cp /scratch/$USER/FastFlowLM/FLM_Xclbin/Qwen3_6/MM/build/xclbins/mm.xclbin ../../xclbins/Qwen3.6-35B-A3B-NPU2/
# cp /scratch/$USER/FastFlowLM/FLM_Xclbin/Qwen3_6/gated_attention_prefill/build/xclbins/attn.xclbin ../../xclbins/Qwen3.6-35B-A3B-NPU2/
# cp /scratch/$USER/FastFlowLM/FLM_Xclbin/Qwen3_6/gate_delta_net_prefill/build/xclbins/GateDeltaNet_prefill.xclbin ../../xclbins/Qwen3.6-35B-A3B-NPU2/
# cp $FLM_DEVKIT_DIR/FLM_Xclbin/Qwen3_5/lm_head_npu_bin/build/QWEN3_5_4B/xclbins/lm_head.xclbin ../../xclbins/Qwen3.5-OMNI-NPU2/

