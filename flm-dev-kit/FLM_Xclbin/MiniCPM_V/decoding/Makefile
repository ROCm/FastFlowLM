#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
# Modified by Alfred

include makefiles/common.mk
MODEL_TYPE ?= QWEN3_5_2B
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# MODEL_TYPE=<model>_SL builds the same design with STATELESS_X (kernels/rms_residual.cc, kernels/gate.cc): every run takes x from
# DDR and returns it to DDR, so the xclbin stays correct when another process's command makes the firmware
# reload the design between two runs, and no core counts the layers of a period. The host sequence must then send x with, and receive x from,
# every run, and write the gate core's mode RTP.
MODEL_BASE := $(patsubst %_SL,%,$(MODEL_TYPE))
KERNEL_ARGS := -DMODEL_TYPE=${MODEL_BASE}
ifneq ($(MODEL_BASE),$(MODEL_TYPE))
KERNEL_ARGS += -DSTATELESS_X
# read by iron/gate.py (the gate core gets a mode RTP)
export STATELESS_X := 1
endif
# MiniCPM-V 1B layer_lm (the 0.8B shape): which kernels select round-to-nearest-even for their calls (bit i = KERNEL_RNE_SCOPE(i)).
# Measured against a fp32 reference of the same Q4 weights (decode-step logits, KL, 500 teacher-forced steps): glu(5) gate(6) rope(7)
# rms_residual(8) and gate_delta_net_decode(10) together get more accurate with it (0.0156 -> 0.0082); attn_kv(9) alone and in the set is
# not better; conv(0), attn_qk(1) and delta_net_norm(2) get worse (0.035); proj_main(3) does not change; ab_proj(4) breaks (its exp /
# softplus / sigmoid lookup tables index with the default floor mode). Those keep the default mode, as does every other MODEL_TYPE of this design.
KERNEL_RNE_MASK ?= $(if $(findstring 08B,$(MODEL_BASE)),1504,0)
KERNEL_ARGS += -DKERNEL_RNE_MASK=${KERNEL_RNE_MASK}
# Every kernel of the 0.8B designs sets its rounding mode itself (floor unless its bit is set in the mask above): a core keeps the mode a previous
# design left, so a kernel that relies on the default runs in another process's mode when that process ran on the core before.
ifneq ($(findstring 08B,$(MODEL_BASE)),)
KERNEL_ARGS += -DFLM_KERNEL_ENTRY_FLOOR
endif
IRON_ARGS := ${MODEL_BASE}

# Kernel makefile
include makefiles/kernel.mk

# Bitstream makefile
include makefiles/bitstream.mk

# Host makefile
include makefiles/host.mk

.PHONY: run all kernel link bitstream host clean clean_all instructions
all: ${XCLBIN_TARGET} ${INSTS_TARGET} ${HOST_C_TARGET}
clean_all:
	-@rm -rf build 
	-@rm -rf log
	-@rm -rf *.exe
	-@rm -rf trace*
	-@rm -rf routes



kernel: ${KERNEL_OBJS}


instructions: ${INSTS_TARGETS}


link: ${MLIR_TARGET} 


bitstream: ${XCLBIN_TARGETS}


host: ${HOST_C_TARGET}


clean:
	-@rm -rf build/host
	-@rm -rf ${HOST_C_TARGET}



test: $(HOST_C_TARGET)
	./$(HOST_C_TARGET) --xclbin build/${MODEL_TYPE}/xclbins/layer.xclbin $(ARGS)

route: ${IRON_BOTH_MLIR_TARGET}
	mkdir -p routes
	aie-opt --aie-create-pathfinder-flows --aie-find-flows ${<} | aie-translate --aie-flows-to-json > ./routes/route.json
	python visualize.py -j ./routes/route.json

FASTFLOWLM_SRC ?= /path/to/FastFlowLM/src
MODEL_DIR ?= /path/to/models/Qwen3.5-2B-NPU2
build/tokenize_ids: tools/tokenize_ids.cpp
	mkdir -p build
	g++-13 -std=c++20 -O2 -I$(FASTFLOWLM_SRC)/include -I$(FASTFLOWLM_SRC)/../third_party/tokenizers-cpp/include \
		$< $(FASTFLOWLM_SRC)/common/tokenizer/tokenizer.cpp -o $@ \
		-L$(FASTFLOWLM_SRC)/build/tokenizers-cpp -L$(FASTFLOWLM_SRC)/build/tokenizers-cpp/sentencepiece/src \
		-ltokenizers_cpp -ltokenizers_c -lsentencepiece -ldl -lpthread

# Token ids of the story text repeated to ~12K tokens (input for --prompt-ids)
build/story_ids.txt: build/tokenize_ids
	for i in $$(seq 13); do cat $(FASTFLOWLM_SRC)/test/qwen3_5vl_npu/bench_text.txt; done > build/story_x13.txt
	./build/tokenize_ids $(MODEL_DIR) build/story_x13.txt $@
