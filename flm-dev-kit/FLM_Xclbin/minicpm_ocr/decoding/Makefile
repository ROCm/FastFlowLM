#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
# Modified by Alfred

include makefiles/common.mk
MODEL_TYPE ?= QWEN3_5_08B
ifeq ($(MODEL_TYPE),QWEN3_5_08B_KVB)
override MODEL_TYPE := QWEN3_5_08B
override KV_BFP16 := 1
endif
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

KERNEL_ARGS := -DMODEL_TYPE=${MODEL_TYPE}
IRON_ARGS := ${MODEL_TYPE}

# KV_BFP16=1: decode attention reads finished K/V chunks as host-packed bfp16 (README "bfp16 K and V");
# builds into build/<MODEL_TYPE>_kvb, the default build is unchanged.
KV_BFP16 ?= 0
ifeq ($(KV_BFP16),1)
BUILD_NAME := ${MODEL_TYPE}_kvb
KERNEL_ARGS += -DKV_BFP16
IRON_ARGS += kvb
RTP_QK_ADDR ?= 25600
RTP_KV_ADDR ?= 53248
else
BUILD_NAME := ${MODEL_TYPE}
RTP_QK_ADDR ?= 53248
RTP_KV_ADDR ?= 53536
endif

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
	./$(HOST_C_TARGET) --xclbin build/${BUILD_NAME}/xclbins/layer.xclbin $(ARGS)

route: ${IRON_BOTH_MLIR_TARGET}
	mkdir -p routes
	aie-opt --aie-create-pathfinder-flows --aie-find-flows ${<} | aie-translate --aie-flows-to-json > ./routes/route.json
	python visualize.py -j ./routes/route.json

FASTFLOWLM_SRC ?= /path/to/FastFlowLM/src
MODEL_DIR ?= /path/to/models/MiniCPM-OCR-1.0-NPU2
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
