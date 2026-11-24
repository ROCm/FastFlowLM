#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
# Modified by Alfred

include makefiles/common.mk
MODEL_TYPE ?= GEMMA4_E4B
L_TOKENS ?= 8
# Which top-level IRON design to build: iron/$(DESIGN)_aie.py.
#   fused8 (default)  the shipped 8-row verify layer (verify.xclbin of nemotron3.5-bolt-dflash:e4b)
#   layer             the decode layer copied from Gemma4/decoding; with L_TOKENS=1 it must stay identical to it (regression build)
#   projonly, projstage, rmsglu, ple8, attn8, attnx, attn8attn, attn8rope   stage test designs (README.md)
DESIGN ?= fused8
# fused8 options: FUSED8_ATTN=x (attention reads K/V once per verify, shipped) or 8 (K/V replayed per token, A/B only);
# FUSED8_STAGE=full (shipped), nople or attn (debug cut-offs that end the layer early: never loop them, they do not drain)
FUSED8_ATTN ?= x
FUSED8_STAGE ?= full
export FUSED8_ATTN FUSED8_STAGE
# builds for different L_TOKENS / designs must not overwrite each other
ifeq ($(DESIGN),fused8)
BUILD_ROOT ?= build/$(MODEL_TYPE)_fused8$(FUSED8_ATTN)_$(FUSED8_STAGE)_L$(L_TOKENS)
else
BUILD_ROOT ?= build/$(MODEL_TYPE)_L$(L_TOKENS)
endif
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

KERNEL_ARGS := -DMODEL_TYPE=${MODEL_TYPE} -DL_TOKENS=${L_TOKENS}
IRON_ARGS := ${MODEL_TYPE}
export L_TOKENS

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
	export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib && ./$(HOST_C_TARGET) -m ${MODEL_TYPE}

route: ${IRON_BOTH_MLIR_TARGET}
	mkdir -p routes
	aie-opt --aie-create-pathfinder-flows --aie-find-flows ${<} | aie-translate --aie-flows-to-json > ./routes/route.json
	python visualize.py -j ./routes/route.json
