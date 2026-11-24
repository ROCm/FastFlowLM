#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred
# Portions of this file consist of AI-generated content.

include makefiles/common.mk
MODEL_TYPE ?= GEMMA3_4B
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

KERNEL_ARGS := -DMODEL_TYPE=${MODEL_TYPE}
IRON_ARGS := ${MODEL_TYPE}

STUB ?= 0
PROBE ?=
BUILD_NAME := $(MODEL_TYPE)$(if $(filter 1,$(STUB)),_STUB)$(if $(PROBE),_P$(PROBE))
ifeq ($(STUB),1)
KERNEL_ARGS += -DFLM_STUB
endif
# timing-only probes: NO_X, NO_W, NO_Y (README "Weight streaming probes"), NO_QK_MATH, NO_KV_MATH (README "Attention
# math")
ifneq ($(PROBE),)
KERNEL_ARGS += -DPROBE_$(PROBE)
endif

# Kernel makefile
include makefiles/kernel.mk

# Bitstream makefile
include makefiles/bitstream.mk

# Host makefile
include makefiles/host.mk

# Standalone lm_head test
include makefiles/test.mk

.PHONY: run all kernel link bitstream txn host test_lm_head test_layer_time clean clean_all instructions
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


txn: ${TXN_TARGETS}


host: ${HOST_C_TARGET}


clean:
	-@rm -rf build/host
	-@rm -rf ${HOST_C_TARGET}



test: $(HOST_C_TARGET)
	make -C ../../libraries/detail/gemma_npu/
	export LD_LIBRARY_PATH=../../libraries/build/lib && ./$(HOST_C_TARGET) --model ../../../models -t $(MODEL_TYPE)

route: ${IRON_BOTH_MLIR_TARGET}
	mkdir -p routes
	aie-opt --aie-create-pathfinder-flows --aie-find-flows ${<} | aie-translate --aie-flows-to-json > ./routes/route.json
	python visualize.py -j ./routes/route.json
