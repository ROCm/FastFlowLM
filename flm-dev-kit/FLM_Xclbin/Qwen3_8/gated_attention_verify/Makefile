#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred

include makefiles/common.mk
MODEL_TYPE ?= QWEN3_8_27B
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Query rows per forward pass. L=8 verifies a 7-token MTP draft plus the bonus
# token in one target pass. It is a compile-time constant everywhere -- buffer
# shapes, DMA sizes and lock counts are all baked at build time -- so this one
# knob feeds all three languages: IRON (env), the AIE kernels (-D), the host
# (-D). They cannot be set independently.
L_TOKENS ?= 8
export L_TOKENS

# Debug lever for the a/b stage. The a/b kernel finishes with exp(a * softplus)
# and a tanh-based sigmoid, both approximated in bf16 (a 64-entry linear-approx
# LUT for softplus), which is a much coarser floor than the q8_0 mvm feeding
# them. Build with AB_SKIP_ACTIVATION=1 to make the kernel emit the raw mvm and
# see which of the two the error is in. The host half of this lever (--ab-raw,
# and the a/b report it scored) went with the stage-by-stage bench; bring it back
# from git history if the a/b stage ever needs re-bisecting.
AB_SKIP_ACTIVATION ?= 0

KERNEL_ARGS := -DMODEL_TYPE=${MODEL_TYPE} -DL_TOKENS=${L_TOKENS} -DAB_SKIP_ACTIVATION=${AB_SKIP_ACTIVATION}
IRON_ARGS := ${MODEL_TYPE}

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
