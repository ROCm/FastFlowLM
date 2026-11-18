#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred

include makefiles/common.mk
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

KERNEL_ARGS := -DMODEL_TYPE=${MODEL_TYPE}
IRON_ARGS := ${MODEL_TYPE}

ITERATION ?= 1

# DMA_ONLY=1 compiles the kernel math out (-DDMA_ONLY). Locks, buffer swaps and the
# input copy stay, so the measured time is the data-movement floor.
DMA_ONLY ?= 0
ifeq ($(DMA_ONLY),1)
KERNEL_ARGS += -DDMA_ONLY
endif

# Kernel makefile
include makefiles/kernel.mk

# Rebuild the kernel when DMA_ONLY or KERNEL_ARGS change: the stamp is only rewritten when its text
# changes, and the objects depend on it. kernel.mk only tracks *.h, so add *.hpp too.
KERNEL_FLAG_STAMP := ${HOME_DIR}/build/kernel_flags.stamp
KERNEL_FLAGS_TEXT := DMA_ONLY=${DMA_ONLY} KERNEL_ARGS=${KERNEL_ARGS} ENABLE_CHESSCC=${ENABLE_CHESSCC}
$(shell mkdir -p ${HOME_DIR}/build; echo "${KERNEL_FLAGS_TEXT}" | cmp -s - ${KERNEL_FLAG_STAMP} || echo "${KERNEL_FLAGS_TEXT}" > ${KERNEL_FLAG_STAMP})
${KERNEL_OBJS}: ${KERNEL_FLAG_STAMP} $(wildcard ${HOME_DIR}/kernel/*.hpp)

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
	export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib && ./$(HOST_C_TARGET)

route: ${IRON_BOTH_MLIR_TARGET}
	mkdir -p routes
	aie-opt --aie-create-pathfinder-flows --aie-find-flows ${<} | aie-translate --aie-flows-to-json > ./routes/route.json
	python visualize.py -j ./routes/route.json

trace: ${HOST_C_TARGET}
	${powershell} ./$< 
	~/mlir-aie/programming_examples/utils/parse_trace.py --input trace.txt --mlir build/mlir/*.mlir --colshift 0 --output trace.json
	~/mlir-aie/programming_examples/utils/get_trace_summary.py --input trace.json