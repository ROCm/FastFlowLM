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

# Kernel makefile
include makefiles/kernel.mk

# Bitstream makefile
include makefiles/bitstream.mk

# Host makefile
include makefiles/host.mk

.PHONY: run all kernel link bitstream host clean clean_all instructions
all: ${XCLBIN_TARGETS} ${INSTS_TARGETS} ${HOST_C_TARGET}

clean_all:
	-@rm -rf build 
	-@rm -rf log
	-@rm -rf *.exe
	-@rm -rf trace*
	-@rm -rf routes



kernel: ${KERNEL_OBJS}


instructions: ${INSTS_TARGETS}


link: ${IRON_BOTH_MLIR_TARGET}


bitstream: ${XCLBIN_TARGETS}


host: ${HOST_C_TARGET}


clean:
	-@rm -rf build/host
	-@rm -rf ${HOST_C_TARGET}


# The xclbin has to be a prerequisite too -- otherwise an edit under iron/ is
# silently ignored and the freshly built host runs against a stale bitstream.
test: ${XCLBIN_TARGETS} ${INSTS_TARGETS} $(HOST_C_TARGET)
	export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib && ./$(HOST_C_TARGET)

route: ${IRON_BOTH_MLIR_TARGET}
	mkdir -p routes
	aie-opt --aie-create-pathfinder-flows --aie-find-flows ${<} | aie-translate --aie-flows-to-json > ./routes/route.json
	python visualize.py -j ./routes/route.json

trace: ${HOST_C_TARGET}
	${powershell} ./$< 
	~/mlir-aie/programming_examples/utils/parse_trace.py --input trace.txt --mlir build/mlir/*.mlir --colshift 0 --output trace.json
	~/mlir-aie/programming_examples/utils/get_trace_summary.py --input trace.json