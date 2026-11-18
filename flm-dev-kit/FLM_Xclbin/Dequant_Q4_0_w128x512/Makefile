# 
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred

include makefiles/common.mk

DEVICE ?= npu2
TRACE_SIZE ?= 0
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
# do you want chesscc? compiles slower
ENABLE_CHESSCC ?= 0

# L ?= 16
IRON_ARGS := ${DEVICE}
# IRON_ARGS += ${L}
IRON_ARGS += ${TRACE_SIZE}

# Kernel makefile
include makefiles/kernel.mk

# Bitstream makefile
include makefiles/bitstream.mk
include makefiles/mlir_bitstream.mk

# Host makefile
include makefiles/host.mk

.PHONY: run all kernel link bitstream host clean instructions route
all: ${XCLBIN_TARGET} ${INSTS_TARGET} ${HOST_C_TARGET}

clean:
	-@rm -rf build 
	-@rm -rf log
	-@rm -rf *.exe
	-@rm -rf trace*
	-@rm -rf route.json
	-@rm -rf generated.txt
	-@rm -rf route*.txt

test:
	echo "test"
	echo ${AIEOPT_DIR}


kernel: ${KERNEL_OBJS}


instructions: ${INSTS_TARGETS}


link: ${MLIR_TARGET} 


bitstream: ${XCLBIN_TARGETS}


host: ${HOST_C_TARGET}


clean_host:
	-@rm -rf build/host


run: ${HOST_C_TARGET} ${XCLBIN_TARGETS}
	./${HOST_C_TARGET}

route: ${IRON_BOTH_MLIR_TARGET}
	aie-opt --aie-create-pathfinder-flows --aie-find-flows ${<} | aie-translate --aie-flows-to-json > route.json
	python visualize.py -j route.json
