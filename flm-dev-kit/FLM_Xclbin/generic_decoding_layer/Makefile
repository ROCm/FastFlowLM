#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred

include makefiles/common.mk
MODEL_TYPE ?= HUNYUAN_DENSE_1_8B

# This directory builds with Peano by default; ENABLE_CHESSCC=1 selects chess.
ENABLE_CHESSCC ?= 0
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Widest vector op the elementwise kernel helpers may emit (128/256/512).
# 256 keeps the tree off the 512-bit load/store/mac forms; set 512 to
# restore the full-register versions. aie::mmul shapes are unaffected.
FLM_VECTOR_BITS ?= 256

KERNEL_ARGS := -DMODEL_TYPE=${MODEL_TYPE} -DFLM_VECTOR_BITS=${FLM_VECTOR_BITS}
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
