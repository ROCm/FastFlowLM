#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
# Modified by Alfred

include makefiles/common.mk
MODEL_TYPE ?= HY_MEMORY_30B_A3B

# 1 = chess, 0 = Peano (default). Both give bit-identical outputs; see README.md, "Peano".
ENABLE_CHESSCC ?= 0
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# ATTN_TAP=1 also sends the attention output (before o_proj) to the host, for
# checking it against the reference. It builds into its own folder, so a tap
# build and a normal build never mix; run it with `run.exe --attn_tap`.
ATTN_TAP ?= 0
# DEBUG_TAP=x (debug only): copies of the x stream to IT4, see layer_aie.py
DEBUG_TAP ?=
BUILD_DIR := build/$(MODEL_TYPE)$(if $(filter 1,$(ATTN_TAP)),_attn_tap)$(if $(DEBUG_TAP),_dbg_$(DEBUG_TAP))

KERNEL_ARGS := -DMODEL_TYPE=${MODEL_TYPE}
IRON_ARGS := ${MODEL_TYPE} ${ATTN_TAP} ${DEBUG_TAP}

VERBOSE ?= 0

# Kernel makefile
include makefiles/kernel.mk

# Bitstream makefile
include makefiles/bitstream.mk

# Host makefile
include makefiles/host.mk

.PHONY: all kernel link bitstream host clean clean_all test route
all: ${XCLBIN_TARGETS} ${HOST_C_TARGET}

clean_all:
	-@rm -rf build
	-@rm -rf log
	-@rm -rf *.exe
	-@rm -rf trace*
	-@rm -rf routes

kernel: ${KERNEL_OBJS}

link: ${IRON_BOTH_MLIR_TARGET}

bitstream: ${XCLBIN_TARGETS}

host: ${HOST_C_TARGET}

clean:
	-@rm -rf build/host
	-@rm -rf ${HOST_C_TARGET}

test: $(HOST_C_TARGET)
	export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib && ./$(HOST_C_TARGET) -m ${MODEL_TYPE}$(if $(filter 1,$(ATTN_TAP)), --attn_tap)

route: ${IRON_BOTH_MLIR_TARGET}
	mkdir -p routes
	aie-opt --aie-create-pathfinder-flows --aie-find-flows ${<} | aie-translate --aie-flows-to-json > ./routes/route.json
	python visualize.py -j ./routes/route.json
