#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred

include ../makefiles/common.mk

DEVICE ?= npu2
TRACE_SIZE ?= 0
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/..
HOST_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
# do you want chesscc? compiles slower
ENABLE_CHESSCC ?= 0


NUM_OF_COLUMNS?=8
NUM_OF_CT_PER_COLUMN?=4
r?=8
s?=8
t?=8
DH?=64
LQ?=56
LK?=56
NUM_HEAD?=16
SHIM_L_SEQ?=1792
sHIM_S_SEQ?=1792
LOW_PRECISION?=1
USE_CASUAL_MASK?=0


MODEL_TYPE ?=

# Model parameter lookup table
# Usage: make bitstream MODEL_TYPE=qwen2-5-vision-window-attention
# If MODEL_TYPE is empty, the defaults above are used.
ifneq ($(MODEL_TYPE),)
  ifeq ($(MODEL_TYPE),QWEN2_5-VISION-WINDOW)
    r=8
    s=8
    t=8
    DH=80
    LQ=64
    LK=64
	NUM_OF_COLUMNS=8
	NUM_OF_CT_PER_COLUMN=4
    LOW_PRECISION=1
    USE_CASUAL_MASK=0
  else ifeq ($(MODEL_TYPE),QWEN3_5-OMNI-VISION-WINDOW)
    r=8
    s=8
    t=8
    DH=64
    LQ=56
    LK=56
	NUM_OF_COLUMNS=8
	NUM_OF_CT_PER_COLUMN=4	
    LOW_PRECISION=1
    USE_CASUAL_MASK=0
  else
    $(error Unknown MODEL_TYPE '$(MODEL_TYPE)'. Valid models: qwen2-5-vision-window-attention, qwen3-5-omni-vision-window-attention)
  endif
endif





IRON_ARGS := ${DEVICE}
IRON_ARGS += ${TRACE_SIZE}

IRON_ARGS += ${NUM_OF_COLUMNS}
IRON_ARGS += ${NUM_OF_CT_PER_COLUMN}
IRON_ARGS += ${r}
IRON_ARGS += ${s}
IRON_ARGS += ${t}
IRON_ARGS += ${DH}
IRON_ARGS += ${LQ}
IRON_ARGS += ${LK}
IRON_ARGS+= ${LOW_PRECISION}
IRON_ARGS+= ${USE_CASUAL_MASK}
IRON_ARGS += ${NUM_HEAD}
IRON_ARGS += ${SHIM_L_SEQ}
IRON_ARGS += ${sHIM_S_SEQ}



# Kernel makefile
include ../makefiles/kernel.mk

# Bitstream makefile
include ../makefiles/bitstream.mk
include ../makefiles/mlir_bitstream.mk

# Host makefile
include ../makefiles/host.mk

.PHONY: run all kernel link bitstream host clean instructions
all: ${XCLBIN_TARGETS} ${INSTS_TARGETS} ${HOST_C_TARGET}

clean:
	-@rm -rf build 
	-@rm -rf log
	-@rm -rf *.exe
	-@rm -rf trace*
	-@rm -rf routes
	-@rm -rf *.json

test:
	echo "test"
	echo ${AIEOPT_DIR}


kernel: ${KERNEL_OBJS}


instructions: ${INSTS_TARGETS}



bitstream: ${XCLBIN_TARGETS}


host: ${HOST_C_TARGET}


clean_host:
	-@rm -rf build/host


run: ${HOST_C_TARGET} ${XCLBIN_TARGETS} ${INSTS_TARGETS}
	./${HOST_C_TARGET} ${IRON_ARGS} | tee out.log

route: ${IRON_BOTH_MLIR_TARGET}
	mkdir -p routes
	aie-opt --aie-create-pathfinder-flows --aie-find-flows ${<} | aie-translate --aie-flows-to-json > ./routes/route.json
	python visualize.py -j ./routes/route.json

