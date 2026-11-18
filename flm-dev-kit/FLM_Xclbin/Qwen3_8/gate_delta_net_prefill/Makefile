#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred

include ./makefiles/common.mk

DEVICE ?= npu2
TRACE_SIZE ?= 0

L=1120
DQ=128
DK=128
DV=128
NUM_OF_ROW=4
NUM_OF_COLUMN=8

MT_L_chunk_size=128
CT_L_chunk_size=16
NUM_HEAD=48
QK_HEAD_GROUP=3 # how many times the QK head is repeated, since QK head is grouped

QKV_buffer_L_head_row_major?=1 # 1 means the qkv buffer in Lx (  DQ*NUM_Q_K_HEAD + DV*NUM_Q_K_HEAD+ DV*NUM_HEAD) row major, 
# 0 means in  LxDQ*NUM_Q_K_HEAD + L*DK*NUM_Q_K_HEAD + L*DV*NUM_HEAD row major. Note that the qkv buffer is always in the order of Q K V, just the layout is different.




HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/
HOST_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
MlirDIR ?= $(or $(MLIR_AIE_DIR),$(HOME)/mlir-aie)
# do you want chesscc? compiles slower
ENABLE_CHESSCC ?= 0

IRON_ARGS := ${DEVICE}
IRON_ARGS += ${TRACE_SIZE}
IRON_ARGS += ${L}
IRON_ARGS += ${DQ}
IRON_ARGS += ${DK}
IRON_ARGS += ${DV}
IRON_ARGS += ${NUM_OF_ROW}
IRON_ARGS += ${NUM_OF_COLUMN}
IRON_ARGS += ${MT_L_chunk_size}
IRON_ARGS += ${CT_L_chunk_size}
IRON_ARGS += ${NUM_HEAD}
IRON_ARGS += ${QK_HEAD_GROUP}
IRON_ARGS += ${QKV_buffer_L_head_row_major}




# Kernel makefile
include ./makefiles/kernel.mk

# Bitstream makefile
include ./makefiles/bitstream.mk
include ./makefiles/mlir_bitstream.mk

# Host makefile
include ./makefiles/host.mk

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


trace: trace.txt
	${MlirDIR}/programming_examples/utils/parse_trace.py --input trace.txt --mlir build/mlir/MVM_q80.mlir  --output MVM_q80.json
	${MlirDIR}/programming_examples/utils/get_trace_summary.py --input MVM_q80.json
