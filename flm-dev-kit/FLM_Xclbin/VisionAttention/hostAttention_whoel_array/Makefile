# 
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred

include ../makefiles/common.mk

DEVICE ?= npu2
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/..
HOST_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
MlirDIR ?= $(or $(MLIR_AIE_DIR),$(HOME)/mlir-aie)
# do you want chesscc? compiles slower
ENABLE_CHESSCC ?= 0

r=8
s=8
t=8

DH=64
LQ_PER_CT=32
LK_PER_CT=512
LQ_internal=32
lk_lv_internal=32


NUM_HEAD=16

SHIM_L_SEQ=4096
SHIM_S_SEQ=4096

NUM_OF_COLUMNS=8
NUM_OF_CT_PER_COLUMN=4
CU_MODE=2
LOW_PRECISION := 1  # 1: low precision, 0: normal precision
RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION := 1  # 1: start from L dimension, 0: start from DH dimension

MODEL_TYPE ?=

# Model parameter lookup table
# Usage: make bitstream MODEL_TYPE=QWEN_2_5-VISION-ATTENTION
# If MODEL_TYPE is empty, the defaults above are used.
ifneq ($(MODEL_TYPE),)
  ifeq ($(MODEL_TYPE),QWEN_2_5-VISION-ATTENTION)
    r=8
    s=8
    t=8
    DH=80
    LQ_PER_CT=32
    LK_PER_CT=512
    LQ_internal=32
    lk_lv_internal=32
    CU_MODE=2
    NUM_OF_COLUMNS=8
    NUM_OF_CT_PER_COLUMN=4
    LOW_PRECISION=1
  else ifeq ($(MODEL_TYPE),QWEN3_4B-VISION-ATTENTION)
    r=8
    s=8
    t=8
    DH=64
    LQ_PER_CT=32
    LK_PER_CT=512
    LQ_internal=32
    lk_lv_internal=32
    CU_MODE=2
    NUM_OF_COLUMNS=8
    NUM_OF_CT_PER_COLUMN=4
    LOW_PRECISION=1
  else ifeq ($(MODEL_TYPE),QWEN3_5_0_8B-VISION-ATTENTION)
    r=8
    s=8
    t=8
    DH=64
    LQ_PER_CT=32
    LK_PER_CT=512
    LQ_internal=32
    lk_lv_internal=32
    CU_MODE=2
    NUM_OF_COLUMNS=8
    NUM_OF_CT_PER_COLUMN=4
    LOW_PRECISION=1
  else ifeq ($(MODEL_TYPE),QWEN3_5_2B-VISION-ATTENTION)
    r=8
    s=8
    t=8
    DH=64
    LQ_PER_CT=32
    LK_PER_CT=512
    LQ_internal=32
    lk_lv_internal=32
    CU_MODE=2
    NUM_OF_COLUMNS=8
    NUM_OF_CT_PER_COLUMN=4
    LOW_PRECISION=1
  else ifeq ($(MODEL_TYPE),QWEN3_5_4B-VISION-ATTENTION)
    r=8
    s=8
    t=8
    DH=64
    LQ_PER_CT=32
    LK_PER_CT=512
    LQ_internal=32
    lk_lv_internal=32
    CU_MODE=2
    NUM_OF_COLUMNS=8
    NUM_OF_CT_PER_COLUMN=4
    LOW_PRECISION=1
  else ifeq ($(MODEL_TYPE),QWEN3_5_9B-VISION-ATTENTION)
    r=8
    s=8
    t=8
    DH=72
    LQ_PER_CT=32
    LK_PER_CT=512
    LQ_internal=32
    lk_lv_internal=32
    CU_MODE=2
    NUM_OF_COLUMNS=8
    NUM_OF_CT_PER_COLUMN=4
    LOW_PRECISION=1
  else ifeq ($(MODEL_TYPE),QWEN3_5_OMNI-VISION-ATTENTION)
    r=8
    s=8
    t=8
    DH=64
    LQ_PER_CT=32
    LK_PER_CT=512
    LQ_internal=32
    lk_lv_internal=32
    CU_MODE=2
    NUM_OF_COLUMNS=8
    NUM_OF_CT_PER_COLUMN=4
    LOW_PRECISION=1
  else ifeq ($(MODEL_TYPE),QWEN3_6_35B-VISION-ATTENTION)
    r=8
    s=8
    t=8
    DH=72
    LQ_PER_CT=32
    LK_PER_CT=512
    LQ_internal=32
    lk_lv_internal=32
    CU_MODE=2
    NUM_OF_COLUMNS=8
    NUM_OF_CT_PER_COLUMN=4
    LOW_PRECISION=1
  else ifeq ($(MODEL_TYPE),GEMMA3_4B-VISION-ATTENTION)
    r=8
    s=8
    t=8
    DH=72
    LQ_PER_CT=32
    LK_PER_CT=512
    LQ_internal=32
    lk_lv_internal=32
    CU_MODE=1
    NUM_OF_COLUMNS=8
    NUM_OF_CT_PER_COLUMN=4
    LOW_PRECISION=1
  else ifeq ($(MODEL_TYPE),GEMMA4_E2B-VISION-ATTENTION)
    r=8
    s=8
    t=8
    DH=64
    LQ_PER_CT=32
    LK_PER_CT=512
    LQ_internal=32
    lk_lv_internal=32
    CU_MODE=2
    NUM_OF_COLUMNS=8
    NUM_OF_CT_PER_COLUMN=4
    LOW_PRECISION=1
  else ifeq ($(MODEL_TYPE),GEMMA4_E4B-VISION-ATTENTION)
    r=8
    s=8
    t=8
    DH=64
    LQ_PER_CT=32
    LK_PER_CT=512
    LQ_internal=32
    lk_lv_internal=32
    CU_MODE=2
    NUM_OF_COLUMNS=8
    NUM_OF_CT_PER_COLUMN=4
    LOW_PRECISION=1
  else
    $(error Unknown MODEL_TYPE '$(MODEL_TYPE)'. Valid models: QWEN_2_5-VISION-ATTENTION, QWEN3_4B-VISION-ATTENTION, QWEN3_5_0_8B-VISION-ATTENTION, QWEN3_5_2B-VISION-ATTENTION, QWEN3_5_4B-VISION-ATTENTION, QWEN3_5_9B-VISION-ATTENTION, QWEN3_6_35B-VISION-ATTENTION, GEMMA3_4B-VISION-ATTENTION, GEMMA4_E2B-VISION-ATTENTION, GEMMA4_E4B-VISION-ATTENTION)
  endif
endif







IRON_ARGS := ${DEVICE}
IRON_ARGS += ${r}
IRON_ARGS += ${s}
IRON_ARGS += ${t}
IRON_ARGS += ${DH}
IRON_ARGS += ${LQ_PER_CT}
IRON_ARGS += ${LK_PER_CT}
IRON_ARGS += ${LQ_internal}
IRON_ARGS += ${lk_lv_internal}
IRON_ARGS += ${NUM_HEAD}
IRON_ARGS += ${SHIM_L_SEQ}
IRON_ARGS += ${SHIM_S_SEQ}
IRON_ARGS += ${NUM_OF_COLUMNS}
IRON_ARGS += ${NUM_OF_CT_PER_COLUMN}
IRON_ARGS += ${CU_MODE}
IRON_ARGS += ${LOW_PRECISION}
IRON_ARGS += ${RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION}

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


trace: trace.txt
	${MlirDIR}/programming_examples/utils/parse_trace.py --input trace.txt --mlir build/mlir/attention_bf16.mlir  --output attention_bf16.json
	${MlirDIR}/programming_examples/utils/get_trace_summary.py --input attention_bf16.json
