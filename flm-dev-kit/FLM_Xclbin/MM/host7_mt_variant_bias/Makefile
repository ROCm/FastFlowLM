#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred
CXXFLAGS?= -O3 -march=native -mavx512f -mavx512bf16
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
HOST_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
MlirDIR ?= $(or $(MLIR_AIE_DIR),$(HOME)/mlir-aie)
DEVICE ?= npu2
TRACE_SIZE ?= 0
# do you want chesscc? compiles slower
ENABLE_CHESSCC ?=0
ENABLE_EMULATION_IN_COMPILER?=0
M=2048
K=2560
N=2048
m=128
k=512
n=64
IS_B_ROW_MAJOR?=0
row_num= 4
col_num= 8
r=4
s=8
t=8
ENABLE_AXI4 ?= 1
IS_B_COL_MAJOR_S_T_ROW_MAJOR?=0  # only vaild when B is Col major. True if B is Col major, but sxt block is row major
IS_BFLOAT16_OUT_TY?=1 # 1 if output in bfloat16, else output in float32 type
ADD_BIAS?=0 # 1 if add bias, 0 else wise
OUTPUT_MODE?=1 # 0 oputput directly, 1 with gelu
OUTPUT_CLAMP?=0 # 1 if clamp output, 0 else wise


MODEL_TYPE ?=
ifneq ($(MODEL_TYPE),)
  ifeq ($(MODEL_TYPE),GENERIC)
	ENABLE_CHESSCC=0
	ENABLE_EMULATION_IN_COMPILER=1
	m=64
	k=512
	n=128
	IS_B_ROW_MAJOR=0
	row_num= 4
	col_num= 8
	r=8
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=1
	IS_BFLOAT16_OUT_TY=1
  else ifeq ($(MODEL_TYPE),GPTOSS-SHORT-SEQ)
	ENABLE_CHESSCC=1
	ENABLE_EMULATION_IN_COMPILER=1
	m=32
	k=128
	n=128
	IS_B_ROW_MAJOR=0
	row_num= 4
	col_num= 8
	r=8
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
	IS_BFLOAT16_OUT_TY=1
  else ifeq ($(MODEL_TYPE),GPTOSS-MAIN)
	ENABLE_CHESSCC=1
	ENABLE_EMULATION_IN_COMPILER=1
	m=64
	k=128
	n=128
	IS_B_ROW_MAJOR=0
	row_num= 4
	col_num= 8
	r=8
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
	IS_BFLOAT16_OUT_TY=1

  else ifeq ($(MODEL_TYPE), GEMMA3_4B-VISION-MM)
	ENABLE_CHESSCC=0
	ENABLE_EMULATION_IN_COMPILER=1
	m=128
	k=256
	n=64
	IS_B_ROW_MAJOR=0
	row_num=4
	col_num=8
	r=8
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
	IS_BFLOAT16_OUT_TY=1
  else ifeq ($(MODEL_TYPE), GEMMA4-VISION-MM)
	ENABLE_CHESSCC=0
	ENABLE_EMULATION_IN_COMPILER=1
	m=128
	k=256
	n=64
	IS_B_ROW_MAJOR=0
	row_num=4
	col_num=8
	r=8
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
	IS_BFLOAT16_OUT_TY=1
  else ifeq ($(MODEL_TYPE), GEMMA4-VISION-MM-HIGH)
	ENABLE_CHESSCC=0
	ENABLE_EMULATION_IN_COMPILER=0
	m=128
	k=256
	n=64
	IS_B_ROW_MAJOR=0
	row_num=4
	col_num=8
	r=4
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
	IS_BFLOAT16_OUT_TY=1
  else ifeq ($(MODEL_TYPE), QWEN2_5_3B-VISION-MM)
	ENABLE_CHESSCC=0
	ENABLE_EMULATION_IN_COMPILER=1
	m=128
	k=256
	n=64
	IS_B_ROW_MAJOR=0
	row_num=4
	col_num=8
	r=8
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
	IS_BFLOAT16_OUT_TY=1
  else ifeq ($(MODEL_TYPE), QWEN3_4B-VISION-MM)
	ENABLE_CHESSCC=0
	ENABLE_EMULATION_IN_COMPILER=1
	m=128
	k=512
	n=64
	IS_B_ROW_MAJOR=0
	row_num=4
	col_num=8
	r=8
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
	IS_BFLOAT16_OUT_TY=1
  else ifeq ($(MODEL_TYPE), QWEN3_5_0_8B-VISION-MM)
	ENABLE_CHESSCC=0
	ENABLE_EMULATION_IN_COMPILER=1
	m=128
	k=256
	n=64
	IS_B_ROW_MAJOR=0
	row_num=4
	col_num=8
	r=8
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
	IS_BFLOAT16_OUT_TY=1
  else ifeq ($(MODEL_TYPE), QWEN3_5_2B-VISION-MM)
	ENABLE_CHESSCC=0
	ENABLE_EMULATION_IN_COMPILER=1
	m=128
	k=512
	n=64
	IS_B_ROW_MAJOR=0
	row_num=4
	col_num=8
	r=8
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
	IS_BFLOAT16_OUT_TY=1
  else ifeq ($(MODEL_TYPE), QWEN3_5_4B-VISION-MM)
	ENABLE_CHESSCC=0
	ENABLE_EMULATION_IN_COMPILER=1
	m=128
	k=512
	n=64
	IS_B_ROW_MAJOR=0
	row_num=4
	col_num=8
	r=8
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
	IS_BFLOAT16_OUT_TY=1

  else ifeq ($(MODEL_TYPE), QWEN3_5_9B-VISION-MM)
	ENABLE_CHESSCC=0
	ENABLE_EMULATION_IN_COMPILER=1
	m=128
	k=256
	n=64
	IS_B_ROW_MAJOR=0
	row_num=4
	col_num=8
	r=8
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
	IS_BFLOAT16_OUT_TY=1

  else ifeq ($(MODEL_TYPE), QWEN3_6_35B-VISION-MM)
	ENABLE_CHESSCC=0
	ENABLE_EMULATION_IN_COMPILER=1
	m=128
	k=256
	n=64
	IS_B_ROW_MAJOR=0
	row_num=4
	col_num=8
	r=8
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
	IS_BFLOAT16_OUT_TY=1
  else ifeq ($(MODEL_TYPE), QWEN3_5-OMNI-VISION-MM)
	ENABLE_CHESSCC=0
	ENABLE_EMULATION_IN_COMPILER=1
	m=128
	k=512
	n=64
	IS_B_ROW_MAJOR=0
	row_num=4
	col_num=8
	r=8
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
	IS_BFLOAT16_OUT_TY=1
  else ifeq ($(MODEL_TYPE), QWEN3_5-OMNI-VISION-MM-HIGH)
	ENABLE_CHESSCC=0
	ENABLE_EMULATION_IN_COMPILER=0
	m=128
	k=512
	n=64
	IS_B_ROW_MAJOR=0
	row_num=4
	col_num=8
	r=4
	s=8
	t=8
	IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
	IS_BFLOAT16_OUT_TY=1
  else
	$(error Unknown MODEL_TYPE: $(MODEL_TYPE))
  endif
 
endif




include ${HOME_DIR}/makefiles/common.mk

IRON_ARGS := ${DEVICE}
IRON_ARGS += ${TRACE_SIZE}
IRON_ARGS += ${M}
IRON_ARGS += ${K}
IRON_ARGS += ${N}
IRON_ARGS += ${m}
IRON_ARGS += ${k}
IRON_ARGS += ${n}
IRON_ARGS += ${IS_B_ROW_MAJOR}
IRON_ARGS += ${row_num}
IRON_ARGS += ${col_num}
IRON_ARGS += ${r}
IRON_ARGS += ${s}
IRON_ARGS += ${t}
IRON_ARGS += ${ENABLE_AXI4}
IRON_ARGS += ${IS_B_COL_MAJOR_S_T_ROW_MAJOR}
IRON_ARGS += ${IS_BFLOAT16_OUT_TY}
IRON_ARGS += ${ADD_BIAS}
IRON_ARGS += ${OUTPUT_MODE}
IRON_ARGS += ${OUTPUT_CLAMP}
# Kernel makefile
include ${HOME_DIR}/makefiles/kernel.mk

# Bitstream makefile
include ${HOME_DIR}/makefiles/bitstream.mk
include ${HOME_DIR}/makefiles/mlir_bitstream.mk

# Host makefile
include ${HOME_DIR}/makefiles/host.mk

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

mlir_test:
	python ./iron/mm_bf16_accf32_aie.py ${IRON_ARGS}

clean_host:
	-@rm -rf build/host


run: ${HOST_C_TARGET} ${XCLBIN_TARGETS} ${INSTS_TARGETS}
	./${HOST_C_TARGET} ${IRON_ARGS} | tee out.log

route: ${IRON_BOTH_MLIR_TARGET}
	mkdir -p routes
	aie-opt --aie-create-pathfinder-flows --aie-find-flows ${<} | aie-translate --aie-flows-to-json > ./routes/route.json
	python ../visualize.py -j ./routes/route.json


trace: trace.txt
	${MlirDIR}/programming_examples/utils/parse_trace.py --input trace.txt --mlir build/mlir/mm_bf16_accf32.mlir  --output mm.json
	${MlirDIR}/programming_examples/utils/get_trace_summary.py --input mm.json
