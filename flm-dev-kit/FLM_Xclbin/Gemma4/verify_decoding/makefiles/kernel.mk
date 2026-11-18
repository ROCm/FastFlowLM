# common.mk
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
# Created by Alfred

# Kernel stuff
KERNEL_O_DIR := $(BUILD_ROOT)/kernels
KERNEL_SRCS := $(wildcard ${HOME_DIR}/kernels/*.cc)
KERNEL_OBJS ?= $(patsubst ${HOME_DIR}/kernels/%.cc, ${KERNEL_O_DIR}/%.o, $(KERNEL_SRCS))
# fused8 links only its own cores (the other kernels belong to the decode layer and the stage designs)
ifeq ($(DESIGN),fused8)
FUSED8_ATTN_KERNELS := $(if $(filter x,$(FUSED8_ATTN)),attnx_score attnx_pv,attn_qk_e4b attn_kv_e4b swa_attn_qk swa_attn_kv)
KERNEL_OBJS := $(patsubst %,${KERNEL_O_DIR}/%.o,proj_main rms_residual_l8 glu_l8 rope swa_rope $(FUSED8_ATTN_KERNELS) ple8_gate ple8_up ple8_norm)
endif
KERNEL_HEADERS := $(wildcard ${HOME_DIR}/kernels/*.h)

# Build kernels
ENABLE_CHESSCC ?= 0

# Peano miscompiles the attention lock loops at -O2 (see ROCm/FastFlowLM PR#4);
# build the attention kernels at -O1. chess (ENABLE_CHESSCC=1) is unaffected.
${KERNEL_O_DIR}/attn_kv.o: KERNEL_PEANO_EXTRA := -O1
${KERNEL_O_DIR}/attn_qk.o: KERNEL_PEANO_EXTRA := -O1
${KERNEL_O_DIR}/swa_attn_kv.o: KERNEL_PEANO_EXTRA := -O1
${KERNEL_O_DIR}/swa_attn_qk.o: KERNEL_PEANO_EXTRA := -O1

${KERNEL_O_DIR}/%.o: ${HOME_DIR}/kernels/%.cc ${KERNEL_HEADERS}
	mkdir -p ${@D}
ifeq ($(ENABLE_CHESSCC),1)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DNPU2 ${KERNEL_ARGS} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${KERNEL_PEANO_EXTRA} -DNPU2 ${KERNEL_ARGS} -c $< -o ${@F}
endif