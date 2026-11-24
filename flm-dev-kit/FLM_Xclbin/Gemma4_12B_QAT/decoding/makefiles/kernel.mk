# common.mk
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
# Created by Alfred

# Kernel stuff
KERNEL_O_DIR := build/$(MODEL_TYPE)/kernels
KERNEL_SRCS := $(wildcard ${HOME_DIR}/kernels/*.cc)
KERNEL_OBJS := $(patsubst ${HOME_DIR}/kernels/%.cc, ${KERNEL_O_DIR}/%.o, $(KERNEL_SRCS))
KERNEL_HEADERS := $(wildcard ${HOME_DIR}/kernels/*.h)

# The sliding-window attention cores run the same sources as the global ones,
# compiled for the other role: different GQA decomposition, different head dim,
# and a swa_-prefixed entry point. See kernels/attn_role.h.
SWA_KERNEL_OBJS := ${KERNEL_O_DIR}/swa_attn_qk.o ${KERNEL_O_DIR}/swa_attn_kv.o
KERNEL_OBJS += ${SWA_KERNEL_OBJS}

ENABLE_CHESSCC ?= 0

# Build kernels
${KERNEL_O_DIR}/%.o: ${HOME_DIR}/kernels/%.cc ${KERNEL_HEADERS}
	mkdir -p ${@D}
ifeq ($(ENABLE_CHESSCC),1)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DNPU2 ${KERNEL_ARGS} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DNPU2 ${KERNEL_ARGS} -c $< -o ${@F}
endif

${SWA_KERNEL_OBJS}: ${KERNEL_O_DIR}/swa_attn_%.o: ${HOME_DIR}/kernels/attn_%.cc ${KERNEL_HEADERS}
	mkdir -p ${@D}
ifeq ($(ENABLE_CHESSCC),1)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DNPU2 ${KERNEL_ARGS} -DATTN_ROLE=ATTN_ROLE_SWA -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DNPU2 ${KERNEL_ARGS} -DATTN_ROLE=ATTN_ROLE_SWA -c $< -o ${@F}
endif