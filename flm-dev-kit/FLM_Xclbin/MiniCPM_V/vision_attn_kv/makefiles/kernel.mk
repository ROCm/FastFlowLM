# common.mk
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
# Created by Alfred

# Kernel stuff
KERNEL_O_DIR := build/$(MODEL_TYPE)/kernel
# attn.o (the core's link_with) = attn.cc + softmax.cc. softmax.cc is compiled alone with the post-pipeliner forced:
# its exp loop drops from ~23 to 3 bundles per vector, while the same flag slows attn.cc's QK / FV mac loops.
KERNEL_OBJS := ${KERNEL_O_DIR}/attn.o
SOFTMAX_FLAGS := -mllvm --aie-force-postpipeliner
KERNEL_HEADERS := $(wildcard ${HOME_DIR}/kernel/*.h)

ENABLE_CHESSCC ?= 0

${KERNEL_O_DIR}/attn.o: ${KERNEL_O_DIR}/attn_core.o ${KERNEL_O_DIR}/softmax.o
	${PEANO_INSTALL_DIR}/bin/ld.lld -r $^ -o $@

${KERNEL_O_DIR}/softmax.o: ${HOME_DIR}/kernel/softmax.cc ${KERNEL_HEADERS}
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DNPU2 ${KERNEL_ARGS} ${SOFTMAX_FLAGS} -c $< -o ${@F}

# Build kernels
${KERNEL_O_DIR}/attn_core.o: ${HOME_DIR}/kernel/attn.cc ${KERNEL_HEADERS}
	mkdir -p ${@D}
ifeq ($(ENABLE_CHESSCC),1)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DNPU2 ${KERNEL_ARGS} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DNPU2 ${KERNEL_ARGS} -c $< -o ${@F}
endif