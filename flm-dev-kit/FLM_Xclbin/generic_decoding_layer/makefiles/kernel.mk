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
KERNEL_HEADERS := $(wildcard ${HOME_DIR}/kernels/*.h) $(wildcard ${HOME_DIR}/kernels/*.hpp)

# Build kernels
# ENABLE_CHESSCC is set by the top-level Makefile (Peano by default).

# attn_qk is built at -O1 under Peano to keep its frame within its stack, which
# cannot grow without moving the RTP buffers (see iron/layer_aie.py). Applies to
# Peano only.
${KERNEL_O_DIR}/attn_qk.o: KERNEL_PEANO_EXTRA := -O1

${KERNEL_O_DIR}/%.o: ${HOME_DIR}/kernels/%.cc ${KERNEL_HEADERS}
	mkdir -p ${@D}
ifeq ($(ENABLE_CHESSCC),1)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DNPU2 ${KERNEL_ARGS} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${KERNEL_PEANO_EXTRA} -DNPU2 ${KERNEL_ARGS} -c $< -o ${@F}
endif