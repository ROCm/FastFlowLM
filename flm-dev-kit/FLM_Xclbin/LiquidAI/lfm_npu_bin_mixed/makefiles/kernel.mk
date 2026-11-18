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
KERNEL_SRCS := $(wildcard ${HOME_DIR}/kernel/*.cc)
KERNEL_OBJS := $(patsubst ${HOME_DIR}/kernel/%.cc, ${KERNEL_O_DIR}/%.o, $(KERNEL_SRCS))
KERNEL_HEADERS := $(wildcard ${HOME_DIR}/kernel/*.h)

ENABLE_CHESSCC ?= 0

# attention kernels need -O1 on peano (spill mitigation)
${KERNEL_O_DIR}/attn_kv.o: KERNEL_PEANO_EXTRA := -O1
${KERNEL_O_DIR}/attn_qk.o: KERNEL_PEANO_EXTRA := -O1
# rms_residual: fused rms+lm_head core; -O2 overflows 16KB prog mem, so build at -O1.
# (The deadlock is fixed by noinline on pseduo_rms_norm in the kernel, not the opt level.)
${KERNEL_O_DIR}/rms_residual.o: KERNEL_PEANO_EXTRA := -O1

# Build kernels
${KERNEL_O_DIR}/%.o: ${HOME_DIR}/kernel/%.cc ${KERNEL_HEADERS}
	mkdir -p ${@D}
ifeq ($(ENABLE_CHESSCC),1)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DAIE_FP32_EMULATION_ACCURACY_FAST -DNPU2 ${KERNEL_ARGS} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${KERNEL_PEANO_EXTRA} -DNPU2 ${KERNEL_ARGS} -c $< -o ${@F}
endif