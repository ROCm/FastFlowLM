# common.mk
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Created by Alfred

# Kernel stuff
KERNEL_O_DIR := build/kernel
KERNEL_SRCS := $(wildcard ${HOME_DIR}/kernel/*.cc)
KERNEL_OBJS := $(patsubst ${HOME_DIR}/kernel/%.cc, ${KERNEL_O_DIR}/%.o, $(KERNEL_SRCS))
KERNEL_HEADERS := $(wildcard ${HOME_DIR}/kernel/*.h)

# Build kernels
${KERNEL_O_DIR}/%.o: ${HOME_DIR}/kernel/%.cc ${KERNEL_HEADERS}
	mkdir -p ${@D}
ifeq ($(DEVICE),npu1)
ifeq ($(ENABLE_CHESSCC),1)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -D AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -D AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -DBIT_WIDTH=8 -c $< -o ${@F}
	${PEANO_INSTALL_DIR}/bin/llvm-objdump -d -g -S -s -t -T -x --full-contents $@ > ${@:.o=.s}
	${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8  -S -fverbose-asm   -o ${@:.o=.asm} $<  
endif
else ifeq ($(DEVICE),npu2)
ifeq ($(ENABLE_CHESSCC),1)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DNPU2 -D AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -D AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -DBIT_WIDTH=8 -c $< -o ${@F}
	${PEANO_INSTALL_DIR}/bin/llvm-objdump -d -g -S -s -t -T -x --full-contents $@ > ${@:.o=.s}
	${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8  -S -fverbose-asm   -o ${@:.o=.asm} $<  
endif
else
	echo "Device type not supported"
endif

