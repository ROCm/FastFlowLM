# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# MLIR stuff

BITSTREAM_O_DIR := build/bitstream


MLIR_BOTH_SRC := $(wildcard ${HOME_DIR}/mlir/*.mlir)
MLIR_BOTH_XCLBIN_TARGET := $(patsubst ${HOME_DIR}/mlir/%.mlir, ${BITSTREAM_O_DIR}/from_mlir/%.xclbin, ${MLIR_BOTH_SRC})
MLIR_BOTH_INSTS_TARGET := $(patsubst ${HOME_DIR}/mlir/%.mlir, ${BITSTREAM_O_DIR}/from_mlir/%.txt, ${MLIR_BOTH_SRC})

AIECC_FLAGS := --aie-generate-cdo --no-compile-host
ifeq ($(ENABLE_CHESSCC),1)
AIECC_FLAGS += --xchesscc --xbridge
else
AIECC_FLAGS += --no-xchesscc --no-xbridge
endif

# Build xclbin
${BITSTREAM_O_DIR}/from_mlir/%.xclbin: ${HOME_DIR}/mlir/%.mlir ${KERNEL_OBJS}
	mkdir -p ${@D}
	cp ${KERNEL_OBJS} ${@D}
	cd ${@D} && aiecc.py ${AIECC_FLAGS} --aie-generate-xclbin \
		--xclbin-name=${@F} $(<:${HOME_DIR}/mlir/%=../../../mlir/%)
	mkdir -p build/xclbins
	cp ${@} build/xclbins/

# Build instructions

${BITSTREAM_O_DIR}/from_mlir/%.txt: ${HOME_DIR}/mlir/%.mlir 
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -n ${AIECC_FLAGS} --aie-generate-npu-insts \
		--npu-insts-name=${@F} $(<:${HOME_DIR}/mlir/%=../../../mlir/%)
	mkdir -p build/insts
	cp ${@} build/insts/


INSTS_TARGETS += ${MLIR_BOTH_INSTS_TARGET} 
XCLBIN_TARGETS += ${MLIR_BOTH_XCLBIN_TARGET}
