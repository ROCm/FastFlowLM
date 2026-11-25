# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

MLIR_O_DIR := build/mlir
BITSTREAM_O_DIR := build/bitstream

IRON_SRC := $(wildcard ${HOME_DIR}/iron/*_aie.py)
IRON_OTHER_SRC := $(wildcard ${HOME_DIR}/iron/*.py)
IRON_MLIR_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${MLIR_O_DIR}/%.mlir, ${IRON_SRC})
IRON_XCLBIN_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${BITSTREAM_O_DIR}/from_iron/%.xclbin, ${IRON_SRC})

.PRECIOUS: ${IRON_MLIR_TARGET}

AIECC_FLAGS := --aie-generate-cdo --no-compile-host --no-xchesscc --no-xbridge --alloc-scheme=basic-sequential

${MLIR_O_DIR}/%.mlir: ${HOME_DIR}/iron/%_aie.py ${IRON_OTHER_SRC}
	mkdir -p ${@D}
	python3 $< > $@

${BITSTREAM_O_DIR}/from_iron/%.xclbin: ${MLIR_O_DIR}/%.mlir ${KERNEL_OBJS}
	mkdir -p ${@D}
	cp ${KERNEL_OBJS} ${@D}
	cd ${@D} && aiecc.py ${AIECC_FLAGS} --aie-generate-xclbin \
		--xclbin-name=${@F} $(<:${MLIR_O_DIR}/%=../../mlir/%)
	python3 ${HOME_DIR}/check_routes.py ${@D}/$(*F).mlir.prj/input_physical.mlir $<
	mkdir -p build/xclbins
	cp ${@} build/xclbins/

XCLBIN_TARGETS += ${IRON_XCLBIN_TARGET}
