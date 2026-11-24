# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# MLIR stuff

MLIR_O_DIR := build/mlir
BITSTREAM_O_DIR := build/bitstream

# Detect IRON_ARGS changes and wipe mlir/bitstream (not kernel) so xclbin rebuilds.
_IRON_STAMP := ${HOME_DIR}/.iron_args_stamp
$(shell \
  if [ "$$(cat $(_IRON_STAMP) 2>/dev/null)" != "$(IRON_ARGS)" ]; then \
    echo "IRON_ARGS changed, wiping build/mlir and build/bitstream..." >&2; \
    rm -rf ${HOME_DIR}/build/mlir ${HOME_DIR}/build/bitstream; \
    echo "$(IRON_ARGS)" > $(_IRON_STAMP); \
  fi)


IRON_BOTH_SRC := $(wildcard ${HOST_DIR}/iron/*_aie.py)
IRON_BOTH_MLIR_TARGET := $(patsubst ${HOST_DIR}/iron/%_aie.py, ${MLIR_O_DIR}/%.mlir, ${IRON_BOTH_SRC})
IRON_BOTH_XCLBIN_TARGET := $(patsubst ${HOST_DIR}/iron/%_aie.py, ${BITSTREAM_O_DIR}/from_iron/%.xclbin, ${IRON_BOTH_SRC})
IRON_BOTH_INSTS_TARGET := $(patsubst ${HOST_DIR}/iron/%_aie.py, ${BITSTREAM_O_DIR}/from_iron/%.txt, ${IRON_BOTH_SRC})

OTHER_IRON_SRC := $(wildcard ${HOST_DIR}/iron/*.py)

.PRECIOUS: ${IRON_BOTH_MLIR_TARGET}

AIECC_FLAGS := --aie-generate-cdo --no-compile-host
ifeq ($(ENABLE_CHESSCC),1)
AIECC_FLAGS += --xchesscc --xbridge
else
AIECC_FLAGS += --no-xchesscc --no-xbridge
endif

# iron generate mlir
${MLIR_O_DIR}/%.mlir: ${HOST_DIR}/iron/%_aie.py ${OTHER_IRON_SRC}
	mkdir -p ${@D}
	python3 $< ${IRON_ARGS} > $@

# Build xclbin
${BITSTREAM_O_DIR}/from_iron/%.xclbin: ${MLIR_O_DIR}/%.mlir ${KERNEL_OBJS}
	mkdir -p ${@D}
	cp ${KERNEL_OBJS} ${@D}
	cd ${@D} && aiecc.py ${AIECC_FLAGS} --aie-generate-xclbin \
		--xclbin-name=${@F} $(<:${MLIR_O_DIR}/%=../../mlir/%)
	mkdir -p build/xclbins
	cp ${@} build/xclbins/

${BITSTREAM_O_DIR}/from_iron/%.txt: ${MLIR_O_DIR}/%.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -n ${AIECC_FLAGS} --aie-generate-npu-insts \
		--npu-insts-name=${@F} $(<:${MLIR_O_DIR}/%=../../mlir/%)
	mkdir -p build/insts
	cp ${@} build/insts/

INSTS_TARGETS += ${IRON_BOTH_INSTS_TARGET}
XCLBIN_TARGETS += ${IRON_BOTH_XCLBIN_TARGET} 
