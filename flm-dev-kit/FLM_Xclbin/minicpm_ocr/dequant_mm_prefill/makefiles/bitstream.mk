# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# MLIR stuff

MLIR_O_DIR := build/$(MODEL_TYPE)/mlir
BITSTREAM_O_DIR := build/$(MODEL_TYPE)/bitstream


IRON_BOTH_SRC := $(wildcard ${HOME_DIR}/iron/*_aie.py)
IRON_OTHER_SRC := $(wildcard ${HOME_DIR}/iron/*.py)
IRON_BOTH_MLIR_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${MLIR_O_DIR}/%.mlir, ${IRON_BOTH_SRC})
IRON_BOTH_XCLBIN_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${BITSTREAM_O_DIR}/from_iron/%.xclbin, ${IRON_BOTH_SRC})
IRON_BOTH_INSTS_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${BITSTREAM_O_DIR}/from_iron/%.txt, ${IRON_BOTH_SRC})

.PRECIOUS: ${IRON_BOTH_MLIR_TARGET} 

AIECC_FLAGS := --aie-generate-cdo --no-compile-host
ENABLE_CHESSCC ?= 0
ifeq ($(ENABLE_CHESSCC),1)
AIECC_FLAGS += --xchesscc --xbridge
else
AIECC_FLAGS += --no-xchesscc --no-xbridge
endif
# iron generate mlir
${MLIR_O_DIR}/%.mlir: ${HOME_DIR}/iron/%_aie.py ${IRON_OTHER_SRC}
	mkdir -p ${@D}
	python3 $< ${IRON_ARGS} > $@

# Build xclbin
${BITSTREAM_O_DIR}/from_iron/%.xclbin: ${MLIR_O_DIR}/%.mlir ${KERNEL_OBJS}
	mkdir -p ${@D}
	cp ${KERNEL_OBJS} ${@D}
	cd ${@D} && aiecc.py ${AIECC_FLAGS} --aie-generate-xclbin \
		--xclbin-name=${@F} $(<:${MLIR_O_DIR}/%=../../mlir/%)
	@grep -hoE 'address = [0-9]+ : i32, sym_name = "(K_div_k|rtp_act)_[0-9]+_[0-9]+"' ${@D}/*/input_with_addresses.mlir \
	  | sed -E 's/address = ([0-9]+) : i32, sym_name = "(K_div_k|rtp_act)_([0-9]+)_[0-9]+"/\1 \2 \3/' \
	  | awk '{ want = ($$2 == "K_div_k" ? 55552 : 55616) + ($$3 % 2 ? 8960 : 0); n++ } \
	         $$1 != want { printf "error: %s buffer on row %s is at %s, FLM_DLL expects %s\n", $$2, $$3, $$1, want; bad = 1 } \
	         END { if (n == 0) { print "error: no K_div_k / rtp_act buffers found"; bad = 1 } exit bad }'
	mkdir -p build/$(MODEL_TYPE)/xclbins
	cp ${@} build/$(MODEL_TYPE)/xclbins/

${BITSTREAM_O_DIR}/from_iron/%.txt: ${MLIR_O_DIR}/%.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -n ${AIECC_FLAGS} --aie-generate-npu-insts \
		--npu-insts-name=${@F} $(<:${MLIR_O_DIR}/%=../../mlir/%)
	mkdir -p build/$(MODEL_TYPE)/insts
	cp ${@} build/$(MODEL_TYPE)/insts/

INSTS_TARGETS += ${IRON_BOTH_INSTS_TARGET}
XCLBIN_TARGETS += ${IRON_BOTH_XCLBIN_TARGET} 
