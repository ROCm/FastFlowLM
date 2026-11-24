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
	ENABLE_CHESSCC=$(ENABLE_CHESSCC) python3 $< ${IRON_ARGS} > $@

# Build xclbin
${BITSTREAM_O_DIR}/from_iron/%.xclbin: ${MLIR_O_DIR}/%.mlir ${KERNEL_OBJS}
	mkdir -p ${@D}
	cp ${KERNEL_OBJS} ${@D}
	cd ${@D} && aiecc.py ${AIECC_FLAGS} --aie-generate-xclbin \
		--xclbin-name=${@F} $(<:${MLIR_O_DIR}/%=../../mlir/%)
ifneq ($(ENABLE_CHESSCC),1)
	@# Peano proj tile layout (iron/proj_tile.py). FLM_DLL writes the proj RTP
	@# buffers at a fixed address (decode_MVM_rtp_address), and the two MXFP4
	@# LUTs must be in different banks or every lookup stalls.
	@grep -hoE 'address = [0-9]+ : i32, mem_bank = [0-9]+ : i32, sym_name = "(proj_core|lut_ab|lut_cd)_[0-9_]+"' ${@D}/*/input_with_addresses.mlir \
	  | sed -E 's/address = ([0-9]+) : i32, mem_bank = ([0-9]+) : i32, sym_name = "(proj_core|lut_ab|lut_cd)_([0-9_]+)"/\1 \2 \3 \4/' \
	  | awk '$$3 == "proj_core" && $$1 != 19040 { printf "error: proj_core_%s is at %s, FLM_DLL expects 19040\n", $$4, $$1; bad = 1 } \
	         $$3 == "lut_ab" { ab[$$4] = $$2 } $$3 == "lut_cd" { cd[$$4] = $$2 } \
	         END { n = 0; for (t in ab) { n++; if (ab[t] == cd[t]) { printf "error: lut_ab_%s and lut_cd_%s are both in bank %s\n", t, t, ab[t]; bad = 1 } } \
	               if (n == 0) { print "error: no MXFP4 LUT buffers found"; bad = 1 } exit bad }'
endif
	mkdir -p build/$(MODEL_TYPE)/xclbins
	cp ${@} build/$(MODEL_TYPE)/xclbins/layer.xclbin

${BITSTREAM_O_DIR}/from_iron/%.txt: ${MLIR_O_DIR}/%.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -n ${AIECC_FLAGS} --aie-generate-npu-insts \
		--npu-insts-name=${@F} $(<:${MLIR_O_DIR}/%=../../mlir/%)
	mkdir -p build/$(MODEL_TYPE)/insts
	cp ${@} build/$(MODEL_TYPE)/insts/

INSTS_TARGETS += ${IRON_BOTH_INSTS_TARGET}
XCLBIN_TARGETS += ${IRON_BOTH_XCLBIN_TARGET} 
