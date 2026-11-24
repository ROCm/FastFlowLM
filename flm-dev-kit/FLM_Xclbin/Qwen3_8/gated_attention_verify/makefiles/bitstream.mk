# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# MLIR stuff

MLIR_O_DIR := build/$(MODEL_TYPE)/mlir
BITSTREAM_O_DIR := build/$(MODEL_TYPE)/bitstream


IRON_BOTH_SRC := $(wildcard ${HOME_DIR}/iron/*_aie.py)
IRON_OTHER_SRC := $(wildcard ${HOME_DIR}/iron/*.py)
IRON_OTHER_SRC += $(wildcard ${HOME_DIR}/iron/models/*.py)
IRON_BOTH_MLIR_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${MLIR_O_DIR}/%.mlir, ${IRON_BOTH_SRC})
IRON_BOTH_XCLBIN_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${BITSTREAM_O_DIR}/from_iron/%.xclbin, ${IRON_BOTH_SRC})

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
	@# FLM_DLL writes the attention context length to fixed RTP addresses. A
	@# Peano-only layout change that moves either buffer leaves those cores
	@# reading a stale length, so fail the build instead.
	@grep -hoE 'address = [0-9]+ : i32, mem_bank = [0-9]+ : i32, sym_name = "RTP_L_attn_(qk|kv)_core_[0-9_]+"' ${@D}/*/input_with_addresses.mlir \
	  | sed -E 's/address = ([0-9]+) .*sym_name = "(RTP_L_attn_(qk|kv))_core_([0-9_]+)"/\1 \2 \4/' \
	  | awk '{ want = ($$2 == "RTP_L_attn_qk") ? 57344 : 36864; n++ } \
	         $$1 != want { printf "error: %s_core_%s is at %s, FLM_DLL expects %s\n", $$2, $$3, $$1, want; bad = 1 } \
	         END { if (n == 0) { print "error: no attention RTP buffers found"; bad = 1 } exit bad }'
endif
	mkdir -p build/$(MODEL_TYPE)/xclbins
	cp ${@} build/$(MODEL_TYPE)/xclbins/
	cat ${@D}/*/input_with_addresses.mlir | grep "%RTP" | grep "aie.buffer" | awk '{print $$1, $$6}' | sed 's/%//g' | tee build/$(MODEL_TYPE)/xclbins/rtp_addresses.txt

XCLBIN_TARGETS += ${IRON_BOTH_XCLBIN_TARGET} 
