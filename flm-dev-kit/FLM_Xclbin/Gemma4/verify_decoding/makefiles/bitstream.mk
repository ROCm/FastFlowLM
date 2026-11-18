# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# MLIR stuff

MLIR_O_DIR := $(BUILD_ROOT)/mlir
BITSTREAM_O_DIR := $(BUILD_ROOT)/bitstream
XCLBIN_DIR := $(BUILD_ROOT)/xclbins
RTP_FILE := $(if $(filter layer,$(DESIGN)),rtp_addresses.txt,$(DESIGN)_rtp_addresses.txt)
# The shipped build's RTP addresses are compiled into the engine (FLM_DLL/detail/gemma4e_dflash/gemma4e_dflash_rtp.hpp),
# because CI ships only the .xclbin: a build whose addresses differ from the committed list fails here instead of
# producing an xclbin the engine would drive with wrong addresses.
RTP_EXPECTED := $(if $(and $(filter fused8,$(DESIGN)),$(filter x,$(FUSED8_ATTN)),$(filter full,$(FUSED8_STAGE))),$(wildcard ${HOME_DIR}/rtp_expected/fused8x_$(MODEL_TYPE).txt))


IRON_BOTH_SRC := ${HOME_DIR}/iron/$(DESIGN)_aie.py
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
	mkdir -p $(XCLBIN_DIR)
	cp ${@} $(XCLBIN_DIR)/$(DESIGN).xclbin
	cat ${@D}/$(DESIGN).mlir.prj/input_with_addresses.mlir | grep "%RTP" | grep "aie.buffer" | awk '{print $$1, $$6}' | sed 's/%//g' | tee $(XCLBIN_DIR)/$(RTP_FILE)
	@if [ -n "$(RTP_EXPECTED)" ]; then \
	  if diff -u $(RTP_EXPECTED) $(XCLBIN_DIR)/$(RTP_FILE); then echo "RTP addresses match $(notdir $(RTP_EXPECTED))"; \
	  else echo "ERROR: RTP addresses moved: the gemma4e_dflash engine compiles in $(RTP_EXPECTED); update both together"; rm -f $@ $(XCLBIN_DIR)/$(DESIGN).xclbin; exit 1; fi; \
	fi

XCLBIN_TARGETS += ${IRON_BOTH_XCLBIN_TARGET} 
