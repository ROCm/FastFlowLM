# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# MLIR stuff

MLIR_O_DIR := build/$(BUILD_NAME)/mlir
BITSTREAM_O_DIR := build/$(BUILD_NAME)/bitstream


IRON_BOTH_SRC := $(wildcard ${HOME_DIR}/iron/*_aie.py)
IRON_OTHER_SRC := $(wildcard ${HOME_DIR}/iron/*.py)
IRON_OTHER_SRC += $(wildcard ${HOME_DIR}/iron/models/*.py)
IRON_BOTH_MLIR_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${MLIR_O_DIR}/%.mlir, ${IRON_BOTH_SRC})
IRON_BOTH_XCLBIN_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${BITSTREAM_O_DIR}/from_iron/%.xclbin, ${IRON_BOTH_SRC})

.PRECIOUS: ${IRON_BOTH_MLIR_TARGET} 

AIECC_FLAGS := --aie-generate-cdo --no-compile-host
# ENABLE_CHESSCC is set by the top-level Makefile (Peano by default).
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
	mkdir -p build/$(BUILD_NAME)/xclbins
	cp ${@} build/$(BUILD_NAME)/xclbins/layer.xclbin
	cat ${@D}/*/input_with_addresses.mlir | grep "%RTP" | grep "aie.buffer" | awk '{print $$1, $$6}' | sed 's/%//g' | tee build/$(BUILD_NAME)/xclbins/rtp_addresses.txt
	@# The kernels reach into these buffers with 512-bit vector ops, which aie_api
	@# requires to be 64-byte aligned on AIE2P -- and the hardware truncates a
	@# misaligned access down to the boundary below instead of faulting, so a
	@# buffer that drifts off 64 bytes corrupts data with no diagnostic at all.
	@# Fail here rather than in the logits.  See kernels/aie_array_layout.h.
	@# gdl_rms_state (iron/rms_residual.py) holds two int32 flags that are only
	@# accessed as scalars, so it is exempt.
	@grep -hoE 'address = [0-9]+ : i32.*sym_name = "[^"]+"' ${@D}/*/input_with_addresses.mlir \
	  | sed -E 's/address = ([0-9]+).*sym_name = "([^"]+)"/\1 \2/' \
	  | awk '$$2 != "gdl_rms_state" && $$1 % 64 != 0 { printf "error: buffer %s at address %s is not 64-byte aligned\n", $$2, $$1; bad = 1 } \
	         END { if (bad) exit 1 }' 
ifneq ($(ENABLE_CHESSCC),1)
	@${HOME_DIR}/makefiles/check_peano_stack.sh ${@D}/*/*.elf
endif

XCLBIN_TARGETS += ${IRON_BOTH_XCLBIN_TARGET} 
