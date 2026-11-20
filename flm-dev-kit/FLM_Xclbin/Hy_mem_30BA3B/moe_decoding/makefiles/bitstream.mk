# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# MLIR stuff

MLIR_O_DIR := $(BUILD_DIR)/mlir
BITSTREAM_O_DIR := $(BUILD_DIR)/bitstream


IRON_BOTH_SRC := $(wildcard ${HOME_DIR}/iron/*_aie.py)
IRON_OTHER_SRC := $(wildcard ${HOME_DIR}/iron/*.py)
IRON_OTHER_SRC += $(wildcard ${HOME_DIR}/iron/models/*.py)
IRON_BOTH_MLIR_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${MLIR_O_DIR}/%.mlir, ${IRON_BOTH_SRC})
IRON_BOTH_XCLBIN_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${BITSTREAM_O_DIR}/from_iron/%.xclbin, ${IRON_BOTH_SRC})

.PRECIOUS: ${IRON_BOTH_MLIR_TARGET} 

AIECC_FLAGS := --aie-generate-cdo --no-compile-host
# ENABLE_CHESSCC is set by the top-level Makefile.
ifeq ($(ENABLE_CHESSCC),1)
AIECC_FLAGS += --xchesscc --xbridge
else
AIECC_FLAGS += --no-xchesscc --no-xbridge
endif
# iron generate mlir
${MLIR_O_DIR}/%.mlir: ${HOME_DIR}/iron/%_aie.py ${IRON_OTHER_SRC}
	mkdir -p ${@D}
	ENABLE_CHESSCC=$(ENABLE_CHESSCC) python3 $< ${IRON_ARGS} > $@

# Routing check (deadlock-prone arbiter sharing, misdelivered flows), run in ${@D}.
ifeq ($(CHECK_ROUTES),0)
CHECK_ROUTES_CMD = echo "check_routes: SKIPPED (CHECK_ROUTES=0; experiments only)"
else
CHECK_ROUTES_CMD = python3 ${HOME_DIR}/check_routes.py $(notdir $<).prj/input_physical.mlir ../../mlir/$(notdir $<)
endif
# aiecc's router gives a different routing from run to run (its choices depend on
# memory addresses, which Linux randomizes), so a failed check is retried with a
# fresh routing, up to ROUTE_TRIES times. A failed try's xclbin is deleted so it
# is never used or seen as up to date.
ROUTE_TRIES ?= 5

# Build xclbin
${BITSTREAM_O_DIR}/from_iron/%.xclbin: ${MLIR_O_DIR}/%.mlir ${KERNEL_OBJS}
	mkdir -p ${@D}
	cp ${KERNEL_OBJS} ${@D}
	cd ${@D} && for try in $$(seq 1 $(ROUTE_TRIES)); do \
		aiecc.py ${AIECC_FLAGS} --aie-generate-xclbin \
			--xclbin-name=${@F} $(<:${MLIR_O_DIR}/%=../../mlir/%) || exit 1; \
		if $(CHECK_ROUTES_CMD); then break; fi; \
		rm -f ${@F}; \
		if [ $$try -eq $(ROUTE_TRIES) ]; then echo "check_routes: failed all $(ROUTE_TRIES) tries"; exit 1; fi; \
		echo "check_routes: try $$try failed, routing again"; \
	done
	mkdir -p $(BUILD_DIR)/xclbins
	cp ${@} $(BUILD_DIR)/xclbins/layer.xclbin
	cat ${@D}/*/input_with_addresses.mlir | grep "%RTP" | grep "aie.buffer" | awk '{print $$1, $$6}' | sed 's/%//g' | tee $(BUILD_DIR)/xclbins/rtp_addresses.txt
ifneq ($(ENABLE_CHESSCC),1)
	@${HOME_DIR}/makefiles/check_peano_stack.sh ${@D}/*/*.elf
endif

XCLBIN_TARGETS += ${IRON_BOTH_XCLBIN_TARGET} 
