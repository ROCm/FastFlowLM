# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# MLIR stuff

MLIR_O_DIR := build/$(BUILD_NAME)/mlir
BITSTREAM_O_DIR := build/$(BUILD_NAME)/bitstream


IRON_BOTH_SRC := $(wildcard ${HOME_DIR}/iron/*_aie.py)
IRON_OTHER_SRC := $(wildcard ${HOME_DIR}/iron/*.py)
IRON_BOTH_MLIR_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${MLIR_O_DIR}/%.mlir, ${IRON_BOTH_SRC})
IRON_BOTH_XCLBIN_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${BITSTREAM_O_DIR}/from_iron/%.xclbin, ${IRON_BOTH_SRC})
IRON_BOTH_INSTS_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${BITSTREAM_O_DIR}/from_iron/%.txt, ${IRON_BOTH_SRC})
IRON_BOTH_TXN_TARGET := $(patsubst ${HOME_DIR}/iron/%_aie.py, ${BITSTREAM_O_DIR}/txn/%_txn.mlir, ${IRON_BOTH_SRC})

.PRECIOUS: ${IRON_BOTH_MLIR_TARGET}

AIECC_FLAGS := --aie-generate-cdo --no-compile-host
ENABLE_CHESSCC ?= 0
ifeq ($(ENABLE_CHESSCC),1)
AIECC_FLAGS += --xchesscc --xbridge
else
AIECC_FLAGS += --no-xchesscc --no-xbridge
endif

# RTP addresses FLM_DLL/detail/gemma_npu/gemma_npu_sequence_detail.hpp writes to.
RTP_PROJ_ADDR := 33280
RTP_RMS_ADDR := 54272
RTP_QK_ADDR := 9216
RTP_KV_ADDR := 33792
RTP_COUNT := 25
RTP_NAMES := IS_ATTN_PROJ_BUFFER_[0-9_]+|IS_ATTN_RMS_BUFFERS|L_attn_(qk|kv)_core_[0-9_]+
PROGRAM_MEMORY_BYTES := 16384

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
	mkdir -p build/$(BUILD_NAME)/xclbins
ifneq ($(ENABLE_CHESSCC),1)
	@grep -hoE 'address = [0-9]+ : i32, mem_bank = [0-9]+ : i32, sym_name = "($(RTP_NAMES))"' ${@D}/*/input_with_addresses.mlir \
	  | sed -E 's/address = ([0-9]+) .*sym_name = "([^"]+)"/\2 \1/' | sort > build/$(BUILD_NAME)/xclbins/rtp_addresses.txt
	@awk '{ n++; \
	        if ($$1 ~ /^IS_ATTN_PROJ/) want = $(RTP_PROJ_ADDR); \
	        else if ($$1 == "IS_ATTN_RMS_BUFFERS") want = $(RTP_RMS_ADDR); \
	        else if ($$1 ~ /^L_attn_qk/) want = $(RTP_QK_ADDR); \
	        else want = $(RTP_KV_ADDR); } \
	      $$2 != want { printf "error: %s is at %s, FLM_DLL expects %s\n", $$1, $$2, want; bad = 1 } \
	      END { if (n != $(RTP_COUNT)) { printf "error: %d RTP buffers found, expected $(RTP_COUNT)\n", n; bad = 1 } exit bad }' \
	  build/$(BUILD_NAME)/xclbins/rtp_addresses.txt
	@for e in ${@D}/*/core_*.elf; do \
	  s=$$(${PEANO_INSTALL_DIR}/bin/llvm-size -A $$e | awk '$$1 == ".text" { print $$2 }'); \
	  echo "program $$(basename $$e) $$s bytes"; \
	  if [ "$$s" -gt $(PROGRAM_MEMORY_BYTES) ]; then echo "error: $$e program is over $(PROGRAM_MEMORY_BYTES) bytes"; exit 1; fi; \
	done
endif
	cp ${@} build/$(BUILD_NAME)/xclbins/layer_lm.xclbin

${BITSTREAM_O_DIR}/from_iron/%.txt: ${MLIR_O_DIR}/%.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -n ${AIECC_FLAGS} --aie-generate-npu-insts \
		--npu-insts-name=${@F} $(<:${MLIR_O_DIR}/%=../../mlir/%)
	mkdir -p build/$(BUILD_NAME)/insts
	cp ${@} build/$(BUILD_NAME)/insts/

# Kernel contract check only: the same design also builds for the aie2p-launchlib flow.
${BITSTREAM_O_DIR}/txn/%_txn.mlir: ${MLIR_O_DIR}/%.mlir ${KERNEL_OBJS}
	mkdir -p ${@D}
	cp ${KERNEL_OBJS} ${@D}
	cd ${@D} && aiecc.py ${AIECC_FLAGS} --aie-generate-txn --txn-name=${@F} \
		$(<:${MLIR_O_DIR}/%=../../mlir/%)

INSTS_TARGETS += ${IRON_BOTH_INSTS_TARGET}
XCLBIN_TARGETS += ${IRON_BOTH_XCLBIN_TARGET}
TXN_TARGETS += ${IRON_BOTH_TXN_TARGET}
