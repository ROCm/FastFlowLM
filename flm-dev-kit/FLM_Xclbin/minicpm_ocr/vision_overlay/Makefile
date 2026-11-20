# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# MiniCPM-OCR vision overlay: vision mm (design A) + vision attention in one xclbin. See README.md.
#
#   make            base xclbin + switch tables
#   make bitstream  build/xclbins/vision_overlay.xclbin (what CI builds)
#   make regs       host/op_configs.hpp (each op's configuration, from the standalone designs)
#   make pm         program size of one core (limit 16384)
#   make engine     copy the host headers into FLM_DLL/detail/minicpm_ocr_npu/image/npu
#
# Peano only. Needs the AIE environment (see ../README.md, "Peano").

HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
AIEOPT_DIR ?= $(shell realpath $(dir $(shell which aie-opt))/..)
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf -Wno-parentheses -Wno-attributes \
	-Wno-macro-redefined -DNDEBUG -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -I ${AIEOPT_DIR}/include
AIECC_FLAGS := --no-compile-host --no-xchesscc --no-xbridge

MM_DIR := ${HOME_DIR}/../vision_mm
ATTN_DIR := ${HOME_DIR}/../vision_attn
MM_BUILD := ${MM_DIR}/build/VISION_MM_A
ATTN_BUILD := ${ATTN_DIR}/build
MM_ADDR := ${MM_BUILD}/bitstream/from_iron/mm.mlir.prj/input_with_addresses.mlir
ATTN_ADDR := ${ATTN_BUILD}/bitstream/from_iron/attn.mlir.prj/input_with_addresses.mlir

LAYOUT := build/layout.json
LAYOUT_H := build/kernel/overlay_layout.h
KERNEL_OBJ := build/kernel/fused.o
KERNEL_SRCS := kernel/fused.cc $(wildcard ${MM_DIR}/kernel/*) $(wildcard ${ATTN_DIR}/kernel/*)
XCLBIN := build/xclbins/vision_overlay.xclbin
REGS := host/op_configs.hpp

.PHONY: all standalone kernel bitstream regs pm engine check_engine clean_all
all: ${XCLBIN} ${REGS}
kernel: ${KERNEL_OBJ}
bitstream: ${XCLBIN} check_engine
regs: ${REGS}

# the two standalone designs: their buffer addresses and their configuration are the overlay's
standalone:
	$(MAKE) -C ${MM_DIR} MODEL_TYPE=VISION_MM_A bitstream
	$(MAKE) -C ${ATTN_DIR} bitstream
${MM_ADDR} ${ATTN_ADDR}: standalone

${LAYOUT} ${LAYOUT_H} host/overlay_layout.hpp: tools/gen_layout.py ${MM_ADDR} ${ATTN_ADDR}
	python3 tools/gen_layout.py ${MM_ADDR} ${ATTN_ADDR}

${KERNEL_OBJ}: ${KERNEL_SRCS} ${LAYOUT_H}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DNPU2 -DMODEL_TYPE=VISION_MM_A -I. \
		-c ${HOME_DIR}/kernel/fused.cc -o fused.o

build/mlir/%.mlir: iron/overlay_aie.py ${LAYOUT}
	mkdir -p ${@D}
	python3 iron/overlay_aie.py $* ${LAYOUT} > $@

# every core's arena must start right after the stack, where the standalone designs' buffers start
build/base/vision_overlay.xclbin: build/mlir/base.mlir ${KERNEL_OBJ}
	mkdir -p ${@D}
	cp ${KERNEL_OBJ} ${@D}
	cd ${@D} && aiecc.py ${AIECC_FLAGS} --aie-generate-cdo --aie-generate-xclbin \
		--xclbin-name=${@F} ../mlir/base.mlir > aiecc.log 2>&1
	python3 tools/check_arena.py build/base/base.mlir.prj/input_with_addresses.mlir ${LAYOUT}

${XCLBIN}: build/base/vision_overlay.xclbin
	mkdir -p ${@D}
	cp $< $@

# each op's load-time configuration, as register writes, from its standalone design
build/txn_mm/txn.mlir: ${MM_ADDR}
	rm -rf ${@D} && mkdir -p ${@D}
	cp ${MM_BUILD}/mlir/mm.mlir ${MM_BUILD}/kernel/mm.o ${@D}
	cd ${@D} && aiecc.py ${AIECC_FLAGS} --aie-generate-txn --txn-name=txn.mlir mm.mlir > aiecc.log 2>&1
build/txn_attn/txn.mlir: ${ATTN_ADDR}
	rm -rf ${@D} && mkdir -p ${@D}
	cp ${ATTN_BUILD}/mlir/attn.mlir ${ATTN_BUILD}/kernel/attn.o ${@D}
	cd ${@D} && aiecc.py ${AIECC_FLAGS} --aie-generate-txn --txn-name=txn.mlir attn.mlir > aiecc.log 2>&1

${REGS}: tools/extract_txn2.py build/txn_mm/txn.mlir build/txn_attn/txn.mlir build/base/vision_overlay.xclbin
	python3 tools/extract_txn2.py --base build/base/base.mlir.prj/input_with_addresses.mlir \
		--op mm=build/txn_mm/txn.mlir --op attn=build/txn_attn/txn.mlir -o $@

pm: build/mlir/pm.mlir ${KERNEL_OBJ}
	rm -rf build/pm && mkdir -p build/pm
	cp ${KERNEL_OBJ} build/pm
	-cd build/pm && aiecc.py ${AIECC_FLAGS} --aie-generate-cdo --xclbin-name=pm.xclbin ../mlir/pm.mlir > aiecc.log 2>&1
	@${PEANO_INSTALL_DIR}/bin/llvm-size build/pm/pm.mlir.prj/core_0_2.elf

# the engine keeps its own copies (FLM_DLL builds without this folder)
ENGINE_DIR := ${HOME_DIR}/../../../FLM_DLL/detail/minicpm_ocr_npu/image/npu
engine: ${REGS} host/vision_overlay.hpp host/overlay_layout.hpp
	cp ${REGS} ${ENGINE_DIR}/vision_overlay_configs.hpp
	cp host/overlay_layout.hpp ${ENGINE_DIR}/vision_overlay_layout.hpp
	sed 's#"op_configs.hpp"#"vision_overlay_configs.hpp"#; s#"overlay_layout.hpp"#"vision_overlay_layout.hpp"#' \
		host/vision_overlay.hpp > ${ENGINE_DIR}/vision_overlay.hpp

# CI runs only `make bitstream`: fail if this build's tables differ from the engine's copies
check_engine: ${REGS} host/overlay_layout.hpp
	cmp ${REGS} ${ENGINE_DIR}/vision_overlay_configs.hpp
	cmp host/overlay_layout.hpp ${ENGINE_DIR}/vision_overlay_layout.hpp

clean_all:
	-rm -rf build
