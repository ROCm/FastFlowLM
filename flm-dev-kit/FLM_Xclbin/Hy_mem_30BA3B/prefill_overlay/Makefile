# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Hy-Memory prefill overlay: dequant_mm + attention in one xclbin (see iron/overlay_aie.py).
#
#   make            kernel objects, base xclbin, txn dumps of both ops, op table, run.exe
#   make bitstream  build/xclbins/overlay.xclbin
#   make regs       host/op_configs.hpp (each op's configuration, from aiecc --aie-generate-txn)
#   make host       run.exe
#
# Needs the kernel env (AIETOOLS_ROOT, mlir-aie, PEANO_INSTALL_DIR). Peano by default;
# ENABLE_CHESSCC=1 builds the same design with chess. Run make clean_all when switching.

HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
ENABLE_CHESSCC ?= 0
export ENABLE_CHESSCC
AIETOOLS_DIR ?= $(AIETOOLS_ROOT)
# Expanded only inside the Chess recipes, so it fires only for ENABLE_CHESSCC=1.
AIETOOLS_CHECK = $(if $(strip $(AIETOOLS_DIR)),,$(error ENABLE_CHESSCC=1 needs Vitis AIE Essentials: set AIETOOLS_ROOT (or AIETOOLS_DIR), or put xchesscc on PATH))
AIEOPT_DIR ?= $(shell realpath $(dir $(shell which aie-opt))/..)

WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined
CHESSCCWRAP2P_FLAGS = aie2p $(AIETOOLS_CHECK)-I ${AIETOOLS_DIR}/include -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG \
	-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -I ${AIEOPT_DIR}/include
KERNEL_INCLUDES = -DNPU2 -I. -I${HOME_DIR}/kernel -I${HOME_DIR}/kernel/dq -I${HOME_DIR}/kernel/attn

ifeq ($(ENABLE_CHESSCC),1)
AIECC_FLAGS := --no-compile-host --xchesscc --xbridge
else
AIECC_FLAGS := --no-compile-host --no-xchesscc --no-xbridge
endif

LAYOUT_H := build/kernel/overlay_layout.h
KERNEL_OBJS := build/kernel/fused_e.o build/kernel/fused_o.o
KERNEL_SRCS := $(wildcard kernel/*.h kernel/*.cc kernel/dq/* kernel/attn/*)
IRON_SRC := $(wildcard iron/*.py)
XCLBIN := build/xclbins/overlay.xclbin
REGS := host/op_configs.hpp

.PHONY: all kernel bitstream regs host clean_all
all: ${XCLBIN} ${REGS} run.exe
kernel: ${KERNEL_OBJS}
bitstream: ${XCLBIN}
regs: ${REGS}
host: run.exe

${LAYOUT_H}: iron/layout.py
	mkdir -p ${@D}
	python3 $< > $@

build/kernel/%.o: kernel/%.cc ${KERNEL_SRCS} ${LAYOUT_H}
ifeq ($(ENABLE_CHESSCC),1)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} ${KERNEL_INCLUDES} -c ${HOME_DIR}/$< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${KERNEL_INCLUDES} -c ${HOME_DIR}/$< -o ${@F}
endif

.PRECIOUS: build/mlir/%.mlir
build/mlir/%.mlir: ${IRON_SRC}
	mkdir -p ${@D}
	cd iron && python3 overlay_aie.py $* > ${HOME_DIR}/$@

# the xclbin that runs: core programs + arenas, no routes or chains
build/base/overlay.xclbin: build/mlir/base.mlir ${KERNEL_OBJS}
	mkdir -p ${@D}
	cp ${KERNEL_OBJS} ${@D}
	cd ${@D} && aiecc.py ${AIECC_FLAGS} --aie-generate-cdo --aie-generate-xclbin \
		--xclbin-name=${@F} ../mlir/base.mlir > aiecc.log 2>&1
ifneq ($(ENABLE_CHESSCC),1)
	@python3 tools/check_stack.py ${PEANO_INSTALL_DIR}/bin/llvm-objdump 4096 build/base/base.mlir.prj/*.elf
endif
build/base/base.mlir.prj/input_with_addresses.mlir: build/base/overlay.xclbin

${XCLBIN}: build/base/overlay.xclbin
	mkdir -p ${@D}
	cp $< $@

# each op's load-time configuration as register writes
build/%/txn.mlir: build/mlir/%.mlir ${KERNEL_OBJS}
	mkdir -p ${@D}
	cp ${KERNEL_OBJS} ${@D}
	cd ${@D} && aiecc.py ${AIECC_FLAGS} --aie-generate-txn --txn-name=txn.mlir ../mlir/$*.mlir > aiecc.log 2>&1

${REGS}: tools/extract_txn2.py build/dq/txn.mlir build/attn/txn.mlir build/base/base.mlir.prj/input_with_addresses.mlir
	python3 tools/extract_txn2.py --base build/base/base.mlir.prj/input_with_addresses.mlir \
		--op dq=build/dq/txn.mlir --op attn=build/attn/txn.mlir -o $@

# ---- host ---------------------------------------------------------------------
HOST_COMMON_DIR := ../../host_common
CXX := g++-13
CXX_FLAGS := -std=c++20 -O2 -fPIC -Wall -Wno-sign-compare -Wno-unused-variable -march=native -fopenmp
CXX_FLAGS += -Ihost -I$(HOST_COMMON_DIR) -I../../dequant_mm_512x512x512/host -I/opt/xilinx/xrt/include -MMD -MP
LDFLAGS := -L/opt/xilinx/xrt/lib -L$(HOST_COMMON_DIR) -lxrt_coreutil $(HOST_COMMON_DIR)/../../../src/lib/xrt/libaiebu.a
LDFLAGS += -lboost_program_options -lboost_filesystem -fopenmp

run.exe: build/host/host.o
	$(CXX) $(CXX_FLAGS) $^ -o $@ $(LDFLAGS)

build/host/host.o: host/host.cpp ${REGS} $(wildcard host/*.hpp)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

-include build/host/host.d

clean_all:
	-rm -rf build run.exe ${REGS}
