# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Hy-MT2 prefill overlay: dequant_mm + attention in one xclbin, taking turns on the array.
#
#   make            kernel objects, base xclbin, txn dumps of both ops, op table, run.exe
#   make bitstream  build/xclbins/prefill_overlay.xclbin
#   make regs       the engine's hunyuan_overlay_configs.hpp (each op's configuration, from
#                   aiecc --aie-generate-txn), written straight into FLM_DLL/detail/hunyuan_npu
#   make host       run.exe
#   make probe      probe_rows.exe (do rows the caller ignores change the others? see README.md)
#
# Needs the mlir-aie kernel environment (see ../README.md, Build notes). Peano by default;
# ENABLE_CHESSCC=1 builds the same design with chess. See README.md.

HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
ENABLE_CHESSCC ?= 0
AIETOOLS_DIR ?= $(AIETOOLS_ROOT)
AIEOPT_DIR ?= $(shell realpath $(dir $(shell which aie-opt))/..)

WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined
CHESSCCWRAP2P_FLAGS = aie2p -I ${AIETOOLS_DIR}/include -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
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
XCLBIN := build/xclbins/prefill_overlay.xclbin
FLM_DLL_DIR := ../../../FLM_DLL
ENGINE_DIR := $(FLM_DLL_DIR)/detail/hunyuan_npu
REGS := $(ENGINE_DIR)/hunyuan_overlay_configs.hpp
BASE_ADDR := build/base/base.mlir.prj/input_with_addresses.mlir

.PHONY: all kernel bitstream regs host probe clean_all
all: ${XCLBIN} ${REGS} run.exe
kernel: ${KERNEL_OBJS}
bitstream: ${XCLBIN}
regs: ${REGS}
host: run.exe
probe: probe_rows.exe

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
	@# the host writes every core's RTP block at layout.RTP_ADDR
	@python3 -c "import re,sys; sys.path.insert(0,'iron'); import layout; \
	a=set(re.findall(r'\{address = ([0-9]+) : i32[^}]*sym_name = \"rtp_', open('${BASE_ADDR}').read())); \
	assert a == {str(layout.RTP_ADDR)}, f'RTP blocks at {a}, host expects {layout.RTP_ADDR}'; print('RTP block address OK:', layout.RTP_ADDR)"
ifneq ($(ENABLE_CHESSCC),1)
	@python3 tools/check_stack.py ${PEANO_INSTALL_DIR}/bin/llvm-objdump 4096 build/base/base.mlir.prj/*.elf
endif
${BASE_ADDR}: build/base/overlay.xclbin

${XCLBIN}: build/base/overlay.xclbin
	mkdir -p ${@D}
	cp $< $@

# each op's load-time configuration as register writes
build/%/txn.mlir: build/mlir/%.mlir ${KERNEL_OBJS}
	mkdir -p ${@D}
	cp ${KERNEL_OBJS} ${@D}
	cd ${@D} && aiecc.py ${AIECC_FLAGS} --aie-generate-txn --txn-name=txn.mlir ../mlir/$*.mlir > aiecc.log 2>&1

${REGS}: tools/extract_txn2.py build/dq/txn.mlir build/attn/txn.mlir build/dq32/txn.mlir build/dq64/txn.mlir ${BASE_ADDR}
	python3 tools/extract_txn2.py --base ${BASE_ADDR} \
		--op dq=build/dq/txn.mlir --op attn=build/attn/txn.mlir \
		--op dq32=build/dq32/txn.mlir --op dq64=build/dq64/txn.mlir --namespace hunyuan_overlay -o $@

# ---- host: built against FLM_DLL, so the CPU references and the run generators
#      (hunyuan_overlay.hpp) are the engine's own ----
CXX := g++-13
CXX_FLAGS := -std=c++20 -O2 -fPIC -Wall -Wno-sign-compare -Wno-unused-variable -fopenmp
CXX_FLAGS += -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -march=native
CXX_FLAGS += -Ihost -I$(FLM_DLL_DIR)/include -I$(FLM_DLL_DIR)/detail/hunyuan_npu -I/opt/xilinx/xrt/include -MMD -MP
LDFLAGS := -L/opt/xilinx/xrt/lib -Wl,-rpath,/opt/xilinx/xrt/lib -lxrt_coreutil $(FLM_DLL_DIR)/external_libs/libaiebu.a
LDFLAGS += -lboost_program_options -lboost_filesystem -fopenmp

run.exe: build/host/host.o
	$(CXX) $(CXX_FLAGS) $^ -o $@ $(LDFLAGS)

build/host/host.o: host/host.cpp ${REGS} $(ENGINE_DIR)/hunyuan_overlay.hpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

probe_rows.exe: build/host/probe_rows.o
	$(CXX) $(CXX_FLAGS) $^ -o $@ $(LDFLAGS)

build/host/probe_rows.o: host/probe_rows.cpp ${REGS} $(ENGINE_DIR)/hunyuan_overlay.hpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

-include build/host/host.d
-include build/host/probe_rows.d

clean_all:
	-rm -rf build run.exe probe_rows.exe
