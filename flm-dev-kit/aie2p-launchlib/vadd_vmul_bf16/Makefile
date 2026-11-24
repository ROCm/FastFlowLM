##===- Makefile - vadd_vmul_bf16 on the AIE2P launch library --------------===##
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
##===---------------------------------------------------------------------===##
#
# y = (A + B) * C + D: two programs on the same cores, switched by the launch
# library's kernels (cold, warm, rerun) in one hw_context.
#
# Run everything through ./env.sh (the toolchain on PATH):
#
#   ./env.sh make                  the package and the host for COLS x CORES
#   ./env.sh make run              run on the NPU (all cases, checks, timing)
#   ./env.sh make run N=4096 COLS=4 CORES=2
#   ./env.sh make check            the library's tests on this package (no NPU)
#   ./env.sh make stress           stress and corner-case tests on the NPU (host/stress.cpp)
#   ./env.sh make ovlp             vadd as an overlay with its op as a state (APP@STATE,
#                                  host/ovlp.cpp): back from vmul with the op set up
#   ./env.sh make precompile       refresh precompiled/<config>/ from source
#   make print-config | clean | clean_all
#
# The package comes from precompiled/<config>/ when it is there (no aiecc, no
# Peano needed); USE_PRECOMPILED=0 builds it from source.
#
##===---------------------------------------------------------------------===##

.DEFAULT_GOAL := all
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
LAUNCHLIB := $(realpath $(HOME_DIR)/../launchlib)

MLIR_AIE_DIR ?= $(error MLIR_AIE_DIR is not set: run make through ./env.sh)
PEANO_DIR    ?= $(MLIR_AIE_DIR)/ironenv/lib/python3.12/site-packages/llvm-aie

APPS := vadd vmul

# The per-core block (bf16, a multiple of 16) and the array: COLS columns (1..8)
# of CORES cores (1..4). The launch length N is a run-time choice.
# LAYOUT (iron/): split (1-2 cores) or pair (1-4 cores; the default for 3-4).
EW_N  ?= 256
COLS  ?= 1
CORES ?= 1
LAYOUT ?= $(if $(filter 1 2,$(CORES)),split,pair)

CONFIG    := n$(EW_N)-c$(COLS)x$(CORES)$(if $(filter pair,$(LAYOUT)),-pair)
BUILD_DIR := build/$(CONFIG)
DEFINES   := -DEW_N=$(EW_N) -DEW_COLS=$(COLS) -DEW_CORES=$(CORES)
IRON_ARGS := --n $(EW_N) --cols $(COLS) --cores $(CORES) --layout $(LAYOUT)

# --- kernels: Peano for AIE2P -------------------------------------------------

PEANO_FLAGS := -O2 -std=c++20 --target=aie2p-none-unknown-elf \
               -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -DNDEBUG \
               -I $(MLIR_AIE_DIR)/install/include -I $(HOME_DIR)/kernels $(DEFINES)
KERNEL_OBJS := $(foreach a,$(APPS),$(BUILD_DIR)/kernels/$(a)_bf16.o)

$(BUILD_DIR)/kernels/%.o: $(HOME_DIR)/kernels/%.cc $(HOME_DIR)/kernels/elementwise_config.h
	mkdir -p $(@D)
	$(PEANO_DIR)/bin/clang++ $(PEANO_FLAGS) -c $< -o $@

# --- one aiecc design per app, then the package ---------------------------------

IRON_SRC := $(HOME_DIR)/iron/vadd_vmul_bf16_aie.py
AIECC_FLAGS := --no-xchesscc --no-xbridge --no-compile-host --no-aiesim

.SECONDARY:

$(BUILD_DIR)/%/design.mlir: $(IRON_SRC)
	mkdir -p $(@D)
	python3 $< $(IRON_ARGS) --op $(patsubst v%,%,$*) > $@

# aiecc resolves link_with against its working directory: the objects go next
# to the MLIR. One run: the xclbin, the instructions, the configure sequence.
$(BUILD_DIR)/%/stamp: $(BUILD_DIR)/%/design.mlir $(KERNEL_OBJS)
	cp $(KERNEL_OBJS) $(@D)/
	cd $(@D) && aiecc.py $(AIECC_FLAGS) --aie-generate-xclbin --xclbin-name=$*.xclbin \
		--aie-generate-npu-insts --npu-insts-name=$*_insts.bin \
		--aie-generate-txn --txn-name=$*_txn.mlir design.mlir
	touch $@

PRECOMPILED_DIR := $(HOME_DIR)/precompiled/$(CONFIG)
USE_PRECOMPILED ?= 1
PRECOMPILED_OK  := $(and $(filter 1,$(USE_PRECOMPILED)),$(filter $(words $(APPS)),$(words $(wildcard $(foreach a,$(APPS),$(PRECOMPILED_DIR)/$(a)/init.txn)))))
# A source newer than the precompiled package: it is stale, build from source.
PRECOMPILED_STALE := $(if $(PRECOMPILED_OK),$(firstword $(shell find $(HOME_DIR)/iron $(HOME_DIR)/kernels \
                       $(LAUNCHLIB)/tools/make_package.py -type f -newer $(PRECOMPILED_DIR)/$(firstword $(APPS))/init.txn)))
ifneq ($(PRECOMPILED_STALE),)
$(warning precompiled/$(CONFIG) is older than $(PRECOMPILED_STALE): the package is built from source (make precompile refreshes it))
PRECOMPILED_OK :=
endif

ifneq ($(PRECOMPILED_OK),)
PKG_DIR   := $(PRECOMPILED_DIR)
PKG_STAMP := $(foreach a,$(APPS),$(PKG_DIR)/$(a)/init.txn)
else
PKG_DIR   := $(BUILD_DIR)/pkg
PKG_STAMP := $(PKG_DIR)/.stamp
$(PKG_STAMP): $(foreach a,$(APPS),$(BUILD_DIR)/$(a)/stamp) $(LAUNCHLIB)/tools/make_package.py
	rm -rf $(PKG_DIR)
	python3 $(LAUNCHLIB)/tools/make_package.py -o $(PKG_DIR) $(foreach a,$(APPS),$(a)=$(BUILD_DIR)/$(a))
	touch $@
endif

# --- host -----------------------------------------------------------------------

HOST_TARGET := $(BUILD_DIR)/vadd_vmul_bf16.exe
HOST_CXX    ?= g++-13
HOST_FLAGS  := -std=c++20 -O2 -Wall -I $(LAUNCHLIB)/include -I $(HOME_DIR)/include -I $(HOME_DIR)/kernels $(DEFINES)
HOST_LIBS   := -lxrt_coreutil -luuid
HOST_DEPS   := $(HOME_DIR)/host/host.cpp $(wildcard $(HOME_DIR)/include/*.hpp $(HOME_DIR)/kernels/*.h) \
               $(shell find $(LAUNCHLIB)/include -name '*.hpp')

$(HOST_TARGET): $(HOST_DEPS)
	mkdir -p $(@D)
	$(HOST_CXX) $(HOST_FLAGS) $(HOME_DIR)/host/host.cpp -o $@ $(HOST_LIBS)

STRESS_TARGET := $(BUILD_DIR)/stress.exe
$(STRESS_TARGET): $(HOME_DIR)/host/stress.cpp $(filter-out $(HOME_DIR)/host/host.cpp,$(HOST_DEPS))
	mkdir -p $(@D)
	$(HOST_CXX) $(HOST_FLAGS) $(HOME_DIR)/host/stress.cpp -o $@ $(HOST_LIBS)

# APP@STATE: vadd split into the overlay ovl and the state add (make_overlay.exe),
# next to a copy of the package.
OVL_TOOL    := $(BUILD_DIR)/make_overlay.exe
OVLP_TARGET := $(BUILD_DIR)/ovlp.exe
OVL_PKG     := $(BUILD_DIR)/pkg_ovl
$(OVL_TOOL): $(HOME_DIR)/host/make_overlay.cpp $(filter-out $(HOME_DIR)/host/host.cpp,$(HOST_DEPS))
	mkdir -p $(@D)
	$(HOST_CXX) $(HOST_FLAGS) $(HOME_DIR)/host/make_overlay.cpp -o $@
$(OVLP_TARGET): $(HOME_DIR)/host/ovlp.cpp $(filter-out $(HOME_DIR)/host/host.cpp,$(HOST_DEPS))
	mkdir -p $(@D)
	$(HOST_CXX) $(HOST_FLAGS) $(HOME_DIR)/host/ovlp.cpp -o $@ $(HOST_LIBS)
$(OVL_PKG)/.stamp: $(PKG_STAMP) $(OVL_TOOL)
	rm -rf $(OVL_PKG)
	mkdir -p $(OVL_PKG)
	cp -r $(foreach a,$(APPS),$(PKG_DIR)/$(a)) $(OVL_PKG)/
	$(OVL_TOOL) $(OVL_PKG) $(BUILD_DIR)/add.txn
	python3 $(LAUNCHLIB)/tools/make_package.py -o $(OVL_PKG) --state ovl@add=$(BUILD_DIR)/add.txn
	touch $@

# --- targets ----------------------------------------------------------------------

N     ?= $(shell echo $$(( 2 * $(EW_N) * $(COLS) * $(CORES) )))
ITERS ?= 50
PAIRS ?= 16
RUN_ARGS ?=

.PHONY: all kernel bitstream package host run stress ovlp check precompile clean clean_all print-config

all: $(PKG_STAMP) $(HOST_TARGET)
kernel: $(KERNEL_OBJS)
bitstream: $(foreach a,$(APPS),$(BUILD_DIR)/$(a)/stamp)
package: $(PKG_STAMP)
host: $(HOST_TARGET)

run: all
	$(HOST_TARGET) --package $(PKG_DIR) --n $(N) --iters $(ITERS) --pairs $(PAIRS) $(RUN_ARGS)

# Stress and corner cases (host/stress.cpp): every mode, then the context test.
SEED     ?= 1
LAUNCHES ?= 20000
stress: $(PKG_STAMP) $(STRESS_TARGET)
	$(STRESS_TARGET) --package $(PKG_DIR) --mode all --seed $(SEED) --launches $(LAUNCHES)
	$(STRESS_TARGET) --package $(PKG_DIR) --mode ctx --seed $(SEED)

ovlp: $(OVL_PKG)/.stamp $(OVLP_TARGET)
	$(OVLP_TARGET) --package $(OVL_PKG) --n $(N) --iters $(ITERS) --pairs $(PAIRS) $(RUN_ARGS)

# The library's tests on this package: the C++ sequence equals aiecc's, and the
# kernels configure what aiecc configures (register model). No NPU.
check: $(PKG_STAMP)
	$(MAKE) -C $(LAUNCHLIB) build/test_rt build/test_build
	$(LAUNCHLIB)/build/test_rt $(PKG_DIR) $(APPS)
	$(LAUNCHLIB)/build/test_build $(PKG_DIR) $(APPS)

precompile:
	$(MAKE) USE_PRECOMPILED=0 package
	rm -rf $(PRECOMPILED_DIR)
	mkdir -p $(PRECOMPILED_DIR)
	cp -r $(BUILD_DIR)/pkg/. $(PRECOMPILED_DIR)/
	rm -f $(PRECOMPILED_DIR)/.stamp

print-config:
	@echo "config    : $(CONFIG)   $(COLS) cols x $(CORES) cores, $(EW_N) bf16 per block, layout $(LAYOUT)"
	@echo "package   : $(PKG_DIR)"
	@echo "launchlib : $(LAUNCHLIB)"
	@echo "mlir-aie  : $(MLIR_AIE_DIR)"

clean:
	-rm -rf $(BUILD_DIR)

clean_all:
	-rm -rf build
