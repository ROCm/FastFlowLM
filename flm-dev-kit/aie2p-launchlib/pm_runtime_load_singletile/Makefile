##===- Makefile - the fp32 designs and the host test of the launch library -===##
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
# The AIE2P counterpart of aie4-launchlib/pm_runtime_load_singletile: three
# designs with three different floorplans (vadd, vmul, vminus), one package,
# and the host test that switches between them (host_test/).
#
#   ./env.sh make package          build each design (aiecc, Peano) -> the package
#   ./env.sh make check            no NPU: every init write has a class, the C++
#                                  sequences equal aiecc's, the kernels on a
#                                  register model configure what aiecc configures
#   ./env.sh make test             the host test on the NPU (APPS=, CONFIG=, ITERS=)
#   ./env.sh make stress           the random walk over the three apps (host/stress)
#   ./env.sh make precompile       refresh precompiled/c<COLS>/
#   make clean
#
# COLS (1..8) columns of 4 cores; every design uses all of them.

.DEFAULT_GOAL := all
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
LAUNCHLIB := $(realpath $(HOME_DIR)/../launchlib)

MLIR_AIE_DIR ?= $(error MLIR_AIE_DIR is not set: run make through ./env.sh)
PEANO_DIR    ?= $(MLIR_AIE_DIR)/ironenv/lib/python3.12/site-packages/llvm-aie

DESIGNS ?= vadd vmul vminus
COLS    ?= 4
BUILD_DIR := build/c$(COLS)

PEANO_FLAGS := -O2 -std=c++20 --target=aie2p-none-unknown-elf \
               -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -DNDEBUG \
               -I $(MLIR_AIE_DIR)/install/include -DFP_N=256 -DFP_TILES=4
AIECC_FLAGS := --no-xchesscc --no-xbridge --no-compile-host --no-aiesim

.SECONDARY:

$(BUILD_DIR)/%/kernel.o: $(HOME_DIR)/%_fp32/kernel.cc
	mkdir -p $(@D)
	$(PEANO_DIR)/bin/clang++ $(PEANO_FLAGS) -c $< -o $(@D)/$*_fp32.o
	touch $@

# One rule per design (make substitutes only one % in a prerequisite).
define DESIGN_MLIR
$(BUILD_DIR)/$(1)/design.mlir: $(HOME_DIR)/$(1)_fp32/$(1)_fp32_placed.py
	mkdir -p $$(@D)
	python3 $$< --cols $(COLS) > $$@
endef
$(foreach d,$(DESIGNS),$(eval $(call DESIGN_MLIR,$(d))))

# One aiecc run per design: the xclbin, the instructions, the configure sequence.
$(BUILD_DIR)/%/stamp: $(BUILD_DIR)/%/design.mlir $(BUILD_DIR)/%/kernel.o
	cd $(@D) && aiecc.py $(AIECC_FLAGS) --aie-generate-xclbin --xclbin-name=$*.xclbin \
		--aie-generate-npu-insts --npu-insts-name=$*_insts.bin \
		--aie-generate-txn --txn-name=$*_txn.mlir design.mlir
	touch $@

PRECOMPILED_DIR := $(HOME_DIR)/precompiled/c$(COLS)
USE_PRECOMPILED ?= 1
PRECOMPILED_OK  := $(and $(filter 1,$(USE_PRECOMPILED)),$(filter $(words $(DESIGNS)),$(words $(wildcard $(foreach d,$(DESIGNS),$(PRECOMPILED_DIR)/$(d)/init.txn)))))
# A source newer than the precompiled package: it is stale, build from source.
PRECOMPILED_STALE := $(if $(PRECOMPILED_OK),$(firstword $(shell find $(foreach d,$(DESIGNS),$(HOME_DIR)/$(d)_fp32) \
                       $(LAUNCHLIB)/tools/make_package.py -type f -newer $(PRECOMPILED_DIR)/$(firstword $(DESIGNS))/init.txn)))
ifneq ($(PRECOMPILED_STALE),)
$(warning precompiled/c$(COLS) is older than $(PRECOMPILED_STALE): the package is built from source (make precompile refreshes it))
PRECOMPILED_OK :=
endif
ifneq ($(PRECOMPILED_OK),)
PKG_DIR   := $(PRECOMPILED_DIR)
PKG_STAMP := $(foreach d,$(DESIGNS),$(PKG_DIR)/$(d)/init.txn)
else
PKG_DIR   := $(BUILD_DIR)/pkg
PKG_STAMP := $(PKG_DIR)/.stamp
$(PKG_STAMP): $(foreach d,$(DESIGNS),$(BUILD_DIR)/$(d)/stamp) $(LAUNCHLIB)/tools/make_package.py
	rm -rf $(PKG_DIR)
	python3 $(LAUNCHLIB)/tools/make_package.py -o $(PKG_DIR) $(foreach d,$(DESIGNS),$(d)=$(BUILD_DIR)/$(d))
	touch $@
endif

# --- host ------------------------------------------------------------------------

HOST_CXX   ?= g++-13
HOST_FLAGS := -std=c++20 -O2 -Wall -I $(LAUNCHLIB)/include -I $(HOME_DIR)/host_test -I $(HOME_DIR)/../vadd_vmul_bf16/include \
              -DCOLS=$(COLS)
HOST_LIBS  := -lxrt_coreutil -luuid
HOST_DEPS  := $(wildcard $(HOME_DIR)/host_test/*.hpp $(HOME_DIR)/host_test/designs/*.hpp) \
              $(shell find $(LAUNCHLIB)/include -name '*.hpp')

$(BUILD_DIR)/%.exe: $(HOME_DIR)/host_test/%.cpp $(HOST_DEPS)
	mkdir -p $(@D)
	$(HOST_CXX) $(HOST_FLAGS) $< -o $@ $(HOST_LIBS)

$(BUILD_DIR)/check_sequence.exe: $(HOME_DIR)/host_test/checks/check_sequence.cpp $(HOST_DEPS)
	mkdir -p $(@D)
	$(HOST_CXX) $(HOST_FLAGS) $< -o $@

# --- targets -----------------------------------------------------------------------

APPS   ?= vadd,vmul,vminus
CONFIG ?=
ITERS  ?= 100
SEED   ?= 1
LAUNCHES ?= 20000
TEST_ARGS = --package $(PKG_DIR) --iters $(ITERS) $(if $(CONFIG),--config $(CONFIG),--apps $(APPS))

.PHONY: all package host test check stress precompile clean print-config

all: $(PKG_STAMP) $(BUILD_DIR)/test.exe $(BUILD_DIR)/stress.exe
package: $(PKG_STAMP)
host: $(BUILD_DIR)/test.exe $(BUILD_DIR)/stress.exe

test: all
	$(BUILD_DIR)/test.exe $(TEST_ARGS)

# No NPU: every init write has a register class and the binaries round-trip
# (test_txn); each C++ sequence equals aiecc's (check_sequence); the kernels
# configure what aiecc configures, between any two designs (test_build).
check: $(PKG_STAMP) $(BUILD_DIR)/check_sequence.exe
	$(MAKE) -C $(LAUNCHLIB) build/test_txn build/test_build
	$(LAUNCHLIB)/build/test_txn $(foreach d,$(DESIGNS),$(PKG_DIR)/$(d)/init.txn $(PKG_DIR)/$(d)/insts.bin)
	$(BUILD_DIR)/check_sequence.exe $(PKG_DIR) $(DESIGNS)
	$(foreach a,$(DESIGNS),$(foreach b,$(filter-out $(a),$(DESIGNS)),$(LAUNCHLIB)/build/test_build $(PKG_DIR) $(a) $(b) &&)) true

stress: all
	$(BUILD_DIR)/stress.exe --package $(PKG_DIR) --seed $(SEED) --launches $(LAUNCHES)

precompile:
	$(MAKE) USE_PRECOMPILED=0 package
	rm -rf $(PRECOMPILED_DIR)
	mkdir -p $(PRECOMPILED_DIR)
	cp -r $(BUILD_DIR)/pkg/. $(PRECOMPILED_DIR)/
	rm -f $(PRECOMPILED_DIR)/.stamp

print-config:
	@echo "designs   : $(DESIGNS), $(COLS) columns x 4 cores"
	@echo "package   : $(PKG_DIR)"

clean:
	-rm -rf $(BUILD_DIR)
