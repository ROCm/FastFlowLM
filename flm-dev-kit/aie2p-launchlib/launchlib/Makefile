##===- Makefile - the host tests of the launch library --------------------===##
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
#   make test            unit tests, and the package tests on PKG (no NPU)
#   make test PKG=dir    another package (two apps of one floorplan: APPS)
#   make test_elf        txn::to_elf against aiebu on PKG (needs libaiebu.a of XRT)
#   make lane_report     the control-packet lanes of every app of PKG
#
# PKG defaults to the example's 16-core package, made by
#   cd ../vadd_vmul_bf16 && ./env.sh make precompile COLS=8 CORES=2

CXX   ?= g++-13
FLAGS := -std=c++20 -O1 -Wall -I include
HDRS  := $(shell find include -name '*.hpp') tests/check.hpp

PKG  ?= ../vadd_vmul_bf16/precompiled/n256-c8x2
APPS ?= vadd vmul

build/%: tests/%.cpp $(HDRS)
	mkdir -p build
	$(CXX) $(FLAGS) $< -o $@

build/test_elf: tests/test_elf.cpp $(HDRS)
	mkdir -p build
	$(CXX) $(FLAGS) $< -o $@ -laiebu -luuid

test_elf: build/test_elf
	./build/test_elf $(PKG) $(APPS)

# On the NPU (XRT): what one TXN op costs (tools/op_cost.cpp; launch::CostModel).
# HOST_CXX: the compiler XRT's libraries were built against (as the examples).
HOST_CXX ?= g++-13
build/op_cost.exe: tools/op_cost.cpp $(HDRS)
	mkdir -p build
	$(HOST_CXX) -std=c++20 -O2 -Wall -I include $< -o $@ -lxrt_coreutil -luuid

op_cost: build/op_cost.exe
	./build/op_cost.exe --package $(PKG) --app $(firstword $(APPS))

# The control-packet lanes of every app of PKG (tools/lane_report.cpp; no NPU).
build/lane_report: tools/lane_report.cpp $(HDRS)
	mkdir -p build
	$(CXX) $(FLAGS) $< -o $@

lane_report: build/lane_report
	./build/lane_report $(PKG)

.PHONY: test test_elf clean op_cost lane_report
test: build/test_txn build/test_rt build/test_build build/test_errors build/test_ctrlpkt build/test_state
	@test -f $(PKG)/$(firstword $(APPS))/init.txn || { echo "no package in $(PKG): see the top of this Makefile"; exit 2; }
	python3 tools/gen_hw.py --check
	./build/test_txn $(foreach a,$(APPS),$(PKG)/$(a)/init.txn $(PKG)/$(a)/insts.bin)
	./build/test_rt $(PKG) $(APPS)
	./build/test_build $(PKG) $(APPS)
	./build/test_errors $(PKG) $(firstword $(APPS))
	./build/test_ctrlpkt $(PKG) $(APPS)
	./build/test_state $(PKG) $(firstword $(APPS))

clean:
	rm -rf build
