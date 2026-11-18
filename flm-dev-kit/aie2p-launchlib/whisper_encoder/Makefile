##===- Makefile - the Whisper encoder trial of the launch library -------===##
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
#   ../vadd_vmul_bf16/env.sh make package    the encoder kernels -> build/pkg (tools/build_packages.sh)
#   make check                               no NPU: register model, binaries round trip
#   make switch                              step 1: warm switch vs hw_context switch, on the NPU

LAUNCHLIB := $(realpath ../launchlib)
PKG       ?= build/pkg
COUNT     ?= 400
HOST_CXX  ?= g++-13
HOST_FLAGS := -std=c++20 -O2 -Wall -I $(LAUNCHLIB)/include
HOST_LIBS  := -lxrt_coreutil -luuid

.PHONY: package xroot engine check switch compare clean
package:
	tools/build_packages.sh

# the five xclbins of this tree for the engine (needs `package` first)
xroot:
	tools/build_xroot.sh

build/%.exe: host/%.cpp $(shell find $(LAUNCHLIB)/include -name '*.hpp')
	mkdir -p $(@D)
	$(HOST_CXX) $(HOST_FLAGS) $< -o $@ $(HOST_LIBS)

# --- step 2: the Whisper engine of this tree (FLM_DLL/detail/whisper_npu) ----------

FLM_DLL   := $(realpath ../../FLM_DLL)
MODEL     ?= $(error set MODEL to the Whisper-V3-Turbo-NPU2 model folder of a FastFlowLM install)
WINDOWS   ?= 3
FLM_FLAGS := -std=c++23 -O3 -fopenmp -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -march=native -ffast-math \
             -I $(FLM_DLL)/include
FLM_LIBS  := -L$(FLM_DLL)/build/whisper_npu/lib -L$(FLM_DLL)/build/q4nx/lib -lwhisper_npu -lq4_npu_eXpress \
             -Wl,-rpath,$(FLM_DLL)/build/whisper_npu/lib -Wl,-rpath,$(FLM_DLL)/build/q4nx/lib \
             -lxrt_coreutil -lboost_program_options -lboost_filesystem -fopenmp

# An engine without FLM_WHISPER_SWITCH_DIR would run today's path twice and compare equal.
engine:
	@grep -q FLM_WHISPER_SWITCH_DIR $(FLM_DLL)/detail/whisper_npu/whisper_encoder.cpp || { \
		echo "this tree's Whisper engine has no FLM_WHISPER_SWITCH_DIR (README, step 2)"; exit 2; }
	$(MAKE) -C $(FLM_DLL)/detail/q4nx
	$(MAKE) -C $(FLM_DLL)/detail/whisper_npu

build/encoder_compare.exe: host/encoder_compare.cpp
	mkdir -p $(@D)
	$(HOST_CXX) $(FLM_FLAGS) $< -o $@ $(FLM_LIBS)

# The switch files hold the packages' programs and configuration: made again when a package changes.
build/switch/to_attn.txn: build/make_switch.exe $(foreach a,mm attn,$(PKG)/$(a)/init.txn $(PKG)/$(a)/manifest.txt)
	build/make_switch.exe --package $(PKG) --out build/switch

# today's path, then the one-context path; the K/V files must be equal
compare: engine build/encoder_compare.exe build/switch/to_attn.txn
	build/encoder_compare.exe --model $(MODEL) --xroot build/xroot --out build/kv_off.bin --windows $(WINDOWS)
	FLM_WHISPER_SWITCH_DIR=$(CURDIR)/build/switch build/encoder_compare.exe --model $(MODEL) --xroot build/xroot \
		--out build/kv_on.bin --windows $(WINDOWS)
	cmp build/kv_off.bin build/kv_on.bin && echo "K/V: IDENTICAL"

check:
	$(MAKE) -C $(LAUNCHLIB) build/test_txn build/test_build
	$(LAUNCHLIB)/build/test_txn $(PKG)/mm/init.txn $(PKG)/attn/init.txn
	$(LAUNCHLIB)/build/test_build $(PKG) mm attn
	$(LAUNCHLIB)/build/test_build $(PKG) attn mm

switch: build/switch_cost.exe
	build/switch_cost.exe --package $(PKG) --count $(COUNT)

clean:
	rm -rf build
