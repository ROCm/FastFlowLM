# common.mk
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
# Created by Alfred

ifneq ($(filter -j,$(MAKEFLAGS)),)
    # Nothing to do; -j is already set
else
    MAKEFLAGS += -j8
endif

# VITIS related variables (Chess only: ENABLE_CHESSCC=1; Peano builds never expand them)
AIETOOLS_DIR ?= $(or $(AIETOOLS_ROOT),$(patsubst %/bin/,%,$(dir $(shell command -v xchesscc 2>/dev/null))))
# Expanded only inside the Chess recipes, so it fires only for ENABLE_CHESSCC=1.
AIETOOLS_CHECK = $(if $(strip $(AIETOOLS_DIR)),,$(error ENABLE_CHESSCC=1 needs Vitis AIE Essentials: set AIETOOLS_ROOT (or AIETOOLS_DIR), or put xchesscc on PATH))
AIE_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/versal_prod/lib
AIE2_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/aie_ml/lib

AIEOPT_DIR ?= $(shell realpath $(dir $(shell which aie-opt))/..)

WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined

# Detect compiler flag changes and wipe build/ so everything recompiles cleanly.
# Stamp lives outside build/ so it survives the wipe.
_CONFIG_STAMP := ${HOME_DIR}/.compiler_config_stamp
# The GPT-OSS tiles take the MM_PEANO_BFP16_STEP path in mm_optimized_common.h under
# Peano. The pre-RA scheduler would overlap its conversions again, so it is off here.
ifeq ($(ENABLE_CHESSCC),0)
ifneq ($(filter GPTOSS-%,$(MODEL_TYPE)),)
MM_PEANO_EXTRA_FLAGS = -DMM_PEANO_BFP16_STEP -mllvm -enable-misched=false
endif
endif
_CONFIG_CONTENT := ENABLE_CHESSCC=$(ENABLE_CHESSCC) ENABLE_EMULATION_IN_COMPILER=$(ENABLE_EMULATION_IN_COMPILER) $(MM_PEANO_EXTRA_FLAGS)
$(shell \
  if [ "$$(cat $(_CONFIG_STAMP) 2>/dev/null)" != "$(_CONFIG_CONTENT)" ]; then \
    echo "Compiler flags changed, wiping build/..." >&2; \
    rm -rf ${HOME_DIR}/build; \
    echo "$(_CONFIG_CONTENT)" > $(_CONFIG_STAMP); \
  fi)


ifneq ($(ENABLE_EMULATION_IN_COMPILER),0)
CHESSCCWRAP2P_FLAGS = aie2p $(AIETOOLS_CHECK)-I ${AIETOOLS_DIR}/include  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
else
CHESSCCWRAP2P_FLAGS = aie2p $(AIETOOLS_CHECK)-I ${AIETOOLS_DIR}/include
endif



ifneq ($(ENABLE_EMULATION_IN_COMPILER),0)
PEANOWRAP2P_FLAGS = -O2 -v -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 $(MM_PEANO_EXTRA_FLAGS)
else
PEANOWRAP2P_FLAGS = -O2 -v -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include $(MM_PEANO_EXTRA_FLAGS)
endif