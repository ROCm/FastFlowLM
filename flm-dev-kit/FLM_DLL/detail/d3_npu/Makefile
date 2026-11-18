# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# =============================================================================
# d3 (Decision 3.0) decision-encoder NPU library Makefile
# =============================================================================
#
# Derived from the minicpm_v_4_7_npu Makefile (same text tower, same host kernels).
#
# This Makefile builds libd3_npu.so.
#
# Usage:
#   make        - Build all targets
#   make clean  - Remove all built files
#   make help   - Show this help message
#
# =============================================================================

# Check if it is in Windows Subsystem for Linux (WSL)
ifeq ($(shell uname -a | grep -i WSL),)
	WSL := 0
else
	WSL := 1
endif

BUILD_DIR := ../../build/d3_npu
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin
FINAL_LIB_DIR := ../../build/lib
ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

COMMON_LIB_DIR := ../../build/common/lib

DEBUG_LEVEL ?= 0
VERBOSE ?= 0

TARGET = $(LIB_DIR)/libd3_npu.so

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall -O3 -fopenmp
CXX_FLAGS += -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -march=native -ffast-math
CXX_FLAGS += -fmax-errors=1
CXX_FLAGS += -I.
CXX_FLAGS += -I../../include
CXX_FLAGS += -I../q4nx
ifeq ($(FLM_USE_HRX),1)
include ../../hrx.mk
CXX_FLAGS += $(HRX_INC)
else
CXX_FLAGS += -I/opt/xilinx/xrt/include
endif
CXX_FLAGS += -MMD -MP -DDEBUG_LEVEL=$(DEBUG_LEVEL) -DVERBOSE=$(VERBOSE)

# Backend selection (d3_prefill_cpu.hpp). The prefill GEMMs run on the array, and so does the
# attention of long prompts (attn.xclbin); conv1d and the gated delta rule run on the host.
#   PREFILL_MM=cpu   every projection on the host too: no xclbin, no device work at all
PREFILL_MM ?= npu
ifeq ($(PREFILL_MM), cpu)
CXX_FLAGS += -DFLM_PREFILL_MM_ON_CPU
endif
# RIG=1 compiles in named-checkpoint dumps (D3_RIG_DUMP=<dir> at run time), named like the
# checkpoints models_simple/verify_d3_lite.py saves.
RIG ?= 0
ifeq ($(RIG), 1)
CXX_FLAGS += -DD3_RIG
endif
# PREFILL_PROFILE=1 prints per-stage prefill times.
PREFILL_PROFILE ?= 0
ifeq ($(PREFILL_PROFILE), 1)
CXX_FLAGS += -DD3_PREFILL_PROFILE
endif
CXX_FLAGS += $(EXTRA_CXX_FLAGS)

# Flags for shared library
LIB_FLAGS := -shared
LIB_FLAGS += $(CXX_FLAGS)
# Each engine .so compiles its own copy of the shared runtime (npu_sequence,
# npu_app_manager, weight_desc_t, bytes/buffer, gen_mha_main, simd_*, ...) and
# exports it with default visibility. flm links every engine at once, so ELF
# interposition makes all references -- including calls from inside this .so --
# bind to whichever engine comes first in DT_NEEDED, silently running another
# model's implementation. -Bsymbolic-functions binds our internal function calls
# to our own definitions at link time. Functions only: typeinfo/vtables stay
# interposable so dynamic_cast across the .so boundary keeps working.
LIB_FLAGS += -Wl,-Bsymbolic-functions

# Flags for test executable
TEST_FLAGS := $(CXX_FLAGS)
TEST_FLAGS += -L$(LIB_DIR)

ifeq ($(FLM_USE_HRX),1)
LDFLAGS += $(HRX_RPATH)
else
LDFLAGS += -L/opt/xilinx/xrt/lib
endif

LDFLAGS += -fopenmp
LDFLAGS += -L../../external_libs
LDFLAGS += -L../../build/lib
ifeq ($(FLM_USE_HRX),1)
LDFLAGS += $(HRX_LIBS)
else
LDFLAGS += -lxrt_coreutil -laiebu
endif
ifeq ($(FLM_USE_HRX),1)
# HRX path links no Boost (no engine source uses Boost); Linux engine .so stay Boost-free.
else
LDFLAGS += -lboost_program_options -lboost_filesystem
endif


# Each source compiles to an object under LIB_DIR, so dependency (.d) files are emitted
# next to the objects and -include picks up header changes. tools/ holds its own main()s
# and is not part of the library.
LOCAL_SOURCES = $(wildcard *.cpp)
GEMM_SOURCES = $(wildcard ../gemm/*.cpp)

# The vision tower (d3_vision.cpp) also compiles the shared vision sources and two of
# qwen3_5vl_npu's, unchanged, from where they live.
VISION_SOURCES = $(wildcard ../vision_common/*.cpp)
QWEN_VISION_SOURCES = ../qwen3_5vl_npu/rot_pos_emb.cpp ../qwen3_5vl_npu/qwen3_5vision_prefill_helper.cpp

LOCAL_OBJS = $(patsubst %.cpp,$(LIB_DIR)/%.o,$(LOCAL_SOURCES))
GEMM_OBJS = $(patsubst %.cpp,$(LIB_DIR)/gemm_%.o,$(notdir $(GEMM_SOURCES)))
VISION_OBJS = $(patsubst %.cpp,$(LIB_DIR)/vision_%.o,$(notdir $(VISION_SOURCES)))
QWEN_VISION_OBJS = $(patsubst %.cpp,$(LIB_DIR)/qwen_vision_%.o,$(notdir $(QWEN_VISION_SOURCES)))

# Compile dequant with hidden visibility to avoid runtime symbol collision
# with other .so files (e.g. libqwen2_npu.so) that export the same Dequant:: symbols.
DEQUANT_SOURCES = $(wildcard ../dequant/*.cpp)
DEQUANT_OBJS = $(patsubst %.cpp,$(LIB_DIR)/dequant_%.o,$(notdir $(DEQUANT_SOURCES)))

OBJS = $(LOCAL_OBJS) $(GEMM_OBJS) $(VISION_OBJS) $(QWEN_VISION_OBJS)

DEPS := $(OBJS:.o=.d) $(DEQUANT_OBJS:.o=.d)

all: directories $(TARGET)
	cp $(TARGET) $(FINAL_LIB_DIR)

directories:
	mkdir -p $(FINAL_LIB_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

# Every object depends on the exact flag set, so flipping a backend flag (or
# DEBUG_LEVEL) rebuilds instead of silently relinking objects built for another
# configuration. The stamp is rewritten only when the flags actually change.
FLAGS_STAMP := $(LIB_DIR)/.cxx_flags
$(FLAGS_STAMP): FORCE
	@mkdir -p $(dir $@)
	@echo '$(CXX_FLAGS)' | cmp -s - $@ || echo '$(CXX_FLAGS)' > $@
FORCE:

$(LIB_DIR)/%.o: %.cpp $(FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(LIB_DIR)/gemm_%.o: ../gemm/%.cpp $(FLAGS_STAMP)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(LIB_DIR)/vision_%.o: ../vision_common/%.cpp $(FLAGS_STAMP)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(LIB_DIR)/qwen_vision_%.o: ../qwen3_5vl_npu/%.cpp $(FLAGS_STAMP)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(LIB_DIR)/dequant_%.o: ../dequant/%.cpp $(FLAGS_STAMP)
	$(CXX) $(CXX_FLAGS) -fvisibility=hidden -c -o $@ $<

$(TARGET): $(OBJS) $(DEQUANT_OBJS)
	$(CXX) $(LIB_FLAGS) -o $@ $^ $(LDFLAGS)


# The engine-level harness (tools/d3_engine_run.cpp): token-id prompts in, final-norm hidden rows
# out, timing per call. Used by the rig and by .github/workflows/d3-check.yml.
harness: all
	$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/d3_engine_run tools/d3_engine_run.cpp \
		-L$(LIB_DIR) -L$(FINAL_LIB_DIR) -ld3_npu -lq4_npu_eXpress $(LDFLAGS) \
		-Wl,-rpath,'$$ORIGIN/../lib' -Wl,-rpath,'$$ORIGIN/../../lib'

# Request-level harness (tools/d3_req_run.cpp): whole requests from files through encode_state_once (M3 / M4 validation).
req_harness: all
	$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/d3_req_run tools/d3_req_run.cpp \
		-L$(LIB_DIR) -L$(FINAL_LIB_DIR) -ld3_npu -lq4_npu_eXpress $(LDFLAGS) \
		-Wl,-rpath,'$$ORIGIN/../lib' -Wl,-rpath,'$$ORIGIN/../../lib'

# The vision tower alone (tools/vision_bench.cpp): timing of many-segment calls, segment independence.
vision_bench: all
	$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/d3_vision_bench tools/vision_bench.cpp \
		-L$(LIB_DIR) -L$(FINAL_LIB_DIR) -ld3_npu -lq4_npu_eXpress $(LDFLAGS) \
		-Wl,-rpath,'$$ORIGIN/../lib' -Wl,-rpath,'$$ORIGIN/../../lib'

# Host-only self-test of prefix_bidir attention (GQA 4 / 6), the gated delta rule at 3 value heads per
# key head and the 48-output alpha / beta projection (tools/test_prefix_bidir_cpu.cpp). No device needed.
test_cpu: directories
	$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/d3_test_prefix_bidir_cpu tools/test_prefix_bidir_cpu.cpp $(LDFLAGS)
	$(BIN_DIR)/d3_test_prefix_bidir_cpu

# Host-only self-test of the state-once kernels (tools/test_state_once_cpu.cpp): attention over a cached state
# plus the block's own keys, conv tail and recurrent-state carry, each bit-identical to the joint pass.
state_once_test: directories
	$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/d3_test_state_once_cpu tools/test_state_once_cpu.cpp $(LDFLAGS)
	$(BIN_DIR)/d3_test_state_once_cpu

# Host-only test of the M-RoPE position planner (d3_positions.hpp) against transformers' get_rope_index fixtures
# (tools/fixtures: the FLM video agent's prompts and 327 real golden-set prompts), incl. state / question-block splits.
positions_test: directories
	$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/d3_test_positions_cpu tools/test_positions_cpu.cpp $(LDFLAGS)
	$(BIN_DIR)/d3_test_positions_cpu tools/fixtures

# Engine-level tests on the array (tools/test_attn_array.cpp): array two-run attention vs host two-call.
attn_array_test: all
	$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/d3_test_attn_array tools/test_attn_array.cpp \
		-L$(LIB_DIR) -L$(FINAL_LIB_DIR) -ld3_npu -lq4_npu_eXpress $(LDFLAGS) \
		-Wl,-rpath,'$$ORIGIN/../lib' -Wl,-rpath,'$$ORIGIN/../../lib'

# The 32,768-row up/gate GEMM of a 27B-shaped layer (result buffer > 2^31 bytes), tools/test_mm_big.cpp.
mm_big_test: all
	$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/d3_test_mm_big tools/test_mm_big.cpp \
		-L$(LIB_DIR) -L$(FINAL_LIB_DIR) -ld3_npu -lq4_npu_eXpress $(LDFLAGS) \
		-Wl,-rpath,'$$ORIGIN/../lib' -Wl,-rpath,'$$ORIGIN/../../lib'

clean:
	rm -rf $(BUILD_DIR)


-include $(DEPS)
.PHONY: all clean test directories harness req_harness vision_bench test_cpu state_once_test positions_test attn_array_test mm_big_test FORCE

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe
HRX_CMAKE_ARGS :=
ifeq ($(FLM_USE_HRX),1)
HRX_CMAKE_ARGS := -DFLM_USE_HRX=ON
endif

BUILD_DIR := ../../build/d3_npu

TARGET = $(BUILD_DIR)/d3_npu.dll

all: directories $(TARGET)

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(TARGET): directories
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../detail/d3_npu $(HRX_CMAKE_ARGS)"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target d3_npu --parallel 4"

clean:
	rm -rf $(BUILD_DIR)


.PHONY: all clean directories

endif 
