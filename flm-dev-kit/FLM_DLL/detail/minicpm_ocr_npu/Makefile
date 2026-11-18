# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# =============================================================================
# MiniCPM-V NPU Library Makefile
# =============================================================================
#
# Created: 2024
#
# This Makefile is used to build the MiniCPM-V NPU library components.
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

BUILD_DIR := ../../build/minicpm_ocr_npu
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin
FINAL_LIB_DIR := ../../build/lib
ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

COMMON_LIB_DIR := ../../build/common/lib

DEBUG_LEVEL ?= 0
VERBOSE ?= 0

TARGET = $(LIB_DIR)/libminicpm_ocr_npu.so

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

# Backend selection, one flag per overlay (minicpm_ocr_prefill_cpu.hpp). Each flag moves
# one stage between the array and the host:
#   PREFILL_MM=cpu    the prefill projections            (mm.xclbin)
#   PREFILL_ATTN=cpu  the full-attention layers' attention (attn.xclbin)
#   PREFILL_GDN=cpu   the gated delta rule               (GateDeltaNet_prefill.xclbin)
#   PREFILL_CONV=cpu  the causal conv1d + q/k L2 norm    (conv8.xclbin)
#   DECODE=cpu        decode and the lm_head             (layer.xclbin, lm_head.xclbin)
# All five set to cpu registers no xclbin at all: the whole model runs on the host.
#
# CONV defaults to the HOST, and that is a measured choice, not an unfinished port:
# a token-serial stage on the array pays a bitstream switch it does not amortize.
#
# GDN defaults to `npu`, but that now means "built in, chosen per prefill call at run time":
# prefill calls of >= 5120 tokens run the gated delta rule on the array, shorter ones decline
# and run it on the host (minicpm_ocr_prefill.hpp, gdn_npu_min_tokens(); override with
# FLM_MINICPM_OCR_GDN_NPU_MIN_TOKENS). That split only pays off with the REWRITTEN
# kernel from FLM_Xclbin/minicpm_ocr/gate_delta_net_prefill (README.md, "GDN and attention
# placement re-tuned"; it was 1536 before the host GDN got faster). With the older
# Qwen3_5 kernel the array never wins and PREFILL_GDN=cpu is the right build.
#
# The history below is from the older kernel, when both stages lost on the array:
# both were token-serial there, so they paid a bitstream switch and a full activation
# round trip for arithmetic the host does with AVX-512 in less time.
# Language prefill, 849-token prompt, idle box (older GDN kernel):
#     conv=npu gdn=npu   589 ms   (1442 tok/s)   <- all-array
#     conv=cpu gdn=npu   514 ms   (1652 tok/s)
#     conv=cpu gdn=cpu   372 ms   (2282 tok/s)   <- the default then, +58%
# Decode is untouched either way (it runs the fused layer.xclbin), and GSM8K/50 moves
# 62% -> 68-72%, i.e. not worse. Set either back to npu to revert; nothing else changes.
#
# The one case to re-measure before trusting the default: `flm serve` under CONCURRENT
# load. These stages now use OpenMP on the host, while decode's run.wait() polls, so a
# prefill in one request can slow a decode in another in a way the all-array build did
# not. The numbers above are single-stream.
PREFILL_MM ?= npu
ifeq ($(PREFILL_MM), cpu)
CXX_FLAGS += -DFLM_PREFILL_MM_ON_CPU
endif
PREFILL_ATTN ?= npu
ifeq ($(PREFILL_ATTN), cpu)
CXX_FLAGS += -DFLM_PREFILL_ATTN_ON_CPU
endif
PREFILL_GDN ?= npu
ifeq ($(PREFILL_GDN), cpu)
CXX_FLAGS += -DFLM_PREFILL_GDN_ON_CPU
endif
PREFILL_CONV ?= cpu
ifeq ($(PREFILL_CONV), cpu)
CXX_FLAGS += -DFLM_PREFILL_CONV_ON_CPU
endif
DECODE ?= npu
ifeq ($(DECODE), cpu)
CXX_FLAGS += -DFLM_DECODE_ON_CPU
endif
# DECODE_PROFILE=1 times the array decode loop per stage (layers / lm_head / host work)
# and prints to stderr every 32 tokens. Compiled out otherwise. Part of CXX_FLAGS so the
# flag stamp below rebuilds when it is flipped.
# VISION_PROFILE=1 times the vision tower per stage (image/vision_profile.hpp);
# a table per encode is printed by the verify driver. Compiled out otherwise.
VISION_PROFILE ?= 0
ifeq ($(VISION_PROFILE), 1)
CXX_FLAGS += -DMINICPM_OCR_VISION_PROFILE
endif
DECODE_PROFILE ?= 0
ifeq ($(DECODE_PROFILE), 1)
CXX_FLAGS += -DFLM_DECODE_PROFILE
endif

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
LDFLAGS += -L../../../../src/lib/xrt
LDFLAGS += -L../../build/lib
ifeq ($(FLM_USE_HRX),1)
LDFLAGS += $(HRX_LIBS)
else
# Link the aiebu that matches ../../include/aiebu by path. With -laiebu, an
# XRT installed in /opt/xilinx/xrt/lib (searched first) supplies its own
# libaiebu.a, whose assembler prints a "Header version ..." banner to stdout
# on every _setup_kernel().
LDFLAGS += -lxrt_coreutil ../../../../src/lib/xrt/libaiebu.a
endif


# Each source compiles to an object under LIB_DIR mirroring the source tree, so
# dependency (.d) files are emitted next to the objects and -include picks up
# header changes. image/ is the vision tower (image/README.md); tools/ and
# host_ref/ hold their own main()s and are not part of the library.
LOCAL_SOURCES = $(wildcard *.cpp) $(wildcard image/*.cpp) $(wildcard image/npu/*.cpp)
GEMM_SOURCES = $(wildcard ../gemm/*.cpp)
LMHEAD_SOURCES = $(wildcard ../lm_head/*.cpp)

LOCAL_OBJS = $(patsubst %.cpp,$(LIB_DIR)/%.o,$(LOCAL_SOURCES))
GEMM_OBJS = $(patsubst %.cpp,$(LIB_DIR)/gemm_%.o,$(notdir $(GEMM_SOURCES)))
LMHEAD_OBJS = $(patsubst %.cpp,$(LIB_DIR)/lm_head_%.o,$(notdir $(LMHEAD_SOURCES)))

# Compile dequant with hidden visibility to avoid runtime symbol collision
# with other .so files (e.g. libqwen2_npu.so) that export the same Dequant:: symbols.
DEQUANT_SOURCES = $(wildcard ../dequant/*.cpp)
DEQUANT_OBJS = $(patsubst %.cpp,$(LIB_DIR)/dequant_%.o,$(notdir $(DEQUANT_SOURCES)))

OBJS = $(LOCAL_OBJS) $(GEMM_OBJS) $(LMHEAD_OBJS)

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

$(LIB_DIR)/lm_head_%.o: ../lm_head/%.cpp $(FLAGS_STAMP)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(LIB_DIR)/dequant_%.o: ../dequant/%.cpp $(FLAGS_STAMP)
	$(CXX) $(CXX_FLAGS) -fvisibility=hidden -c -o $@ $<

$(TARGET): $(OBJS) $(DEQUANT_OBJS)
	$(CXX) $(LIB_FLAGS) -o $@ $^ $(LDFLAGS)


clean:
	rm -rf $(BUILD_DIR)


-include $(DEPS)
.PHONY: all clean test directories FORCE

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe
HRX_CMAKE_ARGS :=
ifeq ($(FLM_USE_HRX),1)
HRX_CMAKE_ARGS := -DFLM_USE_HRX=ON
endif

BUILD_DIR := ../../build/minicpm_ocr_npu

TARGET = $(BUILD_DIR)/minicpm_ocr_npu.dll

all: directories $(TARGET)

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(TARGET): directories
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../detail/minicpm_ocr_npu $(HRX_CMAKE_ARGS)"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target minicpm_ocr_npu --parallel 4"

clean:
	rm -rf $(BUILD_DIR)


.PHONY: all clean directories

endif 
