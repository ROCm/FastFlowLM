# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# =============================================================================
# Qwen NPU Library Makefile
# =============================================================================
#
# Created: 2024
#
# This Makefile is used to build the Qwen NPU library components.
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

BUILD_DIR := ../../build/qwen3_5_omni_npu
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin
FINAL_LIB_DIR := ../../build/lib
ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

COMMON_LIB_DIR := ../../build/common/lib

DEBUG_LEVEL ?= 0
VERBOSE ?= 0

TARGET = $(LIB_DIR)/libqwen3_5_omni_npu.so

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall -O3 -fopenmp
CXX_FLAGS += -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -march=native -ffast-math
CXX_FLAGS += -fmax-errors=1
CXX_FLAGS += -I../../include
CXX_FLAGS += -I../q4nx
CXX_FLAGS += -I.
CXX_FLAGS += -Iqwen3_5_omni_thinker
CXX_FLAGS += -Iqwen3_5_omni_thinker/qwen3_5_omni_image_embed
CXX_FLAGS += -Iqwen3_5_omni_thinker/qwen3_5_omni_audio_embed
CXX_FLAGS += -Iqwen3_5_omni_talker
ifeq ($(FLM_USE_HRX),1)
include ../../hrx.mk
CXX_FLAGS += $(HRX_INC)
else
CXX_FLAGS += -I/opt/xilinx/xrt/include
endif
CXX_FLAGS += -MMD -MP -DDEBUG_LEVEL=$(DEBUG_LEVEL) -DVERBOSE=$(VERBOSE)
# CXX_FLAGS += -DMVPREFILL


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
# Link the aiebu that matches ../../include/aiebu by path. With -laiebu, an
# XRT installed in /opt/xilinx/xrt/lib (searched first) supplies its own
# libaiebu.a, whose assembler prints a "Header version ..." banner to stdout
# on every _setup_kernel().
LDFLAGS += -lxrt_coreutil ../../external_libs/libaiebu.a
endif
ifeq ($(FLM_USE_HRX),1)
# HRX path links no Boost (no engine source uses Boost); Linux engine .so stay Boost-free.
else
LDFLAGS += -lboost_program_options -lboost_filesystem
endif


# Local sources live in the top-level dir plus the talker and thinker subdirs
# (image/audio embed now nest under thinker); VPATH lets the flattened object
# rule below find them by basename.
LOCAL_SOURCES = $(wildcard *.cpp)
LOCAL_SOURCES += $(wildcard qwen3_5_omni_talker/*.cpp)
LOCAL_SOURCES += $(wildcard qwen3_5_omni_thinker/*.cpp)
LOCAL_SOURCES += $(wildcard qwen3_5_omni_thinker/qwen3_5_omni_image_embed/*.cpp)
LOCAL_SOURCES += $(wildcard qwen3_5_omni_thinker/qwen3_5_omni_audio_embed/*.cpp)
GEMM_SOURCES = $(wildcard ../gemm/*.cpp)
MHA_SOURCES = $(wildcard ../mha/*.cpp)
VISION_SOURCES = $(wildcard ../vision_common/*.cpp)

VPATH = qwen3_5_omni_talker:qwen3_5_omni_thinker:qwen3_5_omni_thinker/qwen3_5_omni_image_embed:qwen3_5_omni_thinker/qwen3_5_omni_audio_embed

# Each source compiles to a flattened object in LIB_DIR so dependency (.d)
# files are emitted next to the objects and -include picks up header changes.
LOCAL_OBJS = $(patsubst %.cpp,$(LIB_DIR)/%.o,$(notdir $(LOCAL_SOURCES)))
GEMM_OBJS = $(patsubst %.cpp,$(LIB_DIR)/gemm_%.o,$(notdir $(GEMM_SOURCES)))
MHA_OBJS = $(patsubst %.cpp,$(LIB_DIR)/mha_%.o,$(notdir $(MHA_SOURCES)))
VISION_OBJS = $(patsubst %.cpp,$(LIB_DIR)/vision_%.o,$(notdir $(VISION_SOURCES)))

# Compile dequant with hidden visibility to avoid runtime symbol collision
# with other .so files (e.g. libqwen2_npu.so) that export the same Dequant:: symbols.
DEQUANT_SOURCES = $(wildcard ../dequant/*.cpp)
DEQUANT_OBJS = $(patsubst %.cpp,$(LIB_DIR)/dequant_%.o,$(notdir $(DEQUANT_SOURCES)))

OBJS = $(LOCAL_OBJS) $(GEMM_OBJS) $(MHA_OBJS) $(VISION_OBJS)

DEPS := $(OBJS:.o=.d) $(DEQUANT_OBJS:.o=.d)

all: directories $(TARGET)
	cp $(TARGET) $(FINAL_LIB_DIR)

directories:
	mkdir -p $(FINAL_LIB_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(LIB_DIR)/%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(LIB_DIR)/gemm_%.o: ../gemm/%.cpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

# Compile MHA with hidden visibility, for the same reason as dequant below: this
# is the only engine that builds ../mha/*.cpp into itself, and it used to export
# those symbols. Every other engine leaves the MHA symbols undefined and expects
# libmha.so, but flm lists this .so in DT_NEEDED, so ELF interposition made them
# all bind to the copy frozen into this library instead. A stale copy then went
# unnoticed until some other model hit an mha_type_t this one had never heard of
# and _get_ct_chunk_size fell through to `return 0` -- a divide-by-zero at the
# first prompt. Hidden visibility keeps our copy to ourselves.
$(LIB_DIR)/mha_%.o: ../mha/%.cpp
	$(CXX) $(CXX_FLAGS) -fvisibility=hidden -c -o $@ $<

$(LIB_DIR)/vision_%.o: ../vision_common/%.cpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(LIB_DIR)/dequant_%.o: ../dequant/%.cpp
	$(CXX) $(CXX_FLAGS) -fvisibility=hidden -c -o $@ $<

$(TARGET): $(OBJS) $(DEQUANT_OBJS)
	$(CXX) $(LIB_FLAGS) -o $@ $^ $(LDFLAGS)


clean:
	rm -rf $(BUILD_DIR)


-include $(DEPS)
.PHONY: all clean test directories

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe
HRX_CMAKE_ARGS :=
ifeq ($(FLM_USE_HRX),1)
HRX_CMAKE_ARGS := -DFLM_USE_HRX=ON
endif

BUILD_DIR := ../../build/qwen3_5_omni_npu

TARGET = $(BUILD_DIR)/qwen3_5_omni_npu.dll

all: directories $(TARGET)

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(TARGET): directories
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../detail/qwen3_5_omni $(HRX_CMAKE_ARGS)"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target qwen3_5_omni_npu --parallel 4"

clean:
	rm -rf $(BUILD_DIR)


.PHONY: all clean directories

endif
