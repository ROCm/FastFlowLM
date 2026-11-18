# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# Qwen3.8-27B MTP engine. `make` on Linux builds the .so with g++-13; under WSL
# it shells out to CMake for the Windows .dll.

# Check if it is in Windows Subsystem for Linux (WSL)
ifeq ($(shell uname -a | grep -i WSL),)
	WSL := 0
else
	WSL := 1
endif

BUILD_DIR := ../../build/qwen3_8mtp_npu
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin
FINAL_LIB_DIR := ../../build/lib
ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

COMMON_LIB_DIR := ../../build/common/lib

DEBUG_LEVEL ?= 0
VERBOSE ?= 0

TARGET = $(LIB_DIR)/libqwen3_8mtp_npu.so

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall -O3 -fopenmp
CXX_FLAGS += -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -march=native -ffast-math
CXX_FLAGS += -fmax-errors=1
CXX_FLAGS += -I../../include
CXX_FLAGS += -I../q4nx
# Internal includes are directory-rooted: "common/layer.hpp", "verify/npu/...".
CXX_FLAGS += -I.
ifeq ($(FLM_USE_HRX),1)
include ../../hrx.mk
CXX_FLAGS += $(HRX_INC)
else
CXX_FLAGS += -I/opt/xilinx/xrt/include
endif
CXX_FLAGS += -MMD -MP -DDEBUG_LEVEL=$(DEBUG_LEVEL) -DVERBOSE=$(VERBOSE)
# make PREFILL_PROFILE=1 -- per-stage prefill timings on stdout, see
# prefill/prefill_cpu.hpp's PREFILL8_PROF_TICK. Off in a shipping build: it
# prints once per stage per layer per window.
ifeq ($(PREFILL_PROFILE),1)
CXX_FLAGS += -DQWEN3_8MTP_PREFILL_PROFILE
endif
# make DECODE_PROFILE=1 -- the same treatment for modules 2 and 3, see
# common/npu/decode_prof.hpp. One block per verify pass and the host/device
# split of it. Also a -D flag with no make dependency: `make clean` first.
ifeq ($(DECODE_PROFILE),1)
CXX_FLAGS += -DQWEN3_8MTP_DECODE_PROFILE
endif
# make VISION_PROFILE=1 -- module 4's, see image/vision_profile.hpp. One table
# per encode, printed by whoever calls vision::prof::report(). Same caveat:
# a -D with no make dependency, so `make clean` first.
ifeq ($(VISION_PROFILE),1)
CXX_FLAGS += -DQWEN3_8MTP_VISION_PROFILE
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


# Objects mirror the source tree under LIB_DIR so the .d files land next to
# them. tools/ is excluded -- those files have a main(); the shared NPU modules
# (gemm, lm_head, dequant, vision_common) are unused here and would export
# another copy of symbols that already collide across engine .so files.
LOCAL_SOURCES = $(wildcard *.cpp) \
                $(wildcard common/*.cpp) \
                $(wildcard common/npu/*.cpp) \
                $(wildcard prefill/*.cpp) \
                $(wildcard prefill/npu/*.cpp) \
                $(wildcard speculate/*.cpp) \
                $(wildcard speculate/npu/*.cpp) \
                $(wildcard verify/*.cpp) \
                $(wildcard verify/npu/*.cpp) \
                $(wildcard image/*.cpp) \
                $(wildcard image/npu/*.cpp)

LOCAL_OBJS = $(patsubst %.cpp,$(LIB_DIR)/%.o,$(LOCAL_SOURCES))

OBJS = $(LOCAL_OBJS)

DEPS := $(OBJS:.o=.d)

all: directories $(TARGET)
	cp $(TARGET) $(FINAL_LIB_DIR)

directories:
	mkdir -p $(FINAL_LIB_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(LIB_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(TARGET): $(OBJS)
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

BUILD_DIR := ../../build/qwen3_8mtp_npu

TARGET = $(BUILD_DIR)/qwen3_8mtp_npu.dll

all: directories $(TARGET)

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(TARGET): directories
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../detail/qwen3_8mtp_npu $(HRX_CMAKE_ARGS)"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target qwen3_8mtp_npu --parallel 4"

clean:
	rm -rf $(BUILD_DIR)


.PHONY: all clean directories

endif 
