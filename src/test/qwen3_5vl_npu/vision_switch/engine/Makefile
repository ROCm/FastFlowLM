# Private copy of the Qwen3.5 engine (FLM_DLL/detail/qwen3_5vl_npu at git HEAD) for 2B-only work.
# Builds build/libqwen3_5vl_npu.so; load it ahead of FastFlowLM's copy with LD_LIBRARY_PATH. See README.md.

FLM_DLL := /scratch/michyu/FastFlowLM_IRON/FLM_DLL
BUILD_DIR := build
TARGET := $(BUILD_DIR)/libqwen3_5vl_npu.so
EXTRA_FLAGS ?= -DQWEN35_DECODE_PROFILE

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall -O3 -fopenmp
CXX_FLAGS += -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -march=native -ffast-math
CXX_FLAGS += -fmax-errors=1
CXX_FLAGS += -I/scratch/michyu/FastFlowLM_IRON/FLM_Xclbin/Qwen3_5_2B/vision_overlay/host
CXX_FLAGS += -I. -I$(FLM_DLL)/include -I$(FLM_DLL)/detail/q4nx -I/opt/xilinx/xrt/include
CXX_FLAGS += -MMD -MP -DDEBUG_LEVEL=0 -DVERBOSE=0 $(EXTRA_FLAGS)

LDFLAGS := -L/opt/xilinx/xrt/lib -fopenmp -lxrt_coreutil $(FLM_DLL)/external_libs/libaiebu.a
LDFLAGS += -lboost_program_options -lboost_filesystem

LOCAL_SOURCES := $(wildcard *.cpp)
SHARED_SOURCES := $(wildcard $(FLM_DLL)/detail/gemm/*.cpp) $(wildcard $(FLM_DLL)/detail/lm_head/*.cpp) \
                  $(wildcard $(FLM_DLL)/detail/vision_common/*.cpp)
DEQUANT_SOURCES := $(wildcard $(FLM_DLL)/detail/dequant/*.cpp)

LOCAL_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(LOCAL_SOURCES))
SHARED_OBJS := $(foreach s,$(SHARED_SOURCES),$(BUILD_DIR)/$(notdir $(patsubst %/,%,$(dir $(s))))_$(notdir $(s:.cpp=.o)))
DEQUANT_OBJS := $(foreach s,$(DEQUANT_SOURCES),$(BUILD_DIR)/dequant_$(notdir $(s:.cpp=.o)))

all: $(TARGET)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(BUILD_DIR)/gemm_%.o: $(FLM_DLL)/detail/gemm/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(BUILD_DIR)/lm_head_%.o: $(FLM_DLL)/detail/lm_head/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(BUILD_DIR)/vision_common_%.o: $(FLM_DLL)/detail/vision_common/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(BUILD_DIR)/dequant_%.o: $(FLM_DLL)/detail/dequant/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -fvisibility=hidden -c -o $@ $<

$(TARGET): $(LOCAL_OBJS) $(SHARED_OBJS) $(DEQUANT_OBJS)
	$(CXX) -shared $(CXX_FLAGS) -Wl,-Bsymbolic-functions -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

-include $(wildcard $(BUILD_DIR)/*.d)
.PHONY: all clean
