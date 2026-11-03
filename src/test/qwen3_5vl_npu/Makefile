# =============================================================================
# Qwen NPU Library Makefile
# =============================================================================
#
# Author: Zhenyu Xu (zxu3@clemson.edu)
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
-include ../common.mk


SOURCES += test.cpp
SOURCES += ../../common/AutoModel/automodel.cpp
SOURCES += ../../common/AutoModel/model_backend.cpp
SOURCES += ../../common/prefix_slot_cache.cpp
SOURCES += $(CHAT_SOURCES)
SOURCES += ../../common/AutoModel/modeling_qwen3_5vl.cpp
SOURCES += ../../common/AutoModel/modeling_qwen3_5vl_image.cpp
SOURCES += ../../common/image_process_utils/imageproc.cpp
SOURCES += ../../common/image_process_utils/imageprocAVX512.cpp
SOURCES += ../../common/image/image_reader.cpp
SOURCES += ../../include/lodepng/lodepng.cpp
SOURCES += ../../common/tokenizer/tokenizer.cpp
SOURCES += ../../common/modules/sampler.cpp

HEADERS += ../../include/qwen3_5vl/qwen3_5vl_npu_sequence.hpp
HEADERS += ../../include/qwen3_5vl/qwen3_5vl_npu.hpp


ifeq ($(WSL), 0)

CXX_FLAGS += -DFLM_HAS_QWEN3_5VL_FLASH=1
LDFLAGS += -lqwen3_5vl_npu
LDFLAGS += -lqwen3_5vl_flash
LDFLAGS += -ljpeg
LDFLAGS += -L../../build/libwebp/lib -lwebpdecoder

CPP_SOURCES := $(filter %.cpp,$(SOURCES))
OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CPP_SOURCES:.cpp=.o)))
DEPS := $(OBJECTS:.o=.d)
VPATH := $(sort $(dir $(CPP_SOURCES)))

all: $(BUILD_DIR)/test_qwen3_5vl_npu

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_qwen3_5vl_npu: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(DEPENDENCY_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

FLM_MODEL_PATH ?= /scratch/michyu
# qwen3.5-flash:2b = the optimized 2B (libqwen3_5vl_flash.so), qwen3.5:2b = the original
MODEL ?= qwen3.5-flash:2b

test: $(BUILD_DIR)/test_qwen3_5vl_npu
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cp bench_text.txt $(BUILD_DIR)/bench_text.txt
	cd $(BUILD_DIR) && export LD_LIBRARY_PATH=../../../lib/xrt FLM_MODEL_PATH=$(FLM_MODEL_PATH) && ./test_qwen3_5vl_npu --model $(MODEL) $(ARGS)

-include $(DEPS)
.PHONY: all clean test

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

all: directories test

directories:
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_qwen3_5vl_npu.exe: $(SOURCES)
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../../test/qwen3_5vl_npu"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target test_qwen3_5vl_npu"

clean:
	rm -rf $(BUILD_DIR)


test: directories $(BUILD_DIR)/test_qwen3_5vl_npu.exe
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ${PWSH} -Command "\$$env:PATH = '..\..\..\lib;' + \$$env:PATH; .\test_qwen3_5vl_npu.exe -m qwen3.5:9b -s 1 -p 0"

.PHONY: all clean test directories

endif 
