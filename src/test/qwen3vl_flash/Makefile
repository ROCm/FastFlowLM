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
SOURCES += ../../common/AutoModel/modeling_qwen3vl.cpp
SOURCES += ../../common/AutoModel/modeling_qwen3vl_image.cpp
SOURCES += ../../common/image_process_utils/imageproc.cpp
SOURCES += ../../common/image_process_utils/imageprocAVX512.cpp
SOURCES += ../../common/image/image_reader.cpp
SOURCES += ../../common/tokenizer/tokenizer.cpp
SOURCES += ../../common/modules/sampler.cpp

HEADERS += ../../include/models/qwen3vl_flash/qwen3vl_flash.hpp
HEADERS += ../../include/models/qwen3vl/qwen3vl_npu.hpp


ifeq ($(WSL), 0)

LDFLAGS += -lqwen3vl_flash
# modeling_qwen3vl.cpp serves both wrappers, so both engines must be linked.
LDFLAGS += -lqwen3vl_npu
LDFLAGS += -lavformat -lavcodec -lavutil -lswscale -lswresample

CPP_SOURCES := $(filter %.cpp,$(SOURCES))
OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CPP_SOURCES:.cpp=.o)))
DEPS := $(OBJECTS:.o=.d)
VPATH := $(sort $(dir $(CPP_SOURCES)))

all: $(BUILD_DIR)/test_qwen3vl_flash

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_qwen3vl_flash: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(DEPENDENCY_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

test: $(BUILD_DIR)/test_qwen3vl_flash
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && export LD_LIBRARY_PATH=../../../lib && ./test_qwen3vl_flash --model qwen3vl-flash -s 1

-include $(DEPS)
.PHONY: all clean test

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

all: directories test

directories:
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_qwen3vl_flash.exe: $(SOURCES)
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../../test/qwen3vl_flash"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target test_qwen3vl_flash"

clean:
	rm -rf $(BUILD_DIR)


test: directories $(BUILD_DIR)/test_qwen3vl_flash.exe
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ${PWSH} -Command "\$$env:PATH = '..\..\..\lib;' + \$$env:PATH; .\test_qwen3vl_flash.exe -m qwen3vl-flash:4b -s 1 -p 0"

.PHONY: all clean test directories

endif 
