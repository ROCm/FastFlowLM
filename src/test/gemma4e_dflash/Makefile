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
SOURCES += $(CHAT_SOURCES)
SOURCES += ../../common/AutoModel/modeling_gemma4e.cpp
SOURCES += ../../common/AutoModel/modeling_gemma4e_dflash.cpp
SOURCES += ../../common/AutoModel/modeling_gemma4e_image.cpp
SOURCES += ../../common/AutoModel/modeling_gemma4e_audio.cpp
SOURCES += ../../common/image_process_utils/imageproc.cpp
SOURCES += ../../common/image_process_utils/imageprocAVX512.cpp
SOURCES += ../../common/audio_process_utils/audioproc.cpp
SOURCES += ../../common/audio_process_utils/audioprocAVX512.cpp
SOURCES += ../../common/image/image_reader.cpp
SOURCES += ../../include/lodepng/lodepng.cpp
SOURCES += ../../common/audio/audio_reader.cpp
SOURCES += ../../common/audio/audio_decoder.cpp
SOURCES += ../../common/audio/audio_resampler.cpp
SOURCES += ../../common/tokenizer/tokenizer.cpp
SOURCES += ../../common/modules/sampler.cpp

HEADERS += ../../include/models/gemma4e_dflash/flm/aie2p/gemma4e_dflash.hpp
HEADERS += ../../include/models/gemma4e_flash/flm/aie2p/gemma4e_flash.hpp
HEADERS += ../../include/models/gemma4e/flm/aie2p/gemma4e_npu.hpp

# Test task selection; text mode is default
# 0: text
# 1: images
# 2: audios
# 3: omni (image + audio)
# 4: long prompt from prompt.txt (prefill only)
TYPE ?= 0



ifeq ($(WSL), 0)

LDFLAGS += -lgemma4e_dflash
LDFLAGS += -lgemma4e_flash
# modeling_gemma4e.cpp serves both wrappers, so both engines must be linked.
LDFLAGS += -lgemma4e_npu
LDFLAGS += -ljpeg -lvorbisfile -lvorbis -logg -lopus

CPP_SOURCES := $(filter %.cpp,$(SOURCES))
OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CPP_SOURCES:.cpp=.o)))
DEPS := $(OBJECTS:.o=.d)
VPATH := $(sort $(dir $(CPP_SOURCES)))

all: $(BUILD_DIR)/test_gemma4e_dflash

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_gemma4e_dflash: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(DEPENDENCY_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

test: $(BUILD_DIR)/test_gemma4e_dflash
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && export LD_LIBRARY_PATH=../../../lib && ./test_gemma4e_dflash --model nemotron3.5-bolt-dflash:e4b --type $(TYPE)

-include $(DEPS)
.PHONY: all clean test

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

all: directories test

directories:
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_gemma4e_dflash.exe: $(SOURCES)
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../../test/gemma4e_dflash"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target test_gemma4e_dflash"

clean:
	rm -rf $(BUILD_DIR)


test: directories $(BUILD_DIR)/test_gemma4e_dflash.exe
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ${PWSH} -Command "\$$env:PATH = '..\..\..\lib;' + \$$env:PATH; .\test_gemma4e_dflash.exe -m nemotron3.5-bolt-dflash:e4b -t $(TYPE) -p 0"

.PHONY: all clean test directories

endif 
