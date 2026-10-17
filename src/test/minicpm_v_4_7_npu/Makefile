# =============================================================================
# MiniCPM-V NPU test Makefile
# =============================================================================
#
# Usage:
#   make        - Build the test
#   make test   - Build and run minicpm-v-4.7:1b (prompt, prompt-cache follow-up,
#                 fresh prompt with thinking)
#   make clean  - Remove all built files
#
# =============================================================================
-include ../common.mk

LENGTH ?= 256

SOURCES += test.cpp
SOURCES += ../../common/AutoModel/automodel.cpp
SOURCES += ../../common/AutoModel/model_backend.cpp
SOURCES += ../../common/AutoModel/modeling_minicpm_v.cpp
# Image support. modeling_minicpm_v_image.cpp holds the slicing, the PIL-exact
# resize and image_placeholder(); image_reader.cpp decodes the file (libjpeg-turbo/lodepng);
# imageprocAVX512.cpp supplies rescale_and_normalize_avx512(), and pulls in
# imageproc.cpp for resize_bicubic_* at link time.
# Same file list as the other VLM harnesses. MiniCPM does NOT call the shared
# resize -- it ships a PIL-exact one in modeling_minicpm_v_image.cpp, because the
# shared bicubic differs from PIL by up to 29 levels and SigLIP amplifies that
# more than every device rounding combined
# (notes/tricks/2026-10-05-siglip-amplifies-resize-differences.md in FLM_IRON).
# imageproc.cpp is still linked: imageprocAVX512.cpp references it regardless.
SOURCES += ../../common/AutoModel/modeling_minicpm_v_image.cpp
SOURCES += ../../common/image/image_reader.cpp
SOURCES += ../../include/lodepng/lodepng.cpp
SOURCES += ../../common/image_process_utils/imageproc.cpp
SOURCES += ../../common/image_process_utils/imageprocAVX512.cpp
SOURCES += ../../common/tokenizer/tokenizer.cpp
SOURCES += ../../common/modules/sampler.cpp

HEADERS += ../../include/models/minicpm_v_4_7/flm/aie2p/minicpm_v_4_7_npu.hpp
HEADERS += ../../include/AutoModel/modeling_minicpm_v.hpp


ifeq ($(WSL), 0)

LDFLAGS += -lminicpm_v_4_7_npu
# image_reader.cpp decodes through libjpeg-turbo (+ vendored lodepng), same as every other VLM harness
LDFLAGS += -ljpeg

CPP_SOURCES := $(filter %.cpp,$(SOURCES))
OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CPP_SOURCES:.cpp=.o)))
DEPS := $(OBJECTS:.o=.d)
VPATH := $(sort $(dir $(CPP_SOURCES)))

all: $(BUILD_DIR)/test_minicpm_v_4_7_npu

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_minicpm_v_4_7_npu: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(DEPENDENCY_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

test: $(BUILD_DIR)/test_minicpm_v_4_7_npu
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ./test_minicpm_v_4_7_npu --model minicpm-v-4.7:1b --Length $(LENGTH)

-include $(DEPS)
.PHONY: all clean test

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

all: directories test

directories:
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_minicpm_v_4_7_npu.exe: $(SOURCES)
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../../test/minicpm_v_4_7_npu"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target test_minicpm_v_4_7_npu"

clean:
	rm -rf $(BUILD_DIR)


test: directories $(BUILD_DIR)/test_minicpm_v_4_7_npu.exe
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ${PWSH} -Command "\$$env:PATH = '..\..\..\lib;' + \$$env:PATH; .\test_minicpm_v_4_7_npu.exe -m minicpm-v-4.7:1b -l $(LENGTH)"

.PHONY: all clean test directories

endif
