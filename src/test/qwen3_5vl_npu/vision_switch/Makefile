# Vision encoder timing for Qwen3.5-2B (mm <-> attention switch cost). See README.md.
#   make         - build the timed engine copy and vision_bench
#   make test    - run it (normal, then VISION_PROBE=1); log in $(BUILD_DIR)/vision.log
#   make compare - two-xclbin path vs the vision overlay: encoder output must be bit-identical

SRC := $(abspath ../../..)
BUILD_DIR := $(SRC)/build/test/vision_switch
ENGINE_DIR := $(BUILD_DIR)/engine
FLM_MODEL_PATH ?= /scratch/michyu
OVERLAY_XCLBIN ?= /scratch/michyu/FastFlowLM_IRON/FLM_Xclbin/Qwen3_5_2B/vision_overlay/build/xclbins/vision_overlay.xclbin
ITERS ?= 3
SIZES ?= 256x256 640x480 1024x768 1920x1080
IMAGES := $(foreach s,$(SIZES),$(BUILD_DIR)/img_$(s).png)
COMMA := ,
EMPTY :=
SPACE := $(EMPTY) $(EMPTY)

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall -DUSEAVX2=1 -mavx2 -mfma -march=native -ffast-math -fmax-errors=1
CXX_FLAGS += -I$(SRC)/include -I/opt/xilinx/xrt/include -MMD -MP -DDEV_BUILD -fopenmp
CXX_FLAGS += -DCMAKE_INSTALL_PREFIX="\"/opt/fastflowlm\"" -DCMAKE_XCLBIN_PREFIX="\"/opt/fastflowlm/share/flm/xclbins\""
CXX_FLAGS += -D__FLM_VERSION__="\"0.9.34\"" -D__NPU_VERSION__="\"32.0.203.304\"" -DDISABLE_ABI_CHECK=1
CXX_FLAGS += -I$(SRC)/../third_party/xgrammar/include -isystem $(SRC)/../third_party/xgrammar/3rdparty/dlpack/include
CXX_FLAGS += -DXGRAMMAR_ENABLE_CPPTRACE=0

LIB_DIR := $(SRC)/lib/xrt
LDFLAGS := -lboost_program_options -lboost_filesystem -L$(LIB_DIR) -Wl,-rpath,$(LIB_DIR)
LDFLAGS += -L$(SRC)/build/tokenizers-cpp -L$(SRC)/build/tokenizers-cpp/sentencepiece/src
LDFLAGS += -ltokenizers_cpp -ltokenizers_c -lsentencepiece
LDFLAGS += -L/opt/xilinx/xrt/lib -Wl,-rpath,/opt/xilinx/xrt/lib -lxrt_coreutil -laiebu
LDFLAGS += -lqwen3_5vl_npu -ljpeg -L$(SRC)/build/libwebp/lib -lwebpdecoder
LDFLAGS += -lmha -ldequant -lgemm -llm_head -lq4_npu_eXpress -L$(SRC)/build -lflm_xgrammar

SOURCES := vision_bench.cpp
SOURCES += $(SRC)/common/utils.cpp
SOURCES += $(SRC)/common/AutoModel/automodel.cpp
SOURCES += $(SRC)/common/AutoModel/model_backend.cpp
SOURCES += $(SRC)/common/prefix_slot_cache.cpp
SOURCES += $(wildcard $(SRC)/common/chat/*.cpp)
SOURCES += $(SRC)/common/AutoModel/modeling_qwen3_5vl.cpp
SOURCES += $(SRC)/common/AutoModel/modeling_qwen3_5vl_image.cpp
SOURCES += $(SRC)/common/image_process_utils/imageproc.cpp
SOURCES += $(SRC)/common/image_process_utils/imageprocAVX512.cpp
SOURCES += $(SRC)/common/image/image_reader.cpp
SOURCES += $(SRC)/include/lodepng/lodepng.cpp
SOURCES += $(SRC)/common/tokenizer/tokenizer.cpp
SOURCES += $(SRC)/common/modules/sampler.cpp

OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(SOURCES:.cpp=.o)))
VPATH := $(sort $(dir $(SOURCES)))

all: $(BUILD_DIR)/vision_bench engine

engine:
	$(MAKE) -C engine BUILD_DIR=$(ENGINE_DIR) EXTRA_FLAGS=

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/vision_bench: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/img_%.png: make_image.py
	python3 make_image.py $* $@

RUN := cd $(BUILD_DIR) && cp $(SRC)/model_list.json . && \
	export LD_LIBRARY_PATH=$(ENGINE_DIR):$(LIB_DIR) FLM_MODEL_PATH=$(FLM_MODEL_PATH)

test: all $(IMAGES)
	$(RUN) && ./vision_bench $(subst $(SPACE),$(COMMA),$(IMAGES)) $(ITERS) > vision.log 2>&1
	$(RUN) && VISION_PROBE=1 ./vision_bench $(subst $(SPACE),$(COMMA),$(IMAGES)) $(ITERS) >> vision.log 2>&1
	grep -E "RESULT|answer" $(BUILD_DIR)/vision.log
	python3 summarize.py $(BUILD_DIR)/vision.log

compare: all $(IMAGES)
	mkdir -p $(BUILD_DIR)/dumps
	rm -f $(BUILD_DIR)/dumps/*.bin
	$(RUN) && VISION_DUMP=dumps/ref ./vision_bench $(subst $(SPACE),$(COMMA),$(IMAGES)) $(ITERS) > ref.log 2>&1
	$(RUN) && VISION_DUMP=dumps/ovl VISION_OVERLAY=$(OVERLAY_XCLBIN) ./vision_bench $(subst $(SPACE),$(COMMA),$(IMAGES)) $(ITERS) > ovl.log 2>&1
	cd $(BUILD_DIR)/dumps && for f in ref_*.bin; do cmp -s $$f ovl_$${f#ref_} && echo "$${f#ref_}: bit-identical" || echo "$${f#ref_}: DIFFERS"; done
	grep -E "RESULT|answer" $(BUILD_DIR)/ref.log $(BUILD_DIR)/ovl.log

clean:
	rm -rf $(BUILD_DIR)

-include $(OBJECTS:.o=.d)
.PHONY: all engine test compare clean
