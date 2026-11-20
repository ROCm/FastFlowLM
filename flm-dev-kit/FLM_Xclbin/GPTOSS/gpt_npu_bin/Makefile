#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred

A ?= 1
C ?= 0
B ?= 0
L ?= 16
ifeq ($(shell uname -a | grep -i WSL),)
	WSL := 0
else
	WSL := 1
endif

ifeq ($(WSL), 0)
include makefiles/common.mk
MODEL_TYPE ?= GPTOSS_20B
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

KERNEL_ARGS := -DMODEL_TYPE=${MODEL_TYPE}
IRON_ARGS := ${MODEL_TYPE}


# Kernel makefile
include makefiles/kernel.mk

# Bitstream makefile
include makefiles/bitstream.mk

# Host makefile
include makefiles/host.mk

.PHONY: run all kernel link bitstream host clean clean_all instructions
all: ${XCLBIN_TARGET} ${INSTS_TARGET} ${HOST_C_TARGET}

clean_all:
	-@rm -rf build 
	-@rm -rf log
	-@rm -rf *.exe
	-@rm -rf trace*
	-@rm -rf routes



kernel: ${KERNEL_OBJS}


instructions: ${INSTS_TARGETS}


link: ${MLIR_TARGET} 


bitstream: ${XCLBIN_TARGETS}


host: ${HOST_C_TARGET}


clean:
	-@rm -rf build/host
	-@rm -rf ${HOST_C_TARGET}



run: $(HOST_C_TARGET)
	export LD_LIBRARY_PATH=../../libraries/build/lib && ./$(HOST_C_TARGET)

route: ${IRON_BOTH_MLIR_TARGET}
	mkdir -p routes
	aie-opt --aie-create-pathfinder-flows --aie-find-flows ${<} | aie-translate --aie-flows-to-json > ./routes/route.json
	python visualize.py -j ./routes/route.json

else

.PHONY: directories  run host all clean

PWSH := powershell.exe

BUILD_DIR := build/gpt_oss_npu


TARGET = $(BUILD_DIR)/test_gpt_oss_npu.exe

directories:
	mkdir -p $(BUILD_DIR)

host: $(TARGET)
	

all: directories $(TARGET)


run: $(TARGET)
	cp ../../libraries/build/lib/*.dll $(BUILD_DIR)
	cp ../../libraries/external_libs/*.dll $(BUILD_DIR)
	cd $(BUILD_DIR) && ${PWSH} -Command ".\test_gpt_oss_npu.exe -n $(A) -c $(C) -l $(L) -b $(B)"

clean:
	rm -rf $(BUILD_DIR)

$(TARGET): directories
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../.."
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target test_gpt_oss_npu"
	

endif
