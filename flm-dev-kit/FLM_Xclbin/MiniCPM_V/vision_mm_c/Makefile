#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
# Modified by Alfred

include makefiles/common.mk
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# One design: VISION_MM_C, 64 x 384 x 96 per core, 512 x 384 per array pass
# (columns split M, rows split N); see kernel/typedef.hpp and iron/mm_config.py.
# It is written out under the name the engine opens it by, which is also the
# name the kernel map (FLM_Xclbin/kernel_assignment.md) lists: keep this table,
# kernel_assignment.md and minicpm_v_4_7_npu.cpp in sync.
MODEL_TYPE ?= VISION_MM_C
DESIGNS := VISION_MM_C
DEPLOY_NAME_VISION_MM_C := vision_mm_c.xclbin
DEPLOY_NAME := $(DEPLOY_NAME_$(MODEL_TYPE))
ifeq ($(DEPLOY_NAME),)
$(error Unknown MODEL_TYPE '$(MODEL_TYPE)' -- expected one of: $(DESIGNS))
endif

KERNEL_ARGS := -DMODEL_TYPE=${MODEL_TYPE} $(EXTRA_KARGS)
IRON_ARGS := ${MODEL_TYPE}

ITERATION ?= 1

# Kernel makefile
include makefiles/kernel.mk

# Bitstream makefile
include makefiles/bitstream.mk

# Host makefile
include makefiles/host.mk

.PHONY: run all kernel link bitstream bitstream_all host clean clean_all instructions both rtp_addr
all: ${XCLBIN_TARGET} ${INSTS_TARGET} ${HOST_C_TARGET}

# Every design's bitstream. Each recurses with its own MODEL_TYPE so kernel
# objects and bitstreams land in separate build/<design>/ trees - note make does
# not track the -D flag itself, so the per-design build directory is what keeps
# a stale object from being re-linked. The recursion passes MODEL_TYPE on the
# command line, which is what stops `bitstream` below from looping back here.
bitstream_all:
	@for d in $(DESIGNS); do \
	    echo "=== building $$d ==="; \
	    $(MAKE) --no-print-directory MODEL_TYPE=$$d bitstream || exit 1; \
	    ./tools/extract_rtp_addr.sh $$d || exit 1; \
	done

# The bitstream and the host in one go.
both: $(HOST_C_TARGET)
	@$(MAKE) --no-print-directory bitstream_all

# Re-read the compiler-assigned RTP address for each design. Must be re-run after
# any design change: a stale address does not fault, it hangs in wait().
rtp_addr:
	@for d in $(DESIGNS); do ./tools/extract_rtp_addr.sh $$d || exit 1; done

clean_all:
	-@rm -rf build 
	-@rm -rf log
	-@rm -rf *.exe
	-@rm -rf trace*
	-@rm -rf routes



kernel: ${KERNEL_OBJS}


instructions: ${INSTS_TARGETS}


link: ${MLIR_TARGET} 


# A bare `make bitstream` builds every design (there is one); naming a design
# builds just that one:
#
#     make bitstream MODEL_TYPE=VISION_MM_C
#
# That is also how bitstream_all recurses, and how CI drives a single design
# when it wants one. Checking $(origin) rather than emptiness is deliberate:
# MODEL_TYPE is never empty here, it always has the ?= default.
ifneq ($(filter command line environment,$(origin MODEL_TYPE)),)
bitstream: ${XCLBIN_TARGETS}
else
bitstream: bitstream_all
endif


host: ${HOST_C_TARGET}


clean:
	-@rm -rf build/host
	-@rm -rf ${HOST_C_TARGET}


test: $(HOST_C_TARGET)
	export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib && ./$(HOST_C_TARGET)

route: ${IRON_BOTH_MLIR_TARGET}
	mkdir -p routes
	aie-opt --aie-create-pathfinder-flows --aie-find-flows ${<} | aie-translate --aie-flows-to-json > ./routes/route.json
	python visualize.py -j ./routes/route.json

trace: ${HOST_C_TARGET}
	${powershell} ./$< 
	~/mlir-aie/programming_examples/utils/parse_trace.py --input trace.txt --mlir build/mlir/*.mlir --colshift 0 --output trace.json
	~/mlir-aie/programming_examples/utils/get_trace_summary.py --input trace.json