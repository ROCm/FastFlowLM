# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
#
# Inkling-2-Mini decode overlay. make kernel | bitstream | host

include makefiles/common.mk
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include makefiles/kernel.mk
include makefiles/bitstream.mk
include makefiles/host.mk

.PHONY: all kernel bitstream host clean clean_all
all: bitstream host
kernel: ${KERNEL_OBJS}
bitstream: ${XCLBIN_TARGETS}
host: ${HOST_C_TARGET}

clean:
	-@rm -rf build/host ${HOST_C_TARGET}

clean_all:
	-@rm -rf build *.exe
