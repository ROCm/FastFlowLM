# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Settings shared by the Linux engine Makefiles (detail/*/Makefile and the
# helper Makefiles under them). Each includes this file before it sets
# CXX_FLAGS. Override either variable on the make command line or in the
# environment.
#
# FLM_MARCH  CPU instruction-set flags. The default is the explicit AVX-512 set
#            FastFlowLM's own engine build uses (src/detail/CMakeLists.txt), so
#            a library built on one machine runs on any AVX-512 CPU; gemma,
#            gpt_oss and whisper also need AVX512-BF16, and qwen3_asr needs
#            BF16, VNNI and VBMI (all Zen 4+ Ryzen AI hosts have them; those
#            Makefiles add the flags). For a build tuned to the build host,
#            which then may fault elsewhere:
#                make FLM_MARCH=-march=native
# XRT_ROOT   XRT prefix holding include/xrt and lib/libxrt_coreutil.so.
#            Default /opt/xilinx/xrt (the XRT installer). When that is absent
#            and the distribution's libxrt-dev is installed (headers in
#            /usr/include/xrt, as in FastFlowLM's build-environment container),
#            /usr is used. Not read on the HRX path (FLM_USE_HRX=1, hrx.mk).

FLM_MARCH ?= -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma

ifeq ($(origin XRT_ROOT),undefined)
ifeq ($(wildcard /opt/xilinx/xrt/include/xrt/xrt_bo.h),)
ifneq ($(wildcard /usr/include/xrt/xrt_bo.h),)
XRT_ROOT := /usr
endif
endif
endif
XRT_ROOT ?= /opt/xilinx/xrt
