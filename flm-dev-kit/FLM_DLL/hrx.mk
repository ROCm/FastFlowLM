# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Shared HRX build flags for the FLM engine libraries (HRX runtime path).
#
# On the HRX path the engines call the public hrx_amdxdna_executable_create()
# API (hrx_amdxdna.h) and use only HRX public headers. This consumes HRX from
# its *public package* (public headers + a single libhrx.so + a CMake package).
# It mirrors FLM_DLL/cmake/flm_hrx.cmake, which does the equivalent via
# find_package(hrx CONFIG REQUIRED) for the .dll build.
#
# The package is fetched+extracted by hrx-integration/fetch-hrx-release.sh (pin:
# hrx-integration/hrx-release.env). Provide the package prefix in one of:
#   * source the package's env.sh (exports HRX_HOME), or
#   * pass HRX_HOME=<package-prefix> on the make command line, or
#   * let this file auto-locate the extracted package under .hrx-release.

HRX_MK_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
REPO_ROOT  := $(abspath $(HRX_MK_DIR)/..)

# Directory fetch-hrx-release.sh extracts the pinned public package into.
HRX_RELEASE_DIR ?= $(REPO_ROOT)/hrx-integration/.hrx-release

# Package prefix (the dir containing include/, lib/, lib/cmake/hrx). Prefer
# HRX_HOME (set by the package env.sh / command line); otherwise auto-locate the
# extracted package by its libhrx.so (top-level or one directory deep).
ifeq ($(origin HRX_HOME),undefined)
HRX_HOME := $(patsubst %/lib/libhrx.so,%,$(firstword $(wildcard \
              $(HRX_RELEASE_DIR)/lib/libhrx.so \
              $(HRX_RELEASE_DIR)/*/lib/libhrx.so)))
endif

# Fail early with a clear message if the public package was not found. Skipped
# for `clean`.
ifeq ($(filter clean,$(MAKECMDGOALS)),)
ifeq ($(wildcard $(HRX_HOME)/lib/libhrx.so),)
$(error HRX public package not found (HRX_HOME="$(HRX_HOME)"). Run hrx-integration/fetch-hrx-release.sh (or `source <package>/env.sh`, or pass HRX_HOME=<package-prefix>) before building.)
endif
endif

# Public headers install under include/hrx (hrx_amdxdna.h, hrx_runtime.h, ...);
# include/ is added too so both spellings resolve. -DFLM_USE_HRX selects the
# hrx runtime in the shared FLM sources (flm_rt alias, gated npu_utils/buffer).
HRX_INC := -DFLM_USE_HRX \
           -I$(HRX_HOME)/include/hrx \
           -I$(HRX_HOME)/include

# Link the single public runtime library.
HRX_LIBS := -L$(HRX_HOME)/lib -lhrx

# Relocatable rpath: $ORIGIN first so a shipped engine finds libhrx next to
# itself (or via the FLM package RPATH). The package prefix is a link-time
# fallback for in-tree builds; do not rely on the container /work path.
HRX_RPATH := -Wl,-rpath,'$$ORIGIN' \
             -Wl,-rpath,'$$ORIGIN/../lib' \
             -Wl,-rpath,$(HRX_HOME)/lib \
             -Wl,-rpath,$(HRX_HOME)/lib/rocm_sysdeps/lib \
             -Wl,-rpath-link,$(HRX_HOME)/lib \
             -Wl,-rpath-link,$(HRX_HOME)/lib/rocm_sysdeps/lib
