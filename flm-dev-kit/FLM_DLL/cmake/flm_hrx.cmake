# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# flm_hrx.cmake — HRX runtime configuration for the engine .dll/.so build.
#
# On the HRX path the engines call the public hrx_amdxdna_executable_create()
# API declared in hrx_amdxdna.h and use only HRX public headers. This consumes
# HRX from its public CMake package via find_package(hrx).
#
# Point CMake at the public HRX release package via its CMake package dir, e.g.:
#   -DCMAKE_PREFIX_PATH=<hrx-public-*-x86_64>   (or source the package env.ps1/env.sh)
#
# Exposes to the including CMakeLists (interface unchanged, so the per-engine
# CMakeLists.txt files that reference these variables keep working):
#   HRX_INCLUDE_DIRS  - header search paths (empty: carried by hrx::hrx usage reqs)
#   HRX_LIBRARIES     - libraries to link (the hrx::hrx imported target)

include_guard(GLOBAL)

# Resolve the public HRX CMake package.
#
# 1) Prefer whatever the caller provided (CMAKE_PREFIX_PATH / hrx_DIR / the
#    CMAKE_PREFIX_PATH environment variable). This is how a from-source HRX
#    build is consumed: configure with -DCMAKE_PREFIX_PATH=<hrx-install>.
# 2) Otherwise fall back to the pinned public release that
#    hrx-integration/fetch-hrx-release.{sh,ps1} extracts under
#    hrx-integration/.hrx-release. The per-engine build drives cmake via
#    `powershell.exe -Command "cmake ../../detail/<engine>"` (see the WSL branch
#    of each detail/*/Makefile), which cannot pass -DCMAKE_PREFIX_PATH, so we
#    auto-locate the package here. This mirrors the Linux auto-locate already in
#    flm-dev-kit/FLM_DLL/hrx.mk.
find_package(hrx CONFIG QUIET)
if(NOT hrx_FOUND)
    file(GLOB _flm_hrx_cfgs
        "${CMAKE_CURRENT_LIST_DIR}/../../../hrx-integration/.hrx-release/*/lib/cmake/hrx/hrx-config.cmake"
        "${CMAKE_CURRENT_LIST_DIR}/../../../hrx-integration/.hrx-release/*/lib/cmake/hrx/hrxConfig.cmake"
        "${CMAKE_CURRENT_LIST_DIR}/../../../hrx-integration/.hrx-release/lib/cmake/hrx/hrx-config.cmake"
        "${CMAKE_CURRENT_LIST_DIR}/../../../hrx-integration/.hrx-release/lib/cmake/hrx/hrxConfig.cmake")
    if(_flm_hrx_cfgs)
        list(GET _flm_hrx_cfgs 0 _flm_hrx_cfg)
        get_filename_component(_flm_hrx_cfgdir "${_flm_hrx_cfg}" DIRECTORY)
        set(hrx_DIR "${_flm_hrx_cfgdir}")
        message(STATUS "flm_hrx: using pinned HRX release package at ${hrx_DIR}")
    endif()
    find_package(hrx CONFIG REQUIRED)
endif()

# The published package exports RelWithDebInfo only. Map that onto the other
# CMake configurations for this target alone.
set_target_properties(hrx::hrx PROPERTIES
    MAP_IMPORTED_CONFIG_RELEASE RelWithDebInfo
    MAP_IMPORTED_CONFIG_RELWITHDEBINFO RelWithDebInfo
    MAP_IMPORTED_CONFIG_DEBUG RelWithDebInfo
    MAP_IMPORTED_CONFIG_MINSIZEREL RelWithDebInfo)
# The runtime archive's bin/ also contains its own flm. Keep find_program
# from selecting that binary; hrx::hrx already has an absolute location.
get_target_property(_flm_hrx_loc hrx::hrx IMPORTED_LOCATION_RELWITHDEBINFO)
if(_flm_hrx_loc)
    get_filename_component(_flm_hrx_dir "${_flm_hrx_loc}" DIRECTORY)
    get_filename_component(_flm_hrx_root "${_flm_hrx_dir}" DIRECTORY)
    list(APPEND CMAKE_IGNORE_PATH "${_flm_hrx_root}/bin")
endif()

# Windows: FLM sources include <windows.h> (e.g. utils.hpp) and also use
# std::min/std::max. The public hrx_runtime.h does not define NOMINMAX, so
# windows.h's min/max macros would break std::min/std::max. Define NOMINMAX
# (+ lean headers) globally for every engine target that includes this file.
# Linux is unaffected.
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Select the hrx runtime in the shared FLM sources (flm_rt alias, gated
# npu_utils/buffer). Every engine target that includes this file (the HRX
# branch of its CMakeLists) is compiled with -DFLM_USE_HRX.
add_compile_definitions(FLM_USE_HRX)

# The public hrx::hrx imported target carries its own INTERFACE include
# directories (hrx_amdxdna.h, hrx_runtime.h, ...), so HRX_INCLUDE_DIRS is
# intentionally empty; keeping the name avoids touching every engine CMakeLists.
set(HRX_INCLUDE_DIRS "")
set(HRX_LIBRARIES hrx::hrx)
