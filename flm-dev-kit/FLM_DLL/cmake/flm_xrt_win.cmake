# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# flm_xrt_win.cmake — XRT and aiebu locations for the Windows engine build.
#
# The default layout is the one FastFlowLM's own Windows build uses
# (src/CMakeLists.txt) and that .github/script/ci/setup_windows_deps.ps1 creates:
#   <FLM_WIN_DEPS_ROOT>/XRT/src/runtime_src/core/include   XRT headers
#   <FLM_WIN_DEPS_ROOT>/xrtNPUfromDLL/xrt_coreutil.lib      import lib of the driver's DLL
# Override the root with -DFLM_WIN_DEPS_ROOT=<dir>, or each path with
# -DXRT_INCLUDE_DIR=<dir> -DXRT_LIB_DIR=<dir>.
#
# aiebu_static.lib is the copy FastFlowLM ships in src/lib/xrt/.
#
# Boost headers: XRT's xrt/detail/any.h picks boost::any over std::any when
# __cplusplus reads below 201703L, and MSVC reports 199711L unless given
# /Zc:__cplusplus. The engines build like flm (no /Zc:__cplusplus), so they need
# boost/any.hpp; nothing links a Boost library. Same lookup as src/CMakeLists.txt.
#
# Exposes: XRT_INCLUDE_DIR, XRT_LIB_DIR, FLM_AIEBU_LIB_DIR, FLM_BOOST_INCLUDE_DIR.
# Set FLM_XRT_WIN_XRT_ONLY to ON before the include() for the XRT paths alone:
# no aiebu or Boost-header check (the aie2p-launchlib projects use neither).

include_guard(GLOBAL)

set(FLM_WIN_DEPS_ROOT "C:/dev" CACHE PATH "Root of the Windows XRT dependency layout")
set(XRT_INCLUDE_DIR "${FLM_WIN_DEPS_ROOT}/XRT/src/runtime_src/core/include" CACHE PATH "Where XRT headers live")
set(XRT_LIB_DIR     "${FLM_WIN_DEPS_ROOT}/xrtNPUfromDLL"                    CACHE PATH "Where XRT libs live")
get_filename_component(FLM_AIEBU_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../src/lib/xrt" ABSOLUTE)

if(NOT EXISTS "${XRT_INCLUDE_DIR}/xrt/xrt_bo.h" OR NOT EXISTS "${XRT_LIB_DIR}/xrt_coreutil.lib")
    message(FATAL_ERROR
        "XRT not found (XRT_INCLUDE_DIR=${XRT_INCLUDE_DIR}, XRT_LIB_DIR=${XRT_LIB_DIR}). "
        "Run FastFlowLM's .github/script/ci/setup_windows_deps.ps1, or pass "
        "-DFLM_WIN_DEPS_ROOT=<dir> or -DXRT_INCLUDE_DIR=<dir> -DXRT_LIB_DIR=<dir>.")
endif()
if(FLM_XRT_WIN_XRT_ONLY)
    return()
endif()
if(NOT EXISTS "${FLM_AIEBU_LIB_DIR}/aiebu_static.lib")
    message(FATAL_ERROR "aiebu_static.lib not found in ${FLM_AIEBU_LIB_DIR} (FastFlowLM's src/lib/xrt/).")
endif()
find_path(FLM_BOOST_INCLUDE_DIR NAMES boost/any.hpp
    HINTS "${FLM_WIN_DEPS_ROOT}/boost_1_88_0" "$ENV{CONDA_PREFIX}/Library/include")
if(NOT FLM_BOOST_INCLUDE_DIR)
    message(FATAL_ERROR "boost/any.hpp not found (needed by XRT's headers under MSVC). "
                        "Run setup_windows_deps.ps1, or pass -DFLM_BOOST_INCLUDE_DIR=<boost headers>.")
endif()
