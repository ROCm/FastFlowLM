cmake_minimum_required(VERSION 3.22)
project(flm LANGUAGES CXX VERSION 0.1.0)

if(POLICY CMP0207)
    cmake_policy(SET CMP0207 NEW)
endif()

if(NOT DEFINED FLM_VERSION)
    message(FATAL_ERROR "FLM_VERSION must be specified externally. Use -DFLM_VERSION=<version> when running cmake.")
endif()

if(NOT DEFINED NPU_VERSION)
    message(FATAL_ERROR "NPU_VERSION must be specified externally. Use -DNPU_VERSION=<version> when running cmake.")
endif()

# Set build type to Release
set(CMAKE_BUILD_TYPE Release)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure CMake to stop at first error
set(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION ON)
set(CMAKE_ERROR_DEPRECATED ON)


# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


# Force output directories to be absolute
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# ———————————————————————————————————————————————
# NPU runtime backend selection.
#
# FLM ships two interchangeable NPU dispatch backends:
#   * XRT  — the Xilinx Run Time (the historical default), and
#   * HRX  — the HRX amdxdna runtime.
#
# The reviewer-requested build flag selects between them (0 = XRT, 1 = HRX) with
# XRT as the default, so the XRT build is unchanged and HRX is strictly opt-in.
# FLM_USE_HRX=OFF selects XRT; FLM_USE_HRX=ON selects HRX. The choice drives:
#   - which runtime headers/namespace alias the sources see (FLM_USE_HRX define),
#   - which prebuilt engine libs are consumed (lib/xrt vs lib/hrx), and
#   - which runtime is discovered, linked and bundled below.
# ———————————————————————————————————————————————
option(FLM_USE_HRX "Use the HRX amdxdna NPU runtime instead of XRT (0=XRT default, 1=HRX)" OFF)
option(FLM_PORTABLE_BUILD "Build portable distribution with bundled runtime libraries" OFF)
# Where the kernels come from. FastFlowLM's own flow and ryzenai-corelib share
# no engine, so a build links one or the other; this option is not an extra.
# It also fixes the NPU generation the build targets -- see utils/npu_platform.hpp.
option(FLM_ENABLE_CORE
       "Reach kernels through ryzenai-corelib instead of FastFlowLM's own flow" OFF)

if(FLM_ENABLE_CORE)
    # The corelib prefix lives outside this repository, so its headers have to
    # be found rather than assumed. The hints mirror the library lookup below:
    # a header tree vendored into include/ wins, then a prefix named by
    # RYZENAI_CORELIB_ROOT, and failing both the standard CMake search still
    # honours CMAKE_PREFIX_PATH. Without the hints every configure has to be
    # handed a prefix on the command line, which is the only reason the core
    # build ever needed an environment variable.
    find_path(RYZENAI_CORELIB_INCLUDE_DIR NAMES ryzenai/corelib.h
        HINTS "${CMAKE_SOURCE_DIR}/include"
              "${RYZENAI_CORELIB_ROOT}/include"
              "$ENV{RYZENAI_CORELIB_ROOT}/include" REQUIRED)
    # corelib is linked in rather than dlopened.
    # A conda/pip prefix keeps its import libs beside the headers; a CMake
    # install prefix puts them in lib/ or lib64/ depending on the distribution,
    # so both are offered rather than assuming either.
    # lib/ is where this repository already keeps the binaries it vendors, so a
    # corelib dropped in beside them is found without anyone passing a path.
    # It is searched first: a build that has deliberately been given a corelib
    # should use that one rather than whatever an unrelated prefix happens to
    # hold.
    find_library(RYZENAI_CORELIB_LIBRARY NAMES ryzenai_corelib corelib
        HINTS "${CMAKE_SOURCE_DIR}/lib"
              "${RYZENAI_CORELIB_INCLUDE_DIR}/../lib"
              "${RYZENAI_CORELIB_INCLUDE_DIR}/../lib64"
              "$ENV{RYZENAI_CORELIB_LIB_DIR}" REQUIRED)
    # XRT's xrt/detail/any.h selects boost::any over std::any whenever
    # __cplusplus reads below 201703L. MSVC reports 199711L regardless of /std
    # unless it is handed /Zc:__cplusplus, so a Windows build has to supply
    # Boost headers that nothing else on the core path wants. GCC and Clang
    # report C++20 honestly, so that branch is never taken there and there is
    # nothing to find -- which is why this is a Windows-only lookup rather than
    # a dependency of the core backend.
    if(WIN32)
        find_path(FLM_CORELIB_BOOST_INCLUDE_DIR NAMES boost/any.hpp
            HINTS "$ENV{CONDA_PREFIX}/Library/include"
                  "$ENV{USERPROFILE}/anaconda3/Library/include"
                  "C:/dev/boost_1_88_0" REQUIRED)
    endif()
endif()

if(FLM_USE_HRX)
    set(FLM_RUNTIME_NAME "hrx")
    message(STATUS "FLM NPU runtime backend: HRX (FLM_USE_HRX=ON)")
else()
    set(FLM_RUNTIME_NAME "xrt")
    message(STATUS "FLM NPU runtime backend: XRT (FLM_USE_HRX=OFF, default)")
endif()

# Prebuilt engine libraries live in a per-backend subdirectory so the XRT-built
# and HRX-built .so/.dll/.lib never collide (see src/lib/xrt and src/lib/hrx).
set(FLM_ENGINE_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/${FLM_RUNTIME_NAME}")

# The flash engine libs (qwen3vl_flash, gemma4e_flash) are always shipped in
# src/lib/${FLM_RUNTIME_NAME} for every backend, so their model families are
# always compiled in -- no probe/gate/macro needed.
# Gemma4e can be built from source here; the other engines stay prebuilt.
option(FLM_BUILD_GEMMA4E "Build the Gemma4e engine from src/detail instead of using the prebuilt" OFF)
option(FLM_ENGINE_NATIVE_ARCH "Build the from-source engine with -march=native (host-specific, not redistributable)" OFF)
set(FLM_ENGINE_VERBOSE 0 CACHE STRING "VERBOSE level for the from-source engine")
set(FLM_ENGINE_DEBUG_LEVEL 0 CACHE STRING "DEBUG_LEVEL for the from-source engine")
set(FLM_OVERRIDE_FLAGS "" CACHE STRING "Extra compile flags for the from-source engine (see include/flm_override.hpp)")

# ———————————————————————————————————————————————
# NPU runtime discovery.
#   HRX: consumed from its public CMake package via find_package(hrx).
#   XRT: discovered via pkg-config (system install), a manual /opt/xilinx/xrt
#        fallback, or fetched from source for a portable build.
# ———————————————————————————————————————————————
if(FLM_USE_HRX)
    find_package(hrx CONFIG REQUIRED)
    # The published package exports RelWithDebInfo only. Map that onto the
    # other Visual Studio configurations for this target alone so the vcpkg
    # packages keep their own Release/Debug libraries.
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
else()
    if(WIN32)
        set(XRT_INCLUDE_DIR C:/dev/XRT/src/runtime_src/core/include CACHE PATH "Where XRT headers live")
        set(XRT_LIB_DIR     C:/dev/xrtNPUfromDLL                    CACHE PATH "Where XRT libs live")
    else()
        find_package(PkgConfig)
        if(PkgConfig_FOUND)
            pkg_check_modules(XRT xrt)
            if(XRT_FOUND)
                message(STATUS "Found XRT via pkg-config")
                message(STATUS "  XRT include dirs: ${XRT_INCLUDE_DIRS}")
                message(STATUS "  XRT library dirs: ${XRT_LIBRARY_DIRS}")
                message(STATUS "  XRT libraries: ${XRT_LIBRARIES}")
            endif()
        endif()

        # Portable build: fetch XRT from source if it is not installed on the host.
        if(FLM_PORTABLE_BUILD AND NOT XRT_FOUND)
            message(STATUS "XRT not found on system; fetching from source for portable build")
            include(FetchContent)
            set(XRT_GIT_REPO "https://github.com/Xilinx/XRT.git" CACHE STRING "XRT Git repository URL")
            set(XRT_GIT_TAG "2.21.75" CACHE STRING "XRT Git tag/branch to fetch")
            FetchContent_Declare(xrt_source
                GIT_REPOSITORY ${XRT_GIT_REPO}
                GIT_TAG ${XRT_GIT_TAG}
                GIT_SHALLOW TRUE)
            set(XRT_NATIVE_BUILD ON CACHE BOOL "Native XRT build" FORCE)
            set(BUILD_DOCS OFF CACHE BOOL "Build documentation" FORCE)
            set(XRT_BASE 1 CACHE STRING "Build base XRT only" FORCE)
            set(XRT_ALVEO 0 CACHE STRING "Disable Alveo support" FORCE)
            set(XRT_BUILD_XBMGMT OFF CACHE BOOL "Build xbmgmt" FORCE)
            set(XRT_BUILD_XBUTIL OFF CACHE BOOL "Build xbutil" FORCE)
            set(XRT_BUILD_XBT OFF CACHE BOOL "Build xbt" FORCE)
            FetchContent_MakeAvailable(xrt_source)
            set(XRT_REPO_DIR ${xrt_source_SOURCE_DIR})
            set(XRT_BUILD_DIR ${xrt_source_BINARY_DIR})
            set(XRT_INCLUDE_DIRS ${XRT_REPO_DIR}/src/runtime_src/core/include)
            set(XRT_FOUND TRUE)
            set(XRT_BUILT_FROM_SOURCE TRUE)
            set(XRT_LIBRARY_DIRS ${XRT_BUILD_DIR}/src/runtime_src)
            message(STATUS "Note: XDNA plugin (libxrt_driver_xdna.so.2) must be provided separately")
        endif()

        # Manual fallback if pkg-config didn't find XRT and it wasn't fetched.
        if(NOT XRT_FOUND AND NOT XRT_BUILT_FROM_SOURCE)
            set(XRT_INCLUDE_DIR /opt/xilinx/xrt/include CACHE PATH "Where XRT headers live")
            set(XRT_LIB_DIR     /opt/xilinx/xrt/lib     CACHE PATH "Where XRT libs live")
        endif()
    endif()
endif()

# ———————————————————————————————————————————————
# Add tokenizers-cpp subproject
# ———————————————————————————————————————————————
# Ensure C++17 is available for tokenizers-cpp
if(CMAKE_CXX_STANDARD LESS 17)
    set(CMAKE_CXX_STANDARD 17)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/../third_party/tokenizers-cpp
                 ${CMAKE_BINARY_DIR}/tokenizers-cpp
                 EXCLUDE_FROM_ALL)

# JPEG/PNG decoding for ImageReader: static libjpeg-turbo + vendored lodepng.
include(${CMAKE_SOURCE_DIR}/cmake/FlmImageCodecs.cmake)
# Audio decoding for AudioReader: static libopus/libogg/libvorbis + vendored dr_libs.
include(${CMAKE_SOURCE_DIR}/cmake/FlmAudioCodecs.cmake)

# ———————————————————————————————————————————————
# xgrammar: the grammar engine behind constrained decoding (tool_choice,
# response_format), shared with the test harnesses. The chat sources themselves
# come in with the common/*/*.cpp glob below.
# ———————————————————————————————————————————————
include("${CMAKE_SOURCE_DIR}/common/chat/chat_sources.cmake")

# ———————————————————————————————————————————————
# Gather your sources
# ———————————————————————————————————————————————
file(GLOB SOURCES "src/*.cpp" "runner/*.cpp" "common/*.cpp" "common/*/*.cpp" "server/*.cpp" "pull/*.cpp" )
file(GLOB HEADERS "include/*.hpp" "runner/*.hpp" "common/*.hpp" "common/*/*.hpp" "server/*.hpp" "pull/*.hpp")
list(FILTER SOURCES EXCLUDE REGEX ".*/common/core/.*\\.cpp$")

# Model sources live two levels deeper than the globs above reach. Only the core
# half exists as loose sources; it is built separately, below.

# Exclude files that depend on missing libraries for Linux
if(NOT WIN32)
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma3\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma3_image\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma3_text\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gpt_oss\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_lfm2\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_phi4\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen2\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen3\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen3vl\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen3vl_image\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_whisper\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_whisper_audio\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma_embedding\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*auto_embedding_model\\.cpp$")

    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/common/image_process_utils/imageprocAVX512.cpp
        ${CMAKE_SOURCE_DIR}/common/audio_process_utils/audioprocAVX512.cpp
        PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512dq;-mavx512vl;-mavx512bw;-mfma"
    )

    # Define a macro to indicate limited model support on Linux
    # add_compile_definitions(FASTFLOWLM_LINUX_LIMITED_MODELS=1)
else()
    # Windows cl and clang-cl need the ISA named on these files, and clang-cl
    # ignores the GCC -m flags.
    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/common/image_process_utils/imageprocAVX512.cpp
        ${CMAKE_SOURCE_DIR}/common/audio_process_utils/audioprocAVX512.cpp
        PROPERTIES
        COMPILE_OPTIONS "/arch:AVX512"
    )
endif()


add_executable(flm ${SOURCES} ${HEADERS})

if(FLM_ENABLE_CORE)
    include("${CMAKE_SOURCE_DIR}/common/core/core_sources.cmake")
    add_library(flm_core STATIC ${FLM_CORE_SOURCES})
    target_include_directories(flm_core PUBLIC
        "${CMAKE_SOURCE_DIR}/include" "${RYZENAI_CORELIB_INCLUDE_DIR}")
    # XRT headers come from whichever discovery above succeeded: pkg-config sets
    # XRT_INCLUDE_DIRS, the Windows and /opt/xilinx fallbacks set the singular
    # XRT_INCLUDE_DIR. Same choice the flm target makes below -- naming only the
    # singular one here left flm_core with no XRT headers at all on Linux.
    if(NOT WIN32 AND XRT_FOUND)
        target_include_directories(flm_core PUBLIC ${XRT_INCLUDE_DIRS})
    else()
        target_include_directories(flm_core PUBLIC ${XRT_INCLUDE_DIR})
    endif()
    # Only Windows looks this one up; see the find_path above for why.
    if(FLM_CORELIB_BOOST_INCLUDE_DIR)
        target_include_directories(flm_core PUBLIC "${FLM_CORELIB_BOOST_INCLUDE_DIR}")
    endif()
    # RYZENAI_CORELIB_STATIC drops the vendor header's dllimport decoration;
    # FLM_CORELIB_LINK_STATIC is what tells our own code the symbols are linked
    # in and must be bound directly instead of through LoadLibrary/dlopen.
    target_compile_definitions(flm_core PUBLIC
        FLM_ENABLE_CORE=1 RYZENAI_CORELIB_STATIC=1 FLM_CORELIB_LINK_STATIC=1)
    # CMAKE_DL_LIBS is -ldl on Linux and empty on Windows: corelib_api.cpp still
    # compiles its dlopen path in a statically linked build, so the symbols have
    # to resolve even though nothing calls them here.
    target_link_libraries(flm_core PUBLIC "${RYZENAI_CORELIB_LIBRARY}" ${CMAKE_DL_LIBS})
    target_link_libraries(flm PRIVATE flm_core)
    # corelib loads each ryzenai_corelib_model_<name>.dll, which carries that
    # model's kernels, only from the directory ryzenai_corelib.dll itself was
    # loaded from, so both have to land beside flm.exe.
    if(WIN32)
        get_filename_component(FLM_CORELIB_LIB_DIR "${RYZENAI_CORELIB_LIBRARY}" DIRECTORY)
        file(GLOB FLM_CORELIB_RUNTIME_DLLS
            "${FLM_CORELIB_LIB_DIR}/../bin/ryzenai_corelib.dll"
            "${FLM_CORELIB_LIB_DIR}/../bin/ryzenai_corelib_model_*.dll")
        list(FILTER FLM_CORELIB_RUNTIME_DLLS EXCLUDE REGEX "ryzenai_corelib_model_dummy\\.dll$")
        if(FLM_CORELIB_RUNTIME_DLLS)
            add_custom_command(TARGET flm POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${FLM_CORELIB_RUNTIME_DLLS} $<TARGET_FILE_DIR:flm>)
        else()
            message(WARNING "No ryzenai_corelib DLLs under ${FLM_CORELIB_LIB_DIR}/../bin; "
                            "copy them beside flm.exe by hand")
        endif()
    endif()
endif()

# ———————————————————————————————————————————————
# Opt-in Windows dependency prefix.
#
# Windows otherwise has exactly two routes for Boost/curl: a vcpkg
# toolchain, or the bare CI runner's hardcoded C:/dev paths and CI-specific
# library names. A developer box with neither can point this at one prefix
# laid out as <prefix>/include + <prefix>/lib -- in practice a conda env's
# Library dir (conda-forge boost, libcurl) -- and each block that
# applies the CI runner's include dirs, definitions, link dirs and library
# names switches to it.
# Empty (the default) leaves both existing routes exactly as they were.
# ———————————————————————————————————————————————
set(FLM_WIN_DEPS_PREFIX "" CACHE PATH
    "Windows only: take Boost/curl from <prefix>/include and <prefix>/lib (e.g. a conda env's Library dir) instead of the CI runner's C:/dev paths")
if(FLM_WIN_DEPS_PREFIX)
    if(NOT WIN32)
        message(FATAL_ERROR "FLM_WIN_DEPS_PREFIX is a Windows-only route")
    endif()
    if(VCPKG_TOOLCHAIN)
        message(FATAL_ERROR "FLM_WIN_DEPS_PREFIX and a vcpkg toolchain are two "
                            "different dependency routes; use one")
    endif()
    # The library names a conda-forge Library dir ships: unversioned Boost and
    # curl import libs (both are DLL builds).
    set(FLM_WIN_DEPS_PREFIX_LIBS
        libcurl boost_program_options)
    foreach(_flm_dep IN LISTS FLM_WIN_DEPS_PREFIX_LIBS)
        if(NOT EXISTS "${FLM_WIN_DEPS_PREFIX}/lib/${_flm_dep}.lib")
            message(FATAL_ERROR "FLM_WIN_DEPS_PREFIX: missing "
                                "${FLM_WIN_DEPS_PREFIX}/lib/${_flm_dep}.lib")
        endif()
    endforeach()
    message(STATUS "Windows deps from prefix: ${FLM_WIN_DEPS_PREFIX}")
endif()

if(WIN32)
    if(VCPKG_TOOLCHAIN)
        # A vcpkg toolchain is active (e.g. the rocm-npu-staging dev.py build or
        # a local CMakePresets build): the vcpkg tree ships CMake package configs,
        # so resolve the native deps in CONFIG mode. find_package auto-detects the
        # correct versioned import-lib names, so this works regardless of the
        # installed Boost version.
        find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
        find_package(CURL CONFIG REQUIRED)
    endif()
    # A third, opt-in route (FLM_WIN_DEPS_PREFIX, declared above) takes Boost
    # and curl from one local prefix, wired next to each CI-runner block.
    # Otherwise (the bare self-hosted CI runner) there is no vcpkg toolchain and
    # no Boost in vcpkg: Boost is a standalone b2 build under C:/dev/boost_1_88_0
    # and curl is linked by raw name from the vcpkg lib dir. That
    # path is wired via the include/link dirs and raw library names guarded by
    # NOT VCPKG_TOOLCHAIN below.
else()
    find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
    # The Linux build environment installs these via apt, which provides the
    # CMake FindCURL module and pkg-config files rather than CMake package
    # configs. Resolve CURL via the module and libedit via pkg-config.
    find_package(CURL REQUIRED)
    find_package(Threads REQUIRED)
    find_package(PkgConfig REQUIRED)
    # Line editing comes from libedit (BSD-3-Clause), which implements the
    # readline API; GNU readline is GPL-3 and must not be linked. Portable builds
    # skip it (the CLI falls back to a plain line reader via the
    # FASTFLOWLM_USE_READLINE guard) so the tarball needs no libedit at runtime.
    if(NOT FLM_PORTABLE_BUILD)
        pkg_check_modules(libedit REQUIRED IMPORTED_TARGET libedit)
    endif()
endif()

# ———————————————————————————————————————————————
# Library specific settings
# ———————————————————————————————————————————————
target_include_directories(flm PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/runner
    ${CMAKE_SOURCE_DIR}/server
    ${CMAKE_SOURCE_DIR}/pull
)

# XRT backend: add the XRT headers (HRX carries its own via the hrx::hrx target).
if(NOT FLM_USE_HRX)
    if(NOT WIN32 AND XRT_FOUND)
        target_include_directories(flm PUBLIC ${XRT_INCLUDE_DIRS})
    else()
        target_include_directories(flm PUBLIC ${XRT_INCLUDE_DIR})
    endif()
endif()

if(WIN32 AND NOT VCPKG_TOOLCHAIN AND NOT FLM_WIN_DEPS_PREFIX)
    # Bare CI runner: standalone Boost + vcpkg headers by absolute path.
    target_include_directories(flm PUBLIC
        C:/dev/boost_1_88_0
        C:/dev/vcpkg/installed/x64-windows/include/
    )
elseif(FLM_WIN_DEPS_PREFIX)
    target_include_directories(flm PUBLIC "${FLM_WIN_DEPS_PREFIX}/include")
endif()

target_compile_definitions(flm PUBLIC
    DISABLE_ABI_CHECK=1
    CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
    CMAKE_XCLBIN_PREFIX="${CMAKE_XCLBIN_PREFIX}"
    __FLM_VERSION__=\"${FLM_VERSION}\"
    __NPU_VERSION__=\"${NPU_VERSION}\"
)

# Select the NPU runtime backend seen by the sources (device_runtime.hpp et al.).
if(FLM_USE_HRX)
    target_compile_definitions(flm PUBLIC FLM_USE_HRX=1)
endif()

if(WIN32)
    target_compile_definitions(flm PUBLIC
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        # Handle legacy stdio functions
        _CRT_SECURE_NO_WARNINGS
        _CRT_NONSTDC_NO_DEPRECATE
        __WINDOWS__
    )
endif()

if(WIN32 AND NOT VCPKG_TOOLCHAIN AND NOT FLM_WIN_DEPS_PREFIX)
    # Bare CI runner: statically link the standalone Boost + curl.
    target_compile_definitions(flm PUBLIC
        CURL_STATICLIB
        BOOST_ALL_NO_LIB
        BOOST_ALL_STATIC_LINK
    )
elseif(FLM_WIN_DEPS_PREFIX)
    # conda-forge Boost and curl are DLL builds. program_options exports data
    # (boost::program_options::arg, options_description::m_default_line_length)
    # that only resolves through dllimport, so BOOST_ALL_STATIC_LINK would leave
    # it unresolved at link time; CURL_STATICLIB would mis-declare curl likewise.
    target_compile_definitions(flm PUBLIC
        BOOST_ALL_NO_LIB
        BOOST_ALL_DYN_LINK
    )
endif()

target_link_directories(flm PUBLIC
    ${FLM_ENGINE_LIB_DIR}       # per-backend prebuilt engine libs (lib/xrt or lib/hrx)
    ${CMAKE_SOURCE_DIR}/lib     # shared third-party runtime libs (curl/... on Windows)
)

# XRT backend: add the XRT runtime library dir (aiebu ships in lib/<backend>).
if(NOT FLM_USE_HRX)
    if(NOT WIN32 AND XRT_FOUND)
        target_link_directories(flm PUBLIC ${XRT_LIBRARY_DIRS})
    else()
        target_link_directories(flm PUBLIC ${XRT_LIB_DIR})
    endif()
endif()

if(WIN32 AND NOT VCPKG_TOOLCHAIN AND NOT FLM_WIN_DEPS_PREFIX)
    # Bare CI runner: standalone Boost stage libs + vcpkg import libs.
    target_link_directories(flm PUBLIC
        C:/dev/boost_1_88_0/stage/lib
        C:/dev/vcpkg/installed/x64-windows/lib
    )
elseif(FLM_WIN_DEPS_PREFIX)
    target_link_directories(flm PUBLIC "${FLM_WIN_DEPS_PREFIX}/lib")
endif()

# Link static libraries first
if(MSVC)
    target_link_libraries(flm PUBLIC ${STATIC_LIBS})
endif()

# Defines the gemma4e_npu target, which the link list below then resolves to
# instead of the prebuilt of the same name.
if(FLM_BUILD_GEMMA4E)
    add_subdirectory(detail)
endif()

set(FLM_ENGINE_LINK_LIBS
    q4_npu_eXpress
    llama_npu
    qwen2_npu
    qwen2vl_npu
    qwen3_npu
    qwen3vl_npu
)

list(APPEND FLM_ENGINE_LINK_LIBS qwen3vl_flash)

list(APPEND FLM_ENGINE_LINK_LIBS
    qwen3_5vl_npu
    minicpm_v_4_7_npu
    qwen3_5_omni_npu
    qwen3_6_moe_npu
    qwen3_8mtp_npu
    gemma_npu
    gemma_text_npu
    gemma4e_npu
)

list(APPEND FLM_ENGINE_LINK_LIBS gemma4e_flash)
list(APPEND FLM_ENGINE_LINK_LIBS gemma4e_dflash)

list(APPEND FLM_ENGINE_LINK_LIBS
    gemma4_12b_npu
    gpt_oss_npu
    whisper_npu
    gemma_embedding
    lfm2_npu
    lfm3_npu
    phi4_npu
    nanbeige_npu
    hunyuan_npu
)

# The d3 decision engine (--decision d3:lite). Linked where its library is vendored; elsewhere
# (a Windows build before d3_npu.dll ships) AutoDecisionModel compiles without the engine and
# refuses to load.
if(WIN32)
    set(_d3_npu_lib "${FLM_ENGINE_LIB_DIR}/d3_npu.lib")
else()
    set(_d3_npu_lib "${FLM_ENGINE_LIB_DIR}/libd3_npu.so")
endif()
if(EXISTS "${_d3_npu_lib}")
    list(APPEND FLM_ENGINE_LINK_LIBS d3_npu)
    target_compile_definitions(flm PRIVATE FLM_HAS_D3_NPU=1)
    # Revision 2 of the engine API (d3_npu_abi_version(), d3_npu_capabilities(), d3_npu::encode_state_once: state
    # once per request, see include/models/d3/flm/aie2p/d3_npu.hpp) is used only when the vendored library exports
    # it, so the branch links against revision 1 libraries too and falls back to the per-row prefix_bidir path.
    # AUTO looks for the symbol name in the library (or import library) file; ON / OFF force it, e.g. when
    # relinking by hand against another library.
    set(FLM_D3_NPU_REV2 "AUTO" CACHE STRING "Use the d3_npu state-once API (revision 2): AUTO, ON or OFF")
    set(_d3_rev2 OFF)
    if(FLM_D3_NPU_REV2 STREQUAL "AUTO")
        file(READ "${_d3_npu_lib}" _d3_npu_hex HEX)
        string(HEX "d3_npu_abi_version" _d3_npu_sym)
        string(FIND "${_d3_npu_hex}" "${_d3_npu_sym}" _d3_npu_at)
        if(NOT _d3_npu_at EQUAL -1)
            set(_d3_rev2 ON)
        endif()
        unset(_d3_npu_hex)
    elseif(FLM_D3_NPU_REV2)
        set(_d3_rev2 ON)
    endif()
    if(_d3_rev2)
        target_compile_definitions(flm PRIVATE FLM_D3_NPU_REV2=1)
    endif()
    message(STATUS "d3_npu state-once API (revision 2): ${_d3_rev2}")
endif()

# Video input for d3:turbo (src/common/video): FFmpeg's shared libraries are loaded at run time (no build-time
# dependency); ON by default (the engine runs videos). A machine without the FFmpeg libraries keeps the
# frame-list input and answers encoded videos with a clear 422; -DFLM_VIDEO_FFMPEG=OFF leaves the decoder out.
option(FLM_VIDEO_FFMPEG "Load FFmpeg (libavformat/libavcodec/libswscale) at run time to decode video inputs" ON)
if(FLM_VIDEO_FFMPEG)
    target_link_libraries(flm PRIVATE ${CMAKE_DL_LIBS})
    # AutoDecisionModel installs the video loader (plain questions about videos in the state; D3_VIDEO=0 turns it off)
    target_compile_definitions(flm PRIVATE FLM_D3_VIDEO=1)
else()
    target_compile_definitions(flm PRIVATE FLM_NO_FFMPEG_DL=1)
endif()

# Link your custom libraries (these may still be DLLs if no static versions available)
target_link_libraries(flm PRIVATE flm_xgrammar)

target_link_libraries(flm PUBLIC
    ${FLM_ENGINE_LINK_LIBS}
    # flm's own code calls nothing in the four shared-runtime libs below -- the
    # engine .so above do -- so the linker's default --as-needed drops them from
    # DT_NEEDED entirely. The engines then have to find MHA/Dequant/... in some
    # other engine that happened to compile a copy of them, whichever one comes
    # first in DT_NEEDED. That silently ran a stale MHA for every model until the
    # copy's owner was rebuilt. Keep the real libraries in the link map.
    $<$<NOT:$<BOOL:${WIN32}>>:-Wl,--no-as-needed>
    dequant
    gemm
    lm_head
    mha
    $<$<NOT:$<BOOL:${WIN32}>>:-Wl,--as-needed>
)

# Link the selected NPU runtime after the detail model libraries above so their
# runtime references (libhrx / libxrt_coreutil) resolve.
if(FLM_USE_HRX)
    # (Windows: hrx.lib import lib for hrx.dll; Linux: libhrx.so.)
    target_link_libraries(flm PUBLIC hrx::hrx)
    if(NOT WIN32)
        set_target_properties(flm PROPERTIES
            BUILD_RPATH "$<TARGET_FILE_DIR:hrx::hrx>;${FLM_ENGINE_LIB_DIR}")
        # The HRX package ships its own libtinfo (and libedit, curl, ...) next to
        # libhrx, and the linker looks in BUILD_RPATH for the dependencies of
        # shared libraries. The system libedit linked below for non-portable
        # builds would then resolve its libtinfo to the HRX copy, which lacks the
        # NCURSES6_TINFO symbol versions, and the link fails. -rpath-link is
        # searched first and only at link time, so the system copies win and the
        # installed RPATH is unchanged.
        if(NOT FLM_PORTABLE_BUILD AND CMAKE_LIBRARY_ARCHITECTURE)
            target_link_options(flm PRIVATE
                "LINKER:-rpath-link,/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
        endif()
    endif()
else()
    # XRT: link xrt_coreutil (+ aiebu for the ELF assembler used by npu_utils_xrt).
    if(XRT_BUILT_FROM_SOURCE AND TARGET xrt_coreutil)
        target_link_libraries(flm PUBLIC xrt_coreutil)
    elseif(NOT WIN32 AND XRT_FOUND)
        target_link_libraries(flm PUBLIC ${XRT_LIBRARIES})
    else()
        target_link_libraries(flm PUBLIC xrt_coreutil)
    endif()
    if(WIN32)
        target_link_libraries(flm PUBLIC aiebu_static)
    else()
        target_link_libraries(flm PUBLIC aiebu)
    endif()
    if(NOT WIN32)
        set_target_properties(flm PROPERTIES
            BUILD_RPATH "${FLM_ENGINE_LIB_DIR}")
    endif()
endif()

# The source-built engine sits in the build tree rather than lib/<backend>, and
# has to precede it: a prebuilt of the same name is still sitting there.
if(FLM_BUILD_GEMMA4E AND NOT WIN32)
    get_target_property(_flm_build_rpath flm BUILD_RPATH)
    if(NOT _flm_build_rpath)
        set(_flm_build_rpath "")
    endif()
    set_target_properties(flm PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}/engines;${_flm_build_rpath}")
endif()

if(WIN32 AND VCPKG_TOOLCHAIN)
    # Local/managed vcpkg: link the imported targets from the CONFIG packages
    # found above (versioned import-lib names resolved automatically).
    target_link_libraries(flm PUBLIC
        Boost::program_options
        CURL::libcurl
    )
elseif(WIN32 AND FLM_WIN_DEPS_PREFIX)
    # Local prefix: the names it ships (checked at configure time above).
    target_link_libraries(flm PUBLIC
        ${FLM_WIN_DEPS_PREFIX_LIBS}
    )
elseif(WIN32)
    # Bare CI runner: link the native deps by raw name from the standalone Boost
    # + vcpkg lib dirs wired above (no CMake package configs there).
    target_link_libraries(flm PUBLIC
        libcurl
        libboost_program_options-vc143-mt-x64-1_88
    )
else()
    target_link_libraries(flm PUBLIC
        Boost::program_options
        CURL::libcurl
    )
endif()

if(NOT WIN32)
    target_compile_options(flm PUBLIC -mavx -mavx2)

    # libedit (readline API) is only linked for non-portable builds (see the
    # discovery block); the CLI compiles a plain line-reader fallback when the
    # macro is undefined.
    if(NOT FLM_PORTABLE_BUILD)
        target_compile_definitions(flm PUBLIC FASTFLOWLM_USE_READLINE=1)
        target_link_libraries(flm PUBLIC PkgConfig::libedit)
    endif()

    # Link with dl library for dlopen() support (used for preloading bundled libraries)
    target_link_libraries(flm PUBLIC dl)
else()
    # cl and clang-cl ignore -mavx -mavx2. /arch:AVX2 is the matching flag for
    # the rest of flm.
    target_compile_options(flm PUBLIC /arch:AVX2)
endif()

# ———————————————————————————————————————————————
# Link tokenizers-cpp libraries
# ———————————————————————————————————————————————
# The tokenizers-cpp subproject provides these targets:
# - tokenizers_cpp: main tokenizers C++ library
# - tokenizers_c: tokenizers C FFI bindings  
# - sentencepiece: sentencepiece tokenizer library
target_link_libraries(flm PRIVATE
    tokenizers_cpp
)
target_link_libraries(flm PRIVATE flm_image_codecs flm_audio_codecs)

if(WIN32)
    target_link_libraries(flm PRIVATE
        ntdll wsock32 ws2_32 Bcrypt
        iphlpapi userenv psapi
        crypt32 secur32 advapi32 normaliz wldap32
    )
else()
    target_link_libraries(flm PRIVATE Threads::Threads)
endif()


# ———————————————————————————————————————————————
# Copy the build flm.exe into the out directory (for local dev)
# ———————————————————————————————————————————————
if(WIN32)
    add_custom_command(TARGET flm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:flm>
            ${CMAKE_SOURCE_DIR}/out/flm.exe
    )
endif()

# Default install location for model_list.json / xclbins (matches the app's
# relocatable "<exe_dir>/../share/flm" lookup). Overridden to the prefix root for
# the flat portable layout below.
set(FLM_SHARE_DESTINATION "share/flm")

# Add a custom target to check for remaining DLL dependencies
if(WIN32)
    add_custom_target(check_dependencies ALL
        COMMAND ${CMAKE_COMMAND} -E echo "Checking for DLL dependencies..."
        COMMAND dumpbin /dependents $<TARGET_FILE:flm> | findstr /i ".dll"
        COMMENT "Checking for remaining DLL dependencies"
        DEPENDS flm
    )
else()
    if(NOT CMAKE_INSTALL_LIBDIR)
        set(CMAKE_INSTALL_LIBDIR "lib")
    endif()

    # ————————————————————————————————————————————————————————————
    # Install layout. Portable = flat, self-contained tree rooted at the install
    # prefix (consumed by the portable tarball workflow); otherwise an FHS layout
    # (bin/, lib/, share/) consumed by the .deb packaging.
    # ————————————————————————————————————————————————————————————
    if(FLM_PORTABLE_BUILD)
        set(FLM_BIN_DESTINATION ".")
        set(FLM_ENGINE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
        set(FLM_SHARE_DESTINATION ".")
        set(FLM_FLM_INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
        # Engine libs sit next to libhrx in <root>/lib.
        set(FLM_ENGINE_INSTALL_RPATH "$ORIGIN")
    elseif(FLM_USE_HRX)
        # HRX FHS layout: engine libs sit in <prefix>/lib/flm and the bundled NPU
        # runtime (libhrx) one level up in <prefix>/lib, so engines resolve it via
        # $ORIGIN/.. — this separation is what lets the HRX .deb ship libhrx.
        set(FLM_BIN_DESTINATION "bin")
        set(FLM_ENGINE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}/flm")
        set(FLM_SHARE_DESTINATION "share/flm")
        set(FLM_FLM_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/flm;$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
        set(FLM_ENGINE_INSTALL_RPATH "$ORIGIN:$ORIGIN/..")
    else()
        # XRT FHS layout: identical to upstream main — engine libs install directly
        # into <prefix>/lib and flm resolves them via $ORIGIN/../lib. XRT itself is
        # an external dependency (libxrt-npu2) resolved from the system.
        set(FLM_BIN_DESTINATION "bin")
        set(FLM_ENGINE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
        set(FLM_SHARE_DESTINATION "share/flm")
        set(FLM_FLM_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
        set(FLM_ENGINE_INSTALL_RPATH "$ORIGIN")
    endif()

    file(GLOB so_libs "${FLM_ENGINE_LIB_DIR}/*.so*")
    # Installed from the glob unless a source build supersedes one of them. The
    # name still has to reach _flm_engine_names below, so filter a copy.
    set(_flm_prebuilt_libs ${so_libs})
    if(FLM_BUILD_GEMMA4E)
        list(FILTER _flm_prebuilt_libs EXCLUDE REGEX "/libgemma4e_npu\\.so[^/]*$")
        set_target_properties(gemma4e_npu PROPERTIES INSTALL_RPATH "${FLM_ENGINE_INSTALL_RPATH}")
        install(TARGETS gemma4e_npu LIBRARY DESTINATION "${FLM_ENGINE_LIB_DESTINATION}")
    endif()
    install(FILES ${_flm_prebuilt_libs} DESTINATION "${FLM_ENGINE_LIB_DESTINATION}")
    set_target_properties(flm PROPERTIES INSTALL_RPATH "${FLM_FLM_INSTALL_RPATH}")

    # Engine .so file names, used below to keep them out of the flm dependency
    # closure (they are installed explicitly, above).
    set(_flm_engine_names "")
    set(_flm_engine_post_exclude "")
    foreach(_flm_src ${so_libs})
        get_filename_component(_flm_name "${_flm_src}" NAME)
        list(APPEND _flm_engine_names "${_flm_name}")
        string(REPLACE "." "[.]" _flm_name_re "${_flm_name}")
        list(APPEND _flm_engine_post_exclude ".*/${_flm_name_re}$")
    endforeach()

    # HRX only: the prebuilt HRX engine .so ship with an absolute build-machine
    # RUNPATH and a libhrx.so.0 NEEDED entry. Rewrite the RUNPATH to a
    # relocatable $ORIGIN-based one so libhrx resolves at runtime and, crucially,
    # so dpkg-shlibdeps can locate it during .deb packaging. The XRT engine .so
    # are shipped unmodified: the portable wrapper makes ./lib discoverable via
    # LD_LIBRARY_PATH, and for the .deb XRT is an external dependency resolved
    # from the system.
    if(FLM_USE_HRX)
        # Not REQUIRED at configure time: patchelf is only needed to install an HRX
        # build, so demanding it up front would block a plain `cmake --build`.
        find_program(PATCHELF_EXECUTABLE patchelf)
        if(PATCHELF_EXECUTABLE)
            install(CODE "
                set(_flm_engine_names \"${_flm_engine_names}\")
                foreach(_flm_name \${_flm_engine_names})
                    set(_flm_lib \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${FLM_ENGINE_LIB_DESTINATION}/\${_flm_name}\")
                    if(EXISTS \"\${_flm_lib}\" AND NOT IS_SYMLINK \"\${_flm_lib}\")
                        execute_process(
                            COMMAND \"${PATCHELF_EXECUTABLE}\" --set-rpath \"${FLM_ENGINE_INSTALL_RPATH}\" \"\${_flm_lib}\"
                            RESULT_VARIABLE _flm_patchelf_rc)
                        if(NOT _flm_patchelf_rc EQUAL 0)
                            message(FATAL_ERROR \"patchelf --set-rpath failed for \${_flm_lib}\")
                        endif()
                    endif()
                endforeach()
            ")
        else()
            message(WARNING "patchelf not found: this HRX build can be compiled but not installed. Install patchelf and re-run CMake configure before `cmake --install`.")
            install(CODE "message(FATAL_ERROR \"patchelf not found at configure time; it is required to install an HRX build. Install patchelf and re-run CMake configure.\")")
        endif()
    endif()

    # Portable build ships a small wrapper as `flm`; the packaging step renames the
    # real binary to `flm-real` and this wrapper to `flm`.
    if(FLM_PORTABLE_BUILD)
        configure_file(
            "${CMAKE_SOURCE_DIR}/flm-wrapper.sh.in"
            "${CMAKE_BINARY_DIR}/flm-wrapper.sh"
            @ONLY)
        install(PROGRAMS "${CMAKE_BINARY_DIR}/flm-wrapper.sh" DESTINATION "${FLM_SHARE_DESTINATION}")
    endif()
endif()

# Directory holding the selected NPU runtime shared library, used by the Windows
# dependency set, the portable XRT bundling, and the HRX dependency closure. HRX
# ships libhrx via its imported target; XRT is a system/library-dir dependency.
if(FLM_USE_HRX)
    set(_flm_rt_dep_dir "$<TARGET_FILE_DIR:hrx::hrx>")
    # Sentinel that never matches a real library name (an empty PRE_EXCLUDE regex
    # would match — and thus exclude — everything).
    set(_flm_rt_pre_exclude "__flm_no_such_lib__")
elseif(NOT WIN32 AND XRT_FOUND)
    set(_flm_rt_dep_dir "${XRT_LIBRARY_DIRS}")
else()
    set(_flm_rt_dep_dir "${XRT_LIB_DIR}")
endif()

if(WIN32)
    install(TARGETS flm
        RUNTIME_DEPENDENCY_SET fastflowlm_runtime_dependencies
        RUNTIME DESTINATION bin)
    install(RUNTIME_DEPENDENCY_SET fastflowlm_runtime_dependencies
        DESTINATION bin
        DIRECTORIES
            "${FLM_ENGINE_LIB_DIR}"
            "${CMAKE_SOURCE_DIR}/lib"
            "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
            "${_flm_rt_dep_dir}"
        PRE_EXCLUDE_REGEXES "api-ms-.*" "ext-ms-.*"
            # Optional Windows feature-on-demand / attestation / lockdown DLLs
            # that are not present as files on all hosts, so the resolver would
            # report them unresolved. They are OS components, never bundled.
            # Patterns are case-insensitive because PE import names are lowercase.
            "[Aa][Zz][Uu][Rr][Ee][Aa][Tt][Tt][Ee][Ss][Tt].*"
            "[Hh][Vv][Ss][Ii][Ff][Ii][Ll][Ee][Tt][Rr][Uu][Ss][Tt].*"
            "[Pp][Dd][Mm][Uu][Tt][Ii][Ll][Ii][Tt][Ii][Ee][Ss].*"
            "[Ww][Pp][Aa][Xx][Hh][Oo][Ll][Dd][Ee][Rr].*"
        POST_EXCLUDE_REGEXES ".*[Ww]indows[/\\\\][Ss]ystem32[/\\\\].*")
elseif(NOT FLM_USE_HRX AND FLM_PORTABLE_BUILD)
    # Portable XRT: bundle the XRT runtime explicitly. XRT is deliberately
    # handled here instead of via the dependency closure below: flm only lists
    # libxrt_coreutil as a link-time NEEDED entry, while XRT dlopens libxrt_core,
    # the libxrt_driver_xdna plugin and others at run time via a path it builds
    # as $XILINX_XRT/lib/x86_64-linux-gnu/<lib>. Those are invisible to
    # file(GET_RUNTIME_DEPENDENCIES), so copy the whole libxrt*.so* set (plus
    # boost_program_options), re-root each real lib at $ORIGIN, and mirror them
    # under lib/x86_64-linux-gnu for XRT's internal lookup. Other shared deps
    # (libcurl, libgomp, ...) resolve from the host.
    install(TARGETS flm RUNTIME DESTINATION "${FLM_BIN_DESTINATION}")
    find_program(PATCHELF_EXECUTABLE patchelf REQUIRED)
    install(CODE "
        # Locate a directory that actually holds the XRT runtime libraries.
        set(_flm_xrt_src \"\")
        foreach(_cand \"${_flm_rt_dep_dir}\" \"/usr/lib/x86_64-linux-gnu\" \"/usr/lib\")
            if(_cand AND EXISTS \"\${_cand}/libxrt_coreutil.so.2\")
                set(_flm_xrt_src \"\${_cand}\")
                break()
            endif()
        endforeach()
        if(NOT _flm_xrt_src)
            message(FATAL_ERROR
                \"Portable XRT build: could not locate libxrt_coreutil.so.2 to bundle\")
        endif()
        message(STATUS \"Bundling XRT runtime libraries from \${_flm_xrt_src}\")

        set(_flm_libdir \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
        file(MAKE_DIRECTORY \"\${_flm_libdir}\")

        # Copy every libxrt*.so* (from the XRT dir) plus libboost_program_options.so*
        # (searched across the standard system dirs, since Boost may live apart
        # from XRT — e.g. XRT under /opt/xilinx but Boost in the multiarch dir),
        # preserving the symlink chain.
        file(GLOB _flm_xrt_files \"\${_flm_xrt_src}/libxrt*.so*\")
        foreach(_bdir \"\${_flm_xrt_src}\" \"/usr/lib/x86_64-linux-gnu\" \"/usr/lib\")
            file(GLOB _flm_boost_files \"\${_bdir}/libboost_program_options.so*\")
            if(_flm_boost_files)
                list(APPEND _flm_xrt_files \${_flm_boost_files})
                break()
            endif()
        endforeach()
        foreach(_f \${_flm_xrt_files})
            get_filename_component(_n \"\${_f}\" NAME)
            if(IS_SYMLINK \"\${_f}\")
                file(READ_SYMLINK \"\${_f}\" _t)
                execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink
                    \"\${_t}\" \"\${_flm_libdir}/\${_n}\")
            else()
                file(COPY \"\${_f}\" DESTINATION \"\${_flm_libdir}\"
                    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                     GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
            endif()
        endforeach()

        # Re-root the real XRT libs at \$ORIGIN so they find each other in lib/.
        file(GLOB _flm_xrt_real \"\${_flm_libdir}/libxrt*.so.*\")
        foreach(_l \${_flm_xrt_real})
            if(NOT IS_SYMLINK \"\${_l}\")
                execute_process(
                    COMMAND \"${PATCHELF_EXECUTABLE}\" --set-rpath \"\$ORIGIN\" \"\${_l}\"
                    RESULT_VARIABLE _rc)
                if(NOT _rc EQUAL 0)
                    message(FATAL_ERROR \"patchelf --set-rpath failed for \${_l}\")
                endif()
            endif()
        endforeach()

        # Mirror the bundled XRT libs under lib/x86_64-linux-gnu for XRT's lookup.
        set(_flm_multiarch \"\${_flm_libdir}/x86_64-linux-gnu\")
        file(MAKE_DIRECTORY \"\${_flm_multiarch}\")
        file(GLOB _flm_xrt_all \"\${_flm_libdir}/libxrt*.so*\")
        foreach(_l \${_flm_xrt_all})
            get_filename_component(_n \"\${_l}\" NAME)
            if(IS_SYMLINK \"\${_l}\")
                file(READ_SYMLINK \"\${_l}\" _t)
                execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink
                    \"../\${_t}\" \"\${_flm_multiarch}/\${_n}\")
            else()
                execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink
                    \"../\${_n}\" \"\${_flm_multiarch}/\${_n}\")
            endif()
        endforeach()
    ")
elseif(NOT FLM_USE_HRX)
    # System XRT (.deb): XRT is an external dependency (the libxrt-npu2 package)
    # and flm's other shared libraries are resolved from the system via
    # ${shlibs:Depends}. Nothing is bundled here.
    install(TARGETS flm RUNTIME DESTINATION "${FLM_BIN_DESTINATION}")
else()
    # HRX: bundle the NPU runtime (libhrx) and Boost.Program_options into
    # ${CMAKE_INSTALL_LIBDIR} -- the same two third-party libraries the portable
    # XRT build bundles. Everything else flm and the engine .so link (libcurl and
    # its TLS/IDN/Kerberos stack, the GCC runtime, glibc's libmvec/libresolv)
    # resolves from the host, as it does for XRT. Copying the whole closure
    # shipped LGPL libraries (GnuTLS, nettle, GMP, libidn2, librtmp, ...) that
    # THIRD_PARTY_NOTICES.txt does not cover, plus glibc pieces that have to
    # match the host's libc anyway.
    #
    # The closure is still resolved, rooted at flm together with the engine .so
    # (roots are traversed but never emitted, so the engine .so stay solely under
    # the RUNPATH-patched install above), so a dependency the build host cannot
    # resolve still fails the install. The engine names are still POST_EXCLUDEd
    # as a guard for the case of one engine lib depending on another.
    install(TARGETS flm RUNTIME DESTINATION "${FLM_BIN_DESTINATION}")
    install(CODE "
        set(_flm_engine_libs \"${so_libs}\")
        set(_flm_engine_post_exclude \"${_flm_engine_post_exclude}\")
        file(GET_RUNTIME_DEPENDENCIES
            EXECUTABLES \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${FLM_BIN_DESTINATION}/flm\"
            LIBRARIES \${_flm_engine_libs}
            RESOLVED_DEPENDENCIES_VAR _flm_resolved
            UNRESOLVED_DEPENDENCIES_VAR _flm_unresolved
            DIRECTORIES
                \"${FLM_ENGINE_LIB_DIR}\"
                \"${CMAKE_SOURCE_DIR}/lib\"
                \"${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib\"
                \"${_flm_rt_dep_dir}\"
            PRE_EXCLUDE_REGEXES
                \"linux-vdso\\\\.so.*\"
                \"ld-linux.*\"
                \"lib(c|dl|m|pthread|rt)\\\\.so.*\"
                \"${_flm_rt_pre_exclude}\"
            POST_EXCLUDE_REGEXES \${_flm_engine_post_exclude})
        if(_flm_unresolved)
            message(FATAL_ERROR
                \"FastFlowLM runtime dependency closure is incomplete; unresolved: \${_flm_unresolved}\")
        endif()
        set(_flm_bundled \"\")
        foreach(_flm_dep \${_flm_resolved})
            get_filename_component(_flm_dep_name \"\${_flm_dep}\" NAME)
            if(_flm_dep_name MATCHES \"^lib(hrx|boost_program_options)[.]so\")
                list(APPEND _flm_bundled \"\${_flm_dep}\")
            endif()
        endforeach()
        if(NOT _flm_bundled MATCHES \"libhrx[.]so\")
            message(FATAL_ERROR \"libhrx is not in the flm dependency closure: \${_flm_resolved}\")
        endif()
        # file(INSTALL) prepends \$ENV{DESTDIR} itself, so DESTINATION must not.
        file(INSTALL
            DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\"
            TYPE SHARED_LIBRARY
            FOLLOW_SYMLINK_CHAIN
            FILES \${_flm_bundled})
    ")
endif()

install(FILES model_list.json DESTINATION "${FLM_SHARE_DESTINATION}")
install(FILES model_info.json DESTINATION "${FLM_SHARE_DESTINATION}")

# Ship both licenses with the binaries; LICENSE_BINARY.txt must accompany any
# redistribution of the NPU kernels and engine libraries. THIRD_PARTY_NOTICES.txt
# carries the notices for bundled and statically linked third-party code.
install(FILES
    ${CMAKE_SOURCE_DIR}/../LICENSE_RUNTIME.txt
    ${CMAKE_SOURCE_DIR}/../LICENSE_BINARY.txt
    ${CMAKE_SOURCE_DIR}/../THIRD_PARTY_NOTICES.txt
    DESTINATION "${FLM_SHARE_DESTINATION}")

# xclbins, which are loaded by shared libraries need to be in location
# relative to the executable, so we install them relative to the binary.
install(DIRECTORY xclbins DESTINATION "${FLM_SHARE_DESTINATION}")
