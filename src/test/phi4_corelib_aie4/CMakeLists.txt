cmake_minimum_required(VERSION 3.22)
project(phi4_corelib_aie4_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT WIN32)
  message(FATAL_ERROR "Phi-4 corelib AIE4 tests currently require Windows")
endif()

find_path(RYZENAI_CORELIB_INCLUDE_DIR NAMES ryzenai/corelib.h REQUIRED)
find_path(BOOST_INCLUDE_DIR NAMES boost/program_options.hpp
    HINTS "$ENV{CONDA_PREFIX}/Library/include"
          "$ENV{USERPROFILE}/anaconda3/Library/include" REQUIRED)
find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
    HINTS "$ENV{XRT_INCLUDE_DIR}"
          "${CMAKE_CURRENT_LIST_DIR}/../../../../xrt_package/xrt/include"
          "C:/dev/XRT/src/runtime_src/core/include" REQUIRED)
set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
set(CORELIB_SOURCES
    "${FLM_SOURCE_DIR}/common/corelib/corelib_api.cpp"
    "${FLM_SOURCE_DIR}/common/corelib/corelib_runtime.cpp")

# A synthetic mismatched copy proves that the adapter rejects header drift at
# compile time. The caller-provided header remains untouched.
file(READ "${RYZENAI_CORELIB_INCLUDE_DIR}/ryzenai/corelib.h" CORELIB_HEADER_TEXT)
string(REGEX REPLACE
    "#define RYZENAI_CORELIB_VERSION_PATCH[ \t]+0"
    "#define RYZENAI_CORELIB_VERSION_PATCH 1"
    WRONG_CORELIB_HEADER_TEXT "${CORELIB_HEADER_TEXT}")
set(WRONG_CORELIB_INCLUDE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/wrong-corelib-version/include")
file(MAKE_DIRECTORY "${WRONG_CORELIB_INCLUDE_DIR}/ryzenai")
file(WRITE "${WRONG_CORELIB_INCLUDE_DIR}/ryzenai/corelib.h"
    "${WRONG_CORELIB_HEADER_TEXT}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/wrong-corelib-version/probe.cpp"
    "#define RYZENAI_CORELIB_STATIC 1\n#include \"corelib/corelib_api.hpp\"\n")
try_compile(CORELIB_WRONG_VERSION_ACCEPTED
    SOURCES "${CMAKE_CURRENT_BINARY_DIR}/wrong-corelib-version/probe.cpp"
    CMAKE_FLAGS
        "-DCMAKE_CXX_STANDARD=20"
        "-DCMAKE_CXX_FLAGS=/I${FLM_SOURCE_DIR}/include /I${WRONG_CORELIB_INCLUDE_DIR}"
    OUTPUT_VARIABLE WRONG_CORELIB_COMPILE_OUTPUT)
if(CORELIB_WRONG_VERSION_ACCEPTED)
  message(FATAL_ERROR "Corelib adapter accepted a non-0.3.0 header")
endif()

add_executable(test_corelib_api
    test_corelib_api.cpp fake_corelib.cpp ${CORELIB_SOURCES})
target_include_directories(test_corelib_api PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${FLM_SOURCE_DIR}/include"
    "${RYZENAI_CORELIB_INCLUDE_DIR}")
target_compile_definitions(test_corelib_api PRIVATE RYZENAI_CORELIB_STATIC=1)

add_executable(test_real_corelib test_real_corelib.cpp ${CORELIB_SOURCES})
target_include_directories(test_real_corelib PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${FLM_SOURCE_DIR}/include"
    "${RYZENAI_CORELIB_INCLUDE_DIR}")
target_compile_definitions(test_real_corelib PRIVATE RYZENAI_CORELIB_STATIC=1)

add_executable(test_phi4_gguf
    test_phi4_gguf.cpp fake_corelib.cpp
    "${FLM_SOURCE_DIR}/common/corelib/corelib_api.cpp"
    "${FLM_SOURCE_DIR}/common/corelib/phi4_corelib_gguf.cpp")
target_include_directories(test_phi4_gguf PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${FLM_SOURCE_DIR}/include"
    "${RYZENAI_CORELIB_INCLUDE_DIR}")
target_compile_definitions(test_phi4_gguf PRIVATE RYZENAI_CORELIB_STATIC=1)

add_executable(test_phi4_host
    test_phi4_host.cpp
    "${FLM_SOURCE_DIR}/common/corelib/phi4_corelib_host.cpp")
target_include_directories(test_phi4_host PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${FLM_SOURCE_DIR}/include")

add_executable(test_phi4_shape_plan
    test_phi4_shape_plan.cpp fake_corelib.cpp
    "${FLM_SOURCE_DIR}/common/corelib/corelib_api.cpp"
    "${FLM_SOURCE_DIR}/common/corelib/phi4_corelib_shape_plan.cpp")
target_include_directories(test_phi4_shape_plan PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${FLM_SOURCE_DIR}/include"
    "${RYZENAI_CORELIB_INCLUDE_DIR}")
target_compile_definitions(test_phi4_shape_plan PRIVATE RYZENAI_CORELIB_STATIC=1)

add_executable(test_phi4_engine
    test_phi4_engine.cpp fake_corelib.cpp
    "${FLM_SOURCE_DIR}/common/corelib/corelib_api.cpp"
    "${FLM_SOURCE_DIR}/common/corelib/corelib_runtime.cpp"
    "${FLM_SOURCE_DIR}/common/corelib/phi4_corelib_gguf.cpp"
    "${FLM_SOURCE_DIR}/common/corelib/phi4_corelib_host.cpp"
    "${FLM_SOURCE_DIR}/common/corelib/phi4_corelib_shape_plan.cpp"
    "${FLM_SOURCE_DIR}/common/corelib/phi4_corelib_aie4.cpp")
target_include_directories(test_phi4_engine PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${FLM_SOURCE_DIR}/include"
    "${RYZENAI_CORELIB_INCLUDE_DIR}"
    "${BOOST_INCLUDE_DIR}"
    "${XRT_INCLUDE_DIR}")
target_compile_definitions(test_phi4_engine PRIVATE
    RYZENAI_CORELIB_STATIC=1 USEAVX2=1 DISABLE_ABI_CHECK=1
    _ENABLE_EXTENDED_ALIGNED_STORAGE WIN32_LEAN_AND_MEAN NOMINMAX)
target_compile_options(test_phi4_engine PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
target_link_directories(test_phi4_engine PRIVATE "${XRT_INCLUDE_DIR}/../lib")
target_link_libraries(test_phi4_engine PRIVATE xrt_coreutil)

set(PHI4_FRONTEND_SOURCES
    "${FLM_SOURCE_DIR}/common/AutoModel/automodel.cpp"
    "${FLM_SOURCE_DIR}/common/AutoModel/modeling_phi4.cpp")

add_executable(test_phi4_frontend
    test_phi4_frontend.cpp
    ${PHI4_FRONTEND_SOURCES}
    ${CORELIB_SOURCES}
    "${FLM_SOURCE_DIR}/common/corelib/phi4_corelib_gguf.cpp")
target_include_directories(test_phi4_frontend PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${FLM_SOURCE_DIR}/include"
    "${FLM_SOURCE_DIR}/server"
    "${FLM_SOURCE_DIR}/pull"
    "${FLM_SOURCE_DIR}/runner"
    "${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp/include"
    "${RYZENAI_CORELIB_INCLUDE_DIR}"
    "${BOOST_INCLUDE_DIR}"
    "${XRT_INCLUDE_DIR}")
target_compile_definitions(test_phi4_frontend PRIVATE
    FLM_ENABLE_CORELIB_AIE4=1 FLM_CORELIB_TESTING=1
    RYZENAI_CORELIB_STATIC=1 DEV_BUILD=1 __WINDOWS__ USEAVX2=1
    DISABLE_ABI_CHECK=1 _ENABLE_EXTENDED_ALIGNED_STORAGE
    WIN32_LEAN_AND_MEAN NOMINMAX)
target_compile_options(test_phi4_frontend PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
target_link_directories(test_phi4_frontend PRIVATE "${XRT_INCLUDE_DIR}/../lib")
target_link_libraries(test_phi4_frontend PRIVATE xrt_coreutil)

add_executable(test_phi4_frontend_off
    test_phi4_frontend.cpp
    ${PHI4_FRONTEND_SOURCES})
target_include_directories(test_phi4_frontend_off PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${FLM_SOURCE_DIR}/include"
    "${FLM_SOURCE_DIR}/server"
    "${FLM_SOURCE_DIR}/pull"
    "${FLM_SOURCE_DIR}/runner"
    "${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp/include"
    "${BOOST_INCLUDE_DIR}"
    "${XRT_INCLUDE_DIR}")
target_compile_definitions(test_phi4_frontend_off PRIVATE
    FLM_CORELIB_TESTING=1 DEV_BUILD=1 __WINDOWS__ USEAVX2=1
    DISABLE_ABI_CHECK=1 _ENABLE_EXTENDED_ALIGNED_STORAGE
    WIN32_LEAN_AND_MEAN NOMINMAX)
target_compile_options(test_phi4_frontend_off PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
target_link_directories(test_phi4_frontend_off PRIVATE "${XRT_INCLUDE_DIR}/../lib")
target_link_libraries(test_phi4_frontend_off PRIVATE xrt_coreutil)

# Compile the actual production frontend translation units in both feature modes.
# Empty declaration-only FFmpeg headers isolate this compile check from an
# unrelated optional SDK that is absent on the standalone test host.
set(FRONTEND_STUB_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/frontend-compile-stubs")
foreach(STUB_HEADER IN ITEMS
        libavcodec/avcodec.h
        libavformat/avformat.h
        libswscale/swscale.h
        libavutil/imgutils.h
        libavutil/frame.h
        libavutil/pixfmt.h)
  get_filename_component(STUB_PARENT
      "${FRONTEND_STUB_INCLUDE_DIR}/${STUB_HEADER}" DIRECTORY)
  file(MAKE_DIRECTORY "${STUB_PARENT}")
  file(WRITE "${FRONTEND_STUB_INCLUDE_DIR}/${STUB_HEADER}" "#pragma once\n")
endforeach()

set(FLM_PRODUCTION_FRONTEND_SOURCES
    ${PHI4_FRONTEND_SOURCES}
    "${FLM_SOURCE_DIR}/runner/runner.cpp"
    "${FLM_SOURCE_DIR}/server/rest_handler.cpp"
    "${FLM_SOURCE_DIR}/server/server.cpp")
function(add_frontend_compile_guard TARGET_NAME ENABLE_CORELIB)
  add_library(${TARGET_NAME} OBJECT ${FLM_PRODUCTION_FRONTEND_SOURCES})
  target_include_directories(${TARGET_NAME} PRIVATE
      "${FLM_SOURCE_DIR}/include"
      "${FLM_SOURCE_DIR}/runner"
      "${FLM_SOURCE_DIR}/server"
      "${FLM_SOURCE_DIR}/pull"
      "${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp/include"
      "${FRONTEND_STUB_INCLUDE_DIR}"
      "${BOOST_INCLUDE_DIR}"
      "${XRT_INCLUDE_DIR}")
  target_compile_definitions(${TARGET_NAME} PRIVATE
      DEV_BUILD=1 __WINDOWS__ USEAVX2=1 DISABLE_ABI_CHECK=1
      _ENABLE_EXTENDED_ALIGNED_STORAGE WIN32_LEAN_AND_MEAN NOMINMAX
      __FLM_VERSION__="test" __NPU_VERSION__="0.0.0.0"
      CMAKE_INSTALL_PREFIX="${FLM_SOURCE_DIR}/build"
      CMAKE_XCLBIN_PREFIX="${FLM_SOURCE_DIR}/xclbins")
  target_compile_options(${TARGET_NAME} PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
  if(ENABLE_CORELIB)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        FLM_ENABLE_CORELIB_AIE4=1)
    target_include_directories(${TARGET_NAME} PRIVATE
        "${RYZENAI_CORELIB_INCLUDE_DIR}")
  endif()
endfunction()

# The OFF target deliberately has no RYZENAI_CORELIB_INCLUDE_DIR.
add_frontend_compile_guard(phi4_frontend_compile_off FALSE)
add_frontend_compile_guard(phi4_frontend_compile_on TRUE)

include(CTest)
add_test(NAME test_corelib_api COMMAND test_corelib_api)
add_test(NAME test_real_corelib COMMAND test_real_corelib)
add_test(NAME test_phi4_gguf COMMAND test_phi4_gguf)
add_test(NAME test_phi4_host COMMAND test_phi4_host)
add_test(NAME test_phi4_shape_plan COMMAND test_phi4_shape_plan)
add_test(NAME test_phi4_engine COMMAND test_phi4_engine)
add_test(NAME test_phi4_frontend COMMAND test_phi4_frontend)
add_test(NAME test_phi4_frontend_off COMMAND test_phi4_frontend_off)
set_tests_properties(test_real_corelib PROPERTIES SKIP_RETURN_CODE 77)
