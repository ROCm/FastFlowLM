cmake_minimum_required(VERSION 3.22)
project(test_qwen_frontend LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "oga-hipep frontend tests require Windows")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(_default_flm_source_dir "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(FLM_SOURCE_DIR "${_default_flm_source_dir}" CACHE PATH "FastFlowLM src directory")

find_path(BOOST_INCLUDE_DIR NAMES boost/any.hpp
    HINTS "$ENV{CONDA_PREFIX}/Library/include"
          "$ENV{USERPROFILE}/anaconda3/Library/include" REQUIRED)
find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
    HINTS "$ENV{XRT_INCLUDE_DIR}"
          "C:/dev/XRT/src/runtime_src/core/include" REQUIRED)
set(XRT_LIB_DIR "${XRT_INCLUDE_DIR}/../lib" CACHE PATH "Directory containing xrt_coreutil")

include("${FLM_SOURCE_DIR}/common/chat/chat_sources.cmake")

add_executable(test_qwen_frontend
    "${FLM_SOURCE_DIR}/test/oga_hipep/test_qwen_frontend.cpp"
    "${FLM_SOURCE_DIR}/common/prefix_slot_cache.cpp"
    "${FLM_SOURCE_DIR}/common/AutoModel/automodel.cpp"
    "${FLM_SOURCE_DIR}/common/AutoModel/model_backend.cpp"
    "${FLM_SOURCE_DIR}/common/AutoModel/modeling_qwen3_5vl.cpp"
    ${FLM_CHAT_SOURCES})

target_include_directories(test_qwen_frontend PRIVATE
    "${FLM_SOURCE_DIR}/include"
    "${FLM_SOURCE_DIR}/pull"
    "${FLM_SOURCE_DIR}/runner"
    "${FLM_SOURCE_DIR}/server"
    "${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp/include"
    "${BOOST_INCLUDE_DIR}"
    "${XRT_INCLUDE_DIR}")
target_compile_definitions(test_qwen_frontend PRIVATE
    DEV_BUILD=1 __WINDOWS__ USEAVX2=1 DISABLE_ABI_CHECK=1 _ENABLE_EXTENDED_ALIGNED_STORAGE
    WIN32_LEAN_AND_MEAN NOMINMAX)
target_compile_options(test_qwen_frontend PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
target_link_directories(test_qwen_frontend PRIVATE "${XRT_LIB_DIR}")
target_link_libraries(test_qwen_frontend PRIVATE xrt_coreutil flm_xgrammar)

include(CTest)
add_test(NAME test_qwen_frontend COMMAND test_qwen_frontend)
