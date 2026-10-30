cmake_minimum_required(VERSION 3.22)
project(llama_core_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT WIN32)
  message(FATAL_ERROR "Llama core tests currently require Windows")
endif()

# The frontend never includes the corelib header, so only XRT and Boost are
# needed: xrt/xrt_bo.h comes in through buffer.hpp, and boost/any.hpp through it.
find_path(BOOST_INCLUDE_DIR NAMES boost/any.hpp
    HINTS "$ENV{CONDA_PREFIX}/Library/include"
          "$ENV{USERPROFILE}/anaconda3/Library/include" REQUIRED)
find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
    HINTS "$ENV{XRT_INCLUDE_DIR}"
          "${CMAKE_CURRENT_LIST_DIR}/../../../../xrt_package/xrt/include"
          "C:/dev/XRT/src/runtime_src/core/include" REQUIRED)
set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

# builtin_backends.cpp is left out because it names every prebuilt engine
# library; test_llama_frontend.cpp supplies its own empty
# register_builtin_backends and registers a stub instead.
# automodel.cpp parses output and constrains decoding with the chat library,
# so the target also links flm_xgrammar.
include("${FLM_SOURCE_DIR}/common/chat/chat_sources.cmake")
add_executable(test_llama_frontend
    test_llama_frontend.cpp
    "${FLM_SOURCE_DIR}/common/AutoModel/automodel.cpp"
    "${FLM_SOURCE_DIR}/common/AutoModel/model_backend.cpp"
    "${FLM_SOURCE_DIR}/common/AutoModel/modeling_llama3.cpp"
    "${FLM_SOURCE_DIR}/common/prefix_slot_cache.cpp"
    ${FLM_CHAT_SOURCES})
target_include_directories(test_llama_frontend PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${FLM_SOURCE_DIR}/include"
    "${FLM_SOURCE_DIR}/test/phi4_core"
    "${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp/include"
    "${BOOST_INCLUDE_DIR}"
    "${XRT_INCLUDE_DIR}")
target_compile_definitions(test_llama_frontend PRIVATE
    DEV_BUILD=1 __WINDOWS__ USEAVX2=1 DISABLE_ABI_CHECK=1
    _ENABLE_EXTENDED_ALIGNED_STORAGE WIN32_LEAN_AND_MEAN NOMINMAX)
target_compile_options(test_llama_frontend PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
target_link_directories(test_llama_frontend PRIVATE "${XRT_INCLUDE_DIR}/../lib")
target_link_libraries(test_llama_frontend PRIVATE xrt_coreutil flm_xgrammar)

include(CTest)
add_test(NAME test_llama_frontend COMMAND test_llama_frontend)
