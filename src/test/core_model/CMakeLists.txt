cmake_minimum_required(VERSION 3.22)
project(core_model_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT WIN32)
  message(FATAL_ERROR "core model tests currently require Windows")
endif()

find_path(RYZENAI_CORELIB_INCLUDE_DIR NAMES ryzenai/corelib.h REQUIRED)
find_path(BOOST_INCLUDE_DIR NAMES boost/program_options.hpp
    HINTS "$ENV{CONDA_PREFIX}/Library/include"
          "$ENV{USERPROFILE}/anaconda3/Library/include" REQUIRED)
find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
    HINTS "$ENV{XRT_INCLUDE_DIR}"
          "C:/dev/XRT/src/runtime_src/core/include" REQUIRED)
set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

# The fake corelib is phi4_core's; it fakes the model API as well as the ops.
set(CORE_MODEL_TEST_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/../phi4_core/fake_corelib.cpp"
    "${FLM_SOURCE_DIR}/common/core/corelib_api.cpp"
    "${FLM_SOURCE_DIR}/common/core/corelib_runtime.cpp"
    "${FLM_SOURCE_DIR}/common/core/host_ops.cpp"
    "${FLM_SOURCE_DIR}/common/core/corelib_model.cpp")

include(CTest)

function(add_core_model_test name)
    add_executable(${name} ${name}.cpp ${CORE_MODEL_TEST_SOURCES} ${ARGN})
    target_include_directories(${name} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}"
        "${FLM_SOURCE_DIR}/include"
        "${RYZENAI_CORELIB_INCLUDE_DIR}"
        "${BOOST_INCLUDE_DIR}"
        "${XRT_INCLUDE_DIR}")
    target_compile_definitions(${name} PRIVATE
        RYZENAI_CORELIB_STATIC=1 USEAVX2=1 DISABLE_ABI_CHECK=1
        _ENABLE_EXTENDED_ALIGNED_STORAGE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(${name} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
    target_link_directories(${name} PRIVATE "${XRT_INCLUDE_DIR}/../lib")
    target_link_libraries(${name} PRIVATE xrt_coreutil)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_core_model_test(test_corelib_model)
add_core_model_test(test_corelib_model_backend
    "${FLM_SOURCE_DIR}/common/core/corelib_model_backend.cpp")
add_core_model_test(test_phi4_core_backend
    "${FLM_SOURCE_DIR}/common/core/corelib_model_backend.cpp"
    "${FLM_SOURCE_DIR}/common/models/phi4/core/dev04/phi4_core_backend.cpp")
add_core_model_test(test_llama_core_backend
    "${FLM_SOURCE_DIR}/common/core/corelib_model_backend.cpp"
    "${FLM_SOURCE_DIR}/common/models/llama/core/dev04/llama_core_backend.cpp")
