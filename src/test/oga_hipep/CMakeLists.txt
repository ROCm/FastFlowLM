# The oga-hipep backend is Windows only, and so are its tests. test_oga_hipep
# runs on any Windows host: the OGA C API is a fake. The XRT and Boost headers
# are needed because causal_lm.hpp reaches them through the device runtime.
cmake_minimum_required(VERSION 3.22)
project(oga_hipep_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT WIN32)
  message(FATAL_ERROR "oga-hipep tests require Windows")
endif()

find_path(BOOST_INCLUDE_DIR NAMES boost/any.hpp
    HINTS "$ENV{CONDA_PREFIX}/Library/include"
          "$ENV{USERPROFILE}/anaconda3/Library/include" REQUIRED)
find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
    HINTS "$ENV{XRT_INCLUDE_DIR}"
          "C:/dev/XRT/src/runtime_src/core/include" REQUIRED)
set(XRT_LIB_DIR "${XRT_INCLUDE_DIR}/../lib" CACHE PATH "Where xrt_coreutil.lib lives")
set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

set(OGA_HIPEP_SOURCES
    "${FLM_SOURCE_DIR}/common/oga/oga_api.cpp"
    "${FLM_SOURCE_DIR}/common/oga/oga_engine.cpp"
    "${FLM_SOURCE_DIR}/common/oga/oga_hipep_backend.cpp")

include(CTest)

function(add_oga_hipep_test name)
    add_executable(${name} ${name}.cpp ${OGA_HIPEP_SOURCES})
    target_include_directories(${name} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/../phi4_core"
        "${FLM_SOURCE_DIR}/include"
        "${BOOST_INCLUDE_DIR}"
        "${XRT_INCLUDE_DIR}")
    target_compile_definitions(${name} PRIVATE
        USEAVX2=1 DISABLE_ABI_CHECK=1 _ENABLE_EXTENDED_ALIGNED_STORAGE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(${name} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
    target_link_directories(${name} PRIVATE "${XRT_LIB_DIR}")
    target_link_libraries(${name} PRIVATE xrt_coreutil version)
endfunction()

add_oga_hipep_test(test_oga_hipep)
add_test(NAME test_oga_hipep COMMAND test_oga_hipep)
