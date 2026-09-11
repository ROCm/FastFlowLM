cmake_minimum_required(VERSION 3.22)
project(phi4_corelib_aie4_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT WIN32)
  message(FATAL_ERROR "Phi-4 corelib AIE4 tests currently require Windows")
endif()

find_path(RYZENAI_CORELIB_INCLUDE_DIR NAMES ryzenai/corelib.h REQUIRED)
set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
set(CORELIB_SOURCES
    "${FLM_SOURCE_DIR}/common/corelib/corelib_api.cpp"
    "${FLM_SOURCE_DIR}/common/corelib/corelib_runtime.cpp")

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

# Feature-off frontend code must compile without the pinned corelib include path.
add_library(phi4_frontend_compile_off OBJECT compile_guard.cpp)
target_include_directories(phi4_frontend_compile_off PRIVATE "${FLM_SOURCE_DIR}/include")

# Feature-on frontend code sees both the feature define and pinned ABI headers.
add_library(phi4_frontend_compile_on OBJECT compile_guard.cpp)
target_include_directories(phi4_frontend_compile_on PRIVATE
    "${FLM_SOURCE_DIR}/include" "${RYZENAI_CORELIB_INCLUDE_DIR}")
target_compile_definitions(phi4_frontend_compile_on PRIVATE
    FLM_ENABLE_CORELIB_AIE4=1 RYZENAI_CORELIB_STATIC=1)

include(CTest)
add_test(NAME test_corelib_api COMMAND test_corelib_api)
add_test(NAME test_real_corelib COMMAND test_real_corelib)
set_tests_properties(test_real_corelib PROPERTIES SKIP_RETURN_CODE 77)
