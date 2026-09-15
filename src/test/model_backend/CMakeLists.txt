# Registry-only tests: no NPU, no prebuilt engine library, so this builds and
# runs on Linux CI as well as Windows. The XRT headers are still needed because
# model_backend.hpp reaches causal_lm.hpp through the device runtime.
cmake_minimum_required(VERSION 3.22)
project(model_backend_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

if(WIN32)
    set(XRT_INCLUDE_DIR "C:/dev/XRT/src/runtime_src/core/include"
        CACHE PATH "Where XRT headers live")
else()
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(XRT xrt)
    endif()
    if(XRT_FOUND)
        set(XRT_INCLUDE_DIR "${XRT_INCLUDE_DIRS}")
    else()
        set(XRT_INCLUDE_DIR /opt/xilinx/xrt/include CACHE PATH "Where XRT headers live")
    endif()
endif()

add_executable(test_model_backend
    test_model_backend.cpp
    "${FLM_SOURCE_DIR}/common/AutoModel/model_backend.cpp")
target_include_directories(test_model_backend PRIVATE
    "${FLM_SOURCE_DIR}/include"
    "${XRT_INCLUDE_DIR}")

if(WIN32)
    target_link_directories(test_model_backend PRIVATE "${XRT_INCLUDE_DIR}/../lib")
elseif(XRT_FOUND)
    target_link_directories(test_model_backend PRIVATE "${XRT_LIBRARY_DIRS}")
else()
    target_link_directories(test_model_backend PRIVATE /opt/xilinx/xrt/lib)
endif()
target_link_libraries(test_model_backend PRIVATE xrt_coreutil)

enable_testing()
add_test(NAME test_model_backend COMMAND test_model_backend)
