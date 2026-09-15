# Catalog-only tests: no NPU, no XRT runtime library, so this builds and runs on
# Linux CI as well as Windows. The XRT headers are still needed because
# npu_platform.cpp talks to the device when one is handed to it.
cmake_minimum_required(VERSION 3.22)
project(model_list_platform_tests LANGUAGES CXX)

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

add_executable(test_model_list_platform
    test_model_list_platform.cpp
    "${FLM_SOURCE_DIR}/common/npu_platform.cpp")
target_include_directories(test_model_list_platform PRIVATE
    "${FLM_SOURCE_DIR}/include"
    "${XRT_INCLUDE_DIR}")
target_compile_definitions(test_model_list_platform PRIVATE
    FLM_TEST_MODEL_LIST_PATH="${FLM_SOURCE_DIR}/model_list.json")
# get_platform_columns calls into XRT, so the coreutil library is needed even
# though these tests never open a device.
if(WIN32)
    target_link_directories(test_model_list_platform PRIVATE "${XRT_INCLUDE_DIR}/../lib")
elseif(XRT_FOUND)
    target_link_directories(test_model_list_platform PRIVATE "${XRT_LIBRARY_DIRS}")
else()
    target_link_directories(test_model_list_platform PRIVATE /opt/xilinx/xrt/lib)
endif()
target_link_libraries(test_model_list_platform PRIVATE xrt_coreutil)

enable_testing()
add_test(NAME test_model_list_platform COMMAND test_model_list_platform)
