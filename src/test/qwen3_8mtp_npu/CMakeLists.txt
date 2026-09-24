cmake_minimum_required(VERSION 3.22)
project(qwen3_8mtp_npu VERSION 1.0.0 LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../CMakeLists.txt)
npu_test_setup()

# Find and enable OpenMP for multi-threading
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling multi-threading support for test")
else()
    message(WARNING "OpenMP not found - some optimizations will run single-threaded")
endif()

add_npu_test(
    test_qwen3_8mtp_npu
    test/qwen3_8mtp_npu
    USE_AUTOMODEL
    USE_TOKENIZER
    USE_SAMPLER
    SOURCES
        "${CMAKE_SOURCE_DIR}/../../common/AutoModel/modeling_qwen3_8mtp.cpp"
)

# Add OpenMP compiler flags if available
if(OpenMP_CXX_FOUND)
    target_compile_options(test_qwen3_8mtp_npu PUBLIC
        $<$<CXX_COMPILER_ID:MSVC>:/openmp>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fopenmp>
    )
endif()

target_link_libraries(test_qwen3_8mtp_npu PUBLIC
    qwen3_8mtp_npu
    xrt_coreutil
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_qwen3_8mtp_npu PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add test target
add_custom_target(test_qwen3_8mtp_npu_target
    DEPENDS test_qwen3_8mtp_npu
    COMMENT "Building test_qwen3_8mtp_npu executable"
)
