cmake_minimum_required(VERSION 3.22)
project(qwen3_6_moe_npu VERSION 1.0.0 LANGUAGES CXX)

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
    test_qwen3_6_moe_npu
    test/qwen3_6_moe_npu
    USE_AUTOMODEL
    USE_TOKENIZER
    USE_SAMPLER
    USE_AVX512
    SOURCES
        "${CMAKE_SOURCE_DIR}/../../common/AutoModel/modeling_qwen3_6_moe.cpp"
        "${CMAKE_SOURCE_DIR}/../../common/AutoModel/modeling_qwen3_6_moe_image.cpp"
        "${CMAKE_SOURCE_DIR}/../../common/image_process_utils/imageproc.cpp"
        "${CMAKE_SOURCE_DIR}/../../common/image_process_utils/imageprocAVX512.cpp"
        "${CMAKE_SOURCE_DIR}/../../common/image/image_reader.cpp"
)

target_compile_definitions(test_qwen3_6_moe_npu PUBLIC
    USEAVX512=1
)

# Add OpenMP compiler flags if available
if(OpenMP_CXX_FOUND)
    target_compile_options(test_qwen3_6_moe_npu PUBLIC
        $<$<CXX_COMPILER_ID:MSVC>:/openmp>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fopenmp>
    )
endif()

# Configure runtime linking to support both static Boost and dynamic tokenizers_cpp
if(MSVC)
    # Use dynamic runtime to match tokenizers_cpp library
    set_property(TARGET test_qwen3_6_moe_npu PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    set_property(TARGET test_qwen3_6_moe_npu APPEND_STRING PROPERTY LINK_FLAGS " /NODEFAULTLIB:LIBCMT")
endif()

target_link_libraries(test_qwen3_6_moe_npu PUBLIC
    qwen3_6_moe_npu
    xrt_coreutil
    aiebu_static
    libboost_program_options-vc143-mt-x64-1_88
    avformat
    avcodec
    avutil
    swscale
    zlib
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_qwen3_6_moe_npu PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add test target
add_custom_target(test_qwen3_6_moe_npu_target
    DEPENDS test_qwen3_6_moe_npu
    COMMENT "Building test_qwen3_6_moe_npu executable"
)
