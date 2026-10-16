cmake_minimum_required(VERSION 3.22)
project(minicpm_v_4_7_npu VERSION 1.0.0 LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../CMakeLists.txt)
npu_test_setup()

# Find and enable OpenMP for multi-threading
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling multi-threading support for test")
else()
    message(WARNING "OpenMP not found - some optimizations will run single-threaded")
endif()

# Same image sources as the Makefile: see the note there on why imageproc.cpp is
# linked although MiniCPM resizes with its own PIL-exact code.
add_npu_test(
    test_minicpm_v_4_7_npu
    test/minicpm_v_4_7_npu
    USE_AUTOMODEL
    USE_TOKENIZER
    USE_SAMPLER
    USE_AVX512
    SOURCES
        "${CMAKE_SOURCE_DIR}/../../common/AutoModel/model_backend.cpp"
        "${CMAKE_SOURCE_DIR}/../../common/AutoModel/modeling_minicpm_v.cpp"
        "${CMAKE_SOURCE_DIR}/../../common/AutoModel/modeling_minicpm_v_image.cpp"
        "${CMAKE_SOURCE_DIR}/../../common/image_process_utils/imageproc.cpp"
        "${CMAKE_SOURCE_DIR}/../../common/image_process_utils/imageprocAVX512.cpp"
        "${CMAKE_SOURCE_DIR}/../../common/image/image_reader.cpp"
)

target_compile_definitions(test_minicpm_v_4_7_npu PUBLIC
    USEAVX512=1
)

# Add OpenMP compiler flags if available
if(OpenMP_CXX_FOUND)
    target_compile_options(test_minicpm_v_4_7_npu PUBLIC
        $<$<CXX_COMPILER_ID:MSVC>:/openmp>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fopenmp>
    )
endif()

target_link_libraries(test_minicpm_v_4_7_npu PUBLIC
    minicpm_v_4_7_npu
    xrt_coreutil
    avformat
    avcodec
    avutil
    swscale
    zlib
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_minicpm_v_4_7_npu PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add test target
add_custom_target(test_minicpm_v_4_7_npu_target
    DEPENDS test_minicpm_v_4_7_npu
    COMMENT "Building test_minicpm_v_4_7_npu executable"
)
