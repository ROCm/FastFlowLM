cmake_minimum_required(VERSION 3.22)
project(minicpm_ocr_npu VERSION 1.0.0 LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../CMakeLists.txt)
npu_test_setup()

add_npu_test(
    test_minicpm_ocr_npu
    test/minicpm_ocr_npu
    USE_AUTOMODEL
    USE_TOKENIZER
    USE_SAMPLER
    SOURCES
        "${CMAKE_SOURCE_DIR}/../../common/AutoModel/model_backend.cpp"
        "${CMAKE_SOURCE_DIR}/../../common/AutoModel/modeling_minicpm_ocr.cpp"
)

target_link_libraries(test_minicpm_ocr_npu PUBLIC
    minicpm_ocr_npu
    xrt_coreutil
)

# Add test target
add_custom_target(test_minicpm_ocr_npu_target
    DEPENDS test_minicpm_ocr_npu
    COMMENT "Building test_minicpm_ocr_npu executable"
)
