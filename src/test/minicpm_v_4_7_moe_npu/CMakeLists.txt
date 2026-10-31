cmake_minimum_required(VERSION 3.22)
project(minicpm_v_4_7_moe_npu VERSION 1.0.0 LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../CMakeLists.txt)
npu_test_setup()

add_npu_test(
    test_minicpm_v_4_7_moe_npu
    test/minicpm_v_4_7_moe_npu
    USE_AUTOMODEL
    USE_TOKENIZER
    USE_SAMPLER
    SOURCES
        "${CMAKE_SOURCE_DIR}/../../common/AutoModel/model_backend.cpp"
        "${CMAKE_SOURCE_DIR}/../../common/AutoModel/modeling_minicpm_v_moe.cpp"
)

target_link_libraries(test_minicpm_v_4_7_moe_npu PUBLIC
    minicpm_v_4_7_moe_npu
    xrt_coreutil
)

# Add test target
add_custom_target(test_minicpm_v_4_7_moe_npu_target
    DEPENDS test_minicpm_v_4_7_moe_npu
    COMMENT "Building test_minicpm_v_4_7_moe_npu executable"
)
