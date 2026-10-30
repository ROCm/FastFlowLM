# Hardware-free frontend registry and real dynamic-loader tests.
cmake_minimum_required(VERSION 3.22)
project(model_frontend_tests LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
if(NOT DEFINED FLM_VERSION)
    set(FLM_VERSION "contract-test")
endif()

include("${FLM_SOURCE_DIR}/cmake/FastFlowLMModelExtension.cmake")
flm_add_model_extension_interface(
    flm_model_extension_contract)
add_library(fake_model_extension_dependency SHARED
    fake_model_extension_dependency.c)

set(FLM_TEST_EXTENSION_CLEANUP_PATH
    "${CMAKE_CURRENT_BINARY_DIR}/fake-model-extension-cleanup.txt")
if(FLM_USE_HRX)
    set(FLM_TEST_NPU_RUNTIME FLM_MODEL_EXTENSION_NPU_RUNTIME_HRX)
    set(FLM_TEST_INCOMPATIBLE_NPU_RUNTIME
        FLM_MODEL_EXTENSION_NPU_RUNTIME_XRT)
else()
    set(FLM_TEST_NPU_RUNTIME FLM_MODEL_EXTENSION_NPU_RUNTIME_XRT)
    set(FLM_TEST_INCOMPATIBLE_NPU_RUNTIME
        FLM_MODEL_EXTENSION_NPU_RUNTIME_HRX)
endif()

function(add_fake_model_extension target runtime mode)
    add_library("${target}" SHARED fake_model_extension.c)
    target_link_libraries("${target}" PRIVATE
        flm_model_extension_contract fake_model_extension_dependency)
    target_compile_definitions("${target}" PRIVATE
        FLM_TEST_EXTENSION_CLEANUP_PATH="${FLM_TEST_EXTENSION_CLEANUP_PATH}"
        FLM_TEST_NPU_RUNTIME=${runtime}
        FLM_TEST_EXTENSION_MODE=${mode})
    set_target_properties("${target}" PROPERTIES BUILD_RPATH "$ORIGIN")
endfunction()

set(FLM_TEST_EXTENSION_MODE_BACKEND 0)
set(FLM_TEST_EXTENSION_MODE_COMPLETE 1)
set(FLM_TEST_EXTENSION_MODE_REGISTRATION_FAILURE 2)
set(FLM_TEST_EXTENSION_MODE_NO_CAPABILITIES 3)
add_fake_model_extension(
    fake_model_extension "${FLM_TEST_NPU_RUNTIME}"
    "${FLM_TEST_EXTENSION_MODE_BACKEND}")
add_fake_model_extension(
    fake_incompatible_model_extension
    "${FLM_TEST_INCOMPATIBLE_NPU_RUNTIME}"
    "${FLM_TEST_EXTENSION_MODE_BACKEND}")
add_fake_model_extension(
    fake_complete_model_extension "${FLM_TEST_NPU_RUNTIME}"
    "${FLM_TEST_EXTENSION_MODE_COMPLETE}")
add_fake_model_extension(
    fake_failing_model_extension "${FLM_TEST_NPU_RUNTIME}"
    "${FLM_TEST_EXTENSION_MODE_REGISTRATION_FAILURE}")
add_fake_model_extension(
    fake_no_capability_model_extension "${FLM_TEST_NPU_RUNTIME}"
    "${FLM_TEST_EXTENSION_MODE_NO_CAPABILITIES}")
file(GENERATE
    OUTPUT "$<TARGET_FILE_DIR:fake_model_extension>/fake-model-extension.json"
    CONTENT "{\n  \"schema_version\": 1,\n  \"name\": \"fake-model-extension\",\n  \"version\": \"1.0.0\",\n  \"library\": \"$<TARGET_FILE_NAME:fake_model_extension>\",\n  \"catalog\": \"fake-model-catalog.json\"\n}\n")
file(GENERATE
    OUTPUT "$<TARGET_FILE_DIR:fake_incompatible_model_extension>/fake-incompatible-model-extension.json"
    CONTENT "{\n  \"schema_version\": 1,\n  \"name\": \"fake-model-extension\",\n  \"version\": \"1.0.0\",\n  \"library\": \"$<TARGET_FILE_NAME:fake_incompatible_model_extension>\"\n}\n")
file(GENERATE
    OUTPUT "$<TARGET_FILE_DIR:fake_complete_model_extension>/fake-complete-model-extension.json"
    CONTENT "{\n  \"schema_version\": 1,\n  \"name\": \"fake-model-extension\",\n  \"version\": \"1.0.0\",\n  \"library\": \"$<TARGET_FILE_NAME:fake_complete_model_extension>\",\n  \"catalog\": \"fake-complete-model-catalog.json\"\n}\n")
file(GENERATE
    OUTPUT "$<TARGET_FILE_DIR:fake_complete_model_extension>/fake-complete-model-catalog.json"
    CONTENT "{\n  \"schema_version\": 1,\n  \"models\": {\n    \"complete-plugin-model\": {\n      \"1b\": {\n        \"name\": \"Complete plugin model\",\n        \"supported_platforms\": [\"aie2p\"],\n        \"backend\": \"flm\",\n        \"backend_provider\": \"plugin-provider\",\n        \"frontend_provider\": \"plugin-provider\",\n        \"default_context_length\": 4096,\n        \"max_prefill_len\": 512,\n        \"details\": {\"family\": \"complete-plugin-test\"}\n      }\n    }\n  }\n}\n")
file(GENERATE
    OUTPUT "$<TARGET_FILE_DIR:fake_failing_model_extension>/fake-failing-model-extension.json"
    CONTENT "{\n  \"schema_version\": 1,\n  \"name\": \"fake-model-extension\",\n  \"version\": \"1.0.0\",\n  \"library\": \"$<TARGET_FILE_NAME:fake_failing_model_extension>\"\n}\n")
file(GENERATE
    OUTPUT "$<TARGET_FILE_DIR:fake_no_capability_model_extension>/fake-no-capability-model-extension.json"
    CONTENT "{\n  \"schema_version\": 1,\n  \"name\": \"fake-model-extension\",\n  \"version\": \"1.0.0\",\n  \"library\": \"$<TARGET_FILE_NAME:fake_no_capability_model_extension>\"\n}\n")
file(GENERATE
    OUTPUT "$<TARGET_FILE_DIR:fake_model_extension>/fake-model-catalog.json"
    CONTENT "{\n  \"schema_version\": 1,\n  \"models\": {\n    \"plugin-model\": {\n      \"1b\": {\n        \"name\": \"Plugin model\",\n        \"supported_platforms\": [\"aie2p\"],\n        \"backend\": \"flm\",\n        \"backend_provider\": \"plugin-provider\",\n        \"default_context_length\": 4096,\n        \"max_prefill_len\": 512,\n        \"details\": {\"family\": \"plugin-test\"}\n      }\n    }\n  }\n}\n")

add_executable(test_model_frontend
    test_model_frontend.cpp
    "${FLM_SOURCE_DIR}/common/AutoModel/model_frontend.cpp"
    "${FLM_SOURCE_DIR}/common/AutoModel/model_extension_backend.cpp"
    "${FLM_SOURCE_DIR}/common/AutoModel/model_backend.cpp")

# Compile the production AutoModel adapter in the hardware-free contract job.
# The loader test below substitutes only its factory so it can stay link-light;
# this target still catches C ABI/adapter drift on Linux and Windows.
add_library(model_extension_frontend_compile OBJECT
    "${FLM_SOURCE_DIR}/common/AutoModel/model_extension_frontend.cpp")
target_include_directories(model_extension_frontend_compile PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/stubs"
    "${FLM_SOURCE_DIR}/include"
)
if(WIN32)
    target_compile_definitions(model_extension_frontend_compile PRIVATE
        WIN32_LEAN_AND_MEAN NOMINMAX)
endif()
target_include_directories(test_model_frontend PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/stubs"
    "${FLM_SOURCE_DIR}/include"
)
target_compile_definitions(test_model_frontend PRIVATE
    FLM_TEST_EXTENSION_PATH="$<TARGET_FILE_DIR:fake_model_extension>/fake-model-extension.json"
    FLM_TEST_EXTENSION_DEPENDENCY_PATH="$<TARGET_FILE:fake_model_extension_dependency>"
    FLM_TEST_INCOMPATIBLE_EXTENSION_PATH="$<TARGET_FILE_DIR:fake_incompatible_model_extension>/fake-incompatible-model-extension.json"
    FLM_TEST_COMPLETE_EXTENSION_PATH="$<TARGET_FILE_DIR:fake_complete_model_extension>/fake-complete-model-extension.json"
    FLM_TEST_FAILING_EXTENSION_PATH="$<TARGET_FILE_DIR:fake_failing_model_extension>/fake-failing-model-extension.json"
    FLM_TEST_NO_CAPABILITY_EXTENSION_PATH="$<TARGET_FILE_DIR:fake_no_capability_model_extension>/fake-no-capability-model-extension.json"
    FLM_TEST_EXTENSION_CLEANUP_PATH="${FLM_TEST_EXTENSION_CLEANUP_PATH}"
    FLM_TEST_VALID_MANIFEST_FIXTURE="${CMAKE_CURRENT_LIST_DIR}/fixtures/valid-model-extension-v1.json"
    FLM_TEST_INVALID_MANIFEST_FIXTURE="${CMAKE_CURRENT_LIST_DIR}/fixtures/invalid-model-extension-v1.json"
    FLM_TEST_MANIFEST_SCHEMA="${FLM_SOURCE_DIR}/schemas/model-extension-manifest-v1.schema.json")
if(WIN32)
    target_compile_definitions(test_model_frontend PRIVATE
        WIN32_LEAN_AND_MEAN NOMINMAX)
endif()
target_link_libraries(test_model_frontend PRIVATE
    flm_model_extension_contract ${CMAKE_DL_LIBS})
add_dependencies(test_model_frontend
    fake_model_extension fake_incompatible_model_extension
    fake_complete_model_extension fake_failing_model_extension
    fake_no_capability_model_extension)

enable_testing()
add_test(NAME test_model_frontend COMMAND test_model_frontend)
