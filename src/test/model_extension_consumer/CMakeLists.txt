cmake_minimum_required(VERSION 3.22)
project(model_extension_consumer LANGUAGES C)

# Proves that a separately configured project can consume the installed
# stable-C extension package without access to the FastFlowLM source tree.
find_package(FastFlowLMModelExtension CONFIG REQUIRED)
if(NOT EXISTS
   "${FastFlowLMModelExtension_SCHEMA_DIR}/model-extension-manifest-v1.schema.json")
    message(FATAL_ERROR "Installed model-extension schemas are unavailable")
endif()
if(NOT FastFlowLMModelExtension_NPU_RUNTIME MATCHES "^(xrt|hrx)$")
    message(FATAL_ERROR "Installed package has no valid NPU runtime identity")
endif()

add_library(installed_model_extension SHARED provider.c)
target_link_libraries(
    installed_model_extension PRIVATE FastFlowLM::ModelExtension)
