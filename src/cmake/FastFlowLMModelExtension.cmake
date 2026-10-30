# Defines the build interface for native model extensions. Only
# AutoModel/model_extension.h is public; other headers visible in a source-tree
# build are FastFlowLM implementation details and are not part of this contract.
include_guard(GLOBAL)

# Define the interface consumed by an independently built model extension.
function(flm_add_model_extension_interface target)
    if(TARGET "${target}")
        message(FATAL_ERROR "Target '${target}' already exists")
    endif()
    get_filename_component(
        _flm_source_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)

    add_library("${target}" INTERFACE)
    target_include_directories("${target}" INTERFACE
        "$<BUILD_INTERFACE:${_flm_source_dir}/include>"
        "$<INSTALL_INTERFACE:include>")
    if(FLM_USE_HRX)
        target_compile_definitions("${target}" INTERFACE FLM_USE_HRX=1)
    endif()
    set_target_properties("${target}" PROPERTIES
        EXPORT_NAME ModelExtension)
endfunction()
