# The FLMGEMM operator override, built as a plugin flm loads through FLM_PLUGIN.
# Nothing links against it: it is dlopened at run time.

add_library(iron_gemm_plugin SHARED iron_gemm_plugin.cpp)

set_target_properties(iron_gemm_plugin PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    PREFIX ""
)

# Mandatory for any plugin. npu_app and the classes around it are header only,
# so every engine library carries its own copy of their inline code as weak
# symbols. Without -Bsymbolic-functions the plugin's calls bind to whichever
# library the loader saw first, which is one built against a different revision
# of those headers, and the object it constructs is laid out for someone else.
if(NOT WIN32)
    target_link_options(iron_gemm_plugin PRIVATE -Wl,-Bsymbolic-functions)
endif()

target_include_directories(iron_gemm_plugin PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

if(NOT FLM_USE_HRX)
    if(NOT WIN32 AND XRT_FOUND)
        target_include_directories(iron_gemm_plugin PRIVATE ${XRT_INCLUDE_DIRS})
        target_link_directories(iron_gemm_plugin PRIVATE ${XRT_LIBRARY_DIRS})
    else()
        target_include_directories(iron_gemm_plugin PRIVATE ${XRT_INCLUDE_DIR})
        target_link_directories(iron_gemm_plugin PRIVATE ${XRT_LIB_DIR})
    endif()
endif()

target_link_directories(iron_gemm_plugin PRIVATE ${FLM_ENGINE_LIB_DIR})

# SafeTensors reads the packed weight sidecars; the rest of the override API is
# header only.
# gemm is the shipped mm overlay's sequence generator, which IRON_GEMM_MODE=bf16
# drives directly so the port's kernel can be taken out of a comparison.
target_link_libraries(iron_gemm_plugin PRIVATE q4_npu_eXpress gemm)

if(FLM_USE_HRX)
    target_link_libraries(iron_gemm_plugin PRIVATE hrx::hrx)
    target_compile_definitions(iron_gemm_plugin PRIVATE FLM_USE_HRX=1)
elseif(NOT WIN32)
    target_link_libraries(iron_gemm_plugin PRIVATE xrt_coreutil aiebu)
endif()
