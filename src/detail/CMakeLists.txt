# Builds the Gemma4e engine from source, in place of the prebuilt
# lib/<backend>/libgemma4e_npu.so that ships for every other model.

set(GEMMA4E_ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB GEMMA4E_SOURCES "${GEMMA4E_ENGINE_DIR}/gemma4e_npu/*.cpp")
# Every engine compiles its own copy of these four; they also ship as separate
# libraries for flm itself to link.
file(GLOB GEMMA4E_SHARED_SOURCES
    "${GEMMA4E_ENGINE_DIR}/dequant/*.cpp"
    "${GEMMA4E_ENGINE_DIR}/gemm/*.cpp"
    "${GEMMA4E_ENGINE_DIR}/lm_head/*.cpp"
)
list(APPEND GEMMA4E_SHARED_SOURCES "${GEMMA4E_ENGINE_DIR}/vision_common/norm.cpp")

add_library(gemma4e_npu SHARED ${GEMMA4E_SOURCES} ${GEMMA4E_SHARED_SOURCES})

target_include_directories(gemma4e_npu BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)
if(NOT WIN32)
    target_include_directories(gemma4e_npu PRIVATE /opt/xilinx/xrt/include)
else()
    target_include_directories(gemma4e_npu PRIVATE ${XRT_INCLUDE_DIR})
endif()

target_compile_features(gemma4e_npu PRIVATE cxx_std_20)
target_compile_definitions(gemma4e_npu PRIVATE
    VERBOSE=${FLM_ENGINE_VERBOSE}
    DEBUG_LEVEL=${FLM_ENGINE_DEBUG_LEVEL}
)

if(NOT MSVC)
    target_compile_options(gemma4e_npu PRIVATE
        -O3 -Wall -fmax-errors=1
        -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma
        -ffast-math
    )
    # The Makefile builds with -march=native. It bakes the build host's ISA into
    # the library, so a redistributable build has to turn it off.
    if(FLM_ENGINE_NATIVE_ARCH)
        target_compile_options(gemma4e_npu PRIVATE -march=native)
    endif()
endif()

# Point FLM_OVERRIDES at a header that redefines FLM_OVERRIDE to dispatch an
# operator elsewhere; see include/flm_override.hpp. Empty leaves the build
# byte-identical to an unannotated one.
if(FLM_OVERRIDE_FLAGS)
    separate_arguments(_gemma4e_override_flags NATIVE_COMMAND "${FLM_OVERRIDE_FLAGS}")
    target_compile_options(gemma4e_npu PRIVATE ${_gemma4e_override_flags})
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(gemma4e_npu PRIVATE OpenMP::OpenMP_CXX)

target_link_directories(gemma4e_npu PRIVATE "${FLM_ENGINE_LIB_DIR}")
if(FLM_USE_HRX)
    target_link_libraries(gemma4e_npu PRIVATE hrx::hrx)
else()
    if(NOT WIN32)
        target_link_directories(gemma4e_npu PRIVATE /opt/xilinx/xrt/lib)
        target_link_libraries(gemma4e_npu PRIVATE xrt_coreutil aiebu)
    else()
        target_link_libraries(gemma4e_npu PRIVATE xrt_coreutil aiebu_static)
    endif()
    # The HRX path links no Boost: no engine source uses it there.
    find_package(Boost CONFIG REQUIRED COMPONENTS program_options filesystem)
    target_link_libraries(gemma4e_npu PRIVATE
        Boost::program_options
        Boost::filesystem
    )
endif()

if(NOT WIN32)
    # Each engine .so carries its own copy of the shared runtime (npu_sequence,
    # npu_app_manager, weight_desc_t, buffer, ...) at default visibility. flm
    # loads every engine at once, so without this flag ELF interposition binds
    # references -- including calls from inside this .so -- to whichever engine
    # comes first in DT_NEEDED, running another model's implementation.
    # Functions only: typeinfo and vtables stay interposable so dynamic_cast
    # across the boundary keeps working.
    target_link_options(gemma4e_npu PRIVATE -Wl,-Bsymbolic-functions)
    # q4nx and mha stay prebuilt; flm resolves them when it links the engine.
    target_link_options(gemma4e_npu PRIVATE -Wl,--allow-shlib-undefined)
endif()

# Built into the build tree, leaving lib/<backend>/libgemma4e_npu.so untouched
# so FLM_BUILD_GEMMA4E=OFF still has a prebuilt to fall back on. The parent adds
# this directory to flm's BUILD_RPATH and installs this target in place of the
# prebuilt.
set_target_properties(gemma4e_npu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/engines"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/engines"
)
foreach(_cfg RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)
    set_target_properties(gemma4e_npu PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${_cfg} "${CMAKE_BINARY_DIR}/engines"
        RUNTIME_OUTPUT_DIRECTORY_${_cfg} "${CMAKE_BINARY_DIR}/engines"
    )
endforeach()
