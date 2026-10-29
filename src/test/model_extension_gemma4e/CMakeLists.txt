# Complete external-provider fixture built against the installed extension SDK.
cmake_minimum_required(VERSION 3.22)
project(gemma4_open_model_extension LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT WIN32)
  message(FATAL_ERROR
    "The Gemma4-open integration provider supports Linux and Windows")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  # The packaged Windows AIEBU library uses the static CRT. Keep every
  # provider-owned implementation dependency on that same runtime.
  set(CMAKE_MSVC_RUNTIME_LIBRARY
    "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

find_package(FastFlowLMModelExtension CONFIG REQUIRED)
get_filename_component(_default_flm_source
  "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(FLM_SOURCE_DIR "${_default_flm_source}"
  CACHE PATH "FastFlowLM implementation sources used by this integration test")
if(NOT EXISTS "${FLM_SOURCE_DIR}/detail/gemma4e_npu/gemma4e_npu.cpp")
  message(FATAL_ERROR
    "FLM_SOURCE_DIR does not contain the Gemma4-open implementation sources")
endif()
set(FLM_TOKENIZERS_CPP_DIR
  "${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp"
  CACHE PATH "tokenizers-cpp source directory")
if(NOT EXISTS "${FLM_TOKENIZERS_CPP_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "FLM_TOKENIZERS_CPP_DIR must name an initialized tokenizers-cpp source tree")
endif()

set(BUNDLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/bundle")
set(FLM_GEMMA4_XCLBIN_DIR "" CACHE PATH
  "Directory containing the Gemma4-E2B-IT-NPU2 XCLBINs to package")
if(NOT IS_DIRECTORY "${FLM_GEMMA4_XCLBIN_DIR}")
  message(FATAL_ERROR
    "FLM_GEMMA4_XCLBIN_DIR must name the Gemma4-E2B-IT-NPU2 XCLBIN directory")
endif()

if(WIN32)
  set(XRT_INCLUDE_DIR "" CACHE PATH "Directory containing XRT headers")
  set(XRT_LIBRARY_DIR "" CACHE PATH "Directory containing xrt_coreutil.lib")
  set(BOOST_INCLUDE_DIR "" CACHE PATH "Directory containing Boost headers")
  set(VCPKG_INCLUDE_DIR "" CACHE PATH
    "Directory containing the Windows media-codec dependency headers")
  foreach(_required_dir
      XRT_INCLUDE_DIR XRT_LIBRARY_DIR BOOST_INCLUDE_DIR VCPKG_INCLUDE_DIR)
    if(NOT IS_DIRECTORY "${${_required_dir}}")
      message(FATAL_ERROR "${_required_dir} must name an existing directory")
    endif()
  endforeach()
  set(TOKENIZERS_CPP_RUST_FLAGS
    "-C target-feature=+crt-static"
    CACHE STRING "Rust flags for provider-owned tokenizers-cpp" FORCE)
  set(CARGO_EXTRA_ENVS
    "CFLAGS_x86_64_pc_windows_msvc=/MT"
    CACHE STRING "C flags for Rust crate native dependencies" FORCE)
endif()

add_subdirectory(
  "${FLM_TOKENIZERS_CPP_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/tokenizers-cpp"
  EXCLUDE_FROM_ALL)
if(WIN32)
  # FLM itself uses the dynamic CRT, while this provider must match the
  # packaged static-CRT AIEBU library. Generate provider-local copies of the
  # codec recipes with a consistent runtime; no FLM source is modified.
  file(READ "${FLM_SOURCE_DIR}/cmake/FlmImageCodecs.cmake"
    _flm_image_codecs)
  string(REPLACE "-DWITH_CRT_DLL=ON" "-DWITH_CRT_DLL=OFF"
    _flm_image_codecs "${_flm_image_codecs}")
  string(REPLACE
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL"
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
    _flm_image_codecs "${_flm_image_codecs}")
  string(REPLACE "\${CMAKE_CURRENT_LIST_DIR}/.."
    "${FLM_SOURCE_DIR}" _flm_image_codecs "${_flm_image_codecs}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/FlmImageCodecs.cmake"
    "${_flm_image_codecs}")
  file(READ "${FLM_SOURCE_DIR}/cmake/FlmAudioCodecs.cmake"
    _flm_audio_codecs)
  string(REPLACE
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON\n    -DCMAKE_POLICY_DEFAULT_CMP0091=NEW\n    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
    _flm_audio_codecs "${_flm_audio_codecs}")
  string(REPLACE
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON\n    -DCMAKE_POLICY_DEFAULT_CMP0091=NEW"
    _flm_image_codecs "${_flm_image_codecs}")
  string(REPLACE "\${CMAKE_CURRENT_LIST_DIR}/.."
    "${FLM_SOURCE_DIR}" _flm_audio_codecs "${_flm_audio_codecs}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/FlmAudioCodecs.cmake"
    "${_flm_audio_codecs}")
  include("${CMAKE_CURRENT_BINARY_DIR}/FlmImageCodecs.cmake")
  include("${CMAKE_CURRENT_BINARY_DIR}/FlmAudioCodecs.cmake")
else()
  include("${FLM_SOURCE_DIR}/cmake/FlmImageCodecs.cmake")
  include("${FLM_SOURCE_DIR}/cmake/FlmAudioCodecs.cmake")
endif()
if(NOT WIN32)
  find_program(CURL_EXECUTABLE curl REQUIRED)
endif()
find_package(OpenMP REQUIRED)

add_library(gemma4_open_model_extension SHARED
  model_extension.cpp
  gemma4_open_frontend.cpp
  "${FLM_SOURCE_DIR}/common/chat/chat_continue.cpp"
  "${FLM_SOURCE_DIR}/common/image/image_reader.cpp"
  "${FLM_SOURCE_DIR}/common/image_process_utils/imageproc.cpp"
  "${FLM_SOURCE_DIR}/common/image_process_utils/imageprocAVX512.cpp"
  "${FLM_SOURCE_DIR}/common/audio/audio_reader.cpp"
  "${FLM_SOURCE_DIR}/common/audio/audio_decoder.cpp"
  "${FLM_SOURCE_DIR}/common/audio/audio_resampler.cpp"
  "${FLM_SOURCE_DIR}/common/audio_process_utils/audioproc.cpp"
  "${FLM_SOURCE_DIR}/common/audio_process_utils/audioprocAVX512.cpp")
file(GLOB GEMMA4_OPEN_ENGINE_SOURCES
  "${FLM_SOURCE_DIR}/detail/gemma4e_npu/*.cpp"
  "${FLM_SOURCE_DIR}/detail/dequant/*.cpp"
  "${FLM_SOURCE_DIR}/detail/gemm/*.cpp"
  "${FLM_SOURCE_DIR}/detail/lm_head/*.cpp")
list(APPEND GEMMA4_OPEN_ENGINE_SOURCES
  "${FLM_SOURCE_DIR}/detail/vision_common/norm.cpp")
target_sources(gemma4_open_model_extension PRIVATE
  ${GEMMA4_OPEN_ENGINE_SOURCES})
target_include_directories(gemma4_open_model_extension PRIVATE
  "${FLM_SOURCE_DIR}/include"
  "${FLM_SOURCE_DIR}/detail")
if(WIN32)
  target_include_directories(gemma4_open_model_extension PRIVATE
    "${XRT_INCLUDE_DIR}"
    "${BOOST_INCLUDE_DIR}"
    "${VCPKG_INCLUDE_DIR}")
endif()
target_compile_definitions(gemma4_open_model_extension PRIVATE
  VERBOSE=0
  DEBUG_LEVEL=0)
if(WIN32)
  target_compile_definitions(gemma4_open_model_extension PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
    _CRT_NONSTDC_NO_DEPRECATE
    __WINDOWS__)
  target_compile_options(gemma4_open_model_extension PRIVATE /arch:AVX512)
  set_target_properties(gemma4_open_model_extension PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
  target_compile_options(gemma4_open_model_extension PRIVATE
    -O3 -Wall -fmax-errors=1
    -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma
    -ffast-math)
  target_link_options(gemma4_open_model_extension PRIVATE
    -Wl,-Bsymbolic-functions
    -Wl,--no-undefined)
endif()
target_link_libraries(gemma4_open_model_extension PRIVATE
  FastFlowLM::ModelExtension
  tokenizers_cpp
  flm_image_codecs
  flm_audio_codecs
  OpenMP::OpenMP_CXX)
set_target_properties(gemma4_open_model_extension PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_RPATH "$ORIGIN"
  RUNTIME_OUTPUT_DIRECTORY "${BUNDLE_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${BUNDLE_DIR}/lib")
foreach(_config RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)
  set_target_properties(gemma4_open_model_extension PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${_config} "${BUNDLE_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY_${_config} "${BUNDLE_DIR}/lib")
endforeach()

if(FastFlowLMModelExtension_NPU_RUNTIME STREQUAL "hrx")
  target_compile_definitions(gemma4_open_model_extension PRIVATE FLM_USE_HRX=1)
  find_package(hrx CONFIG REQUIRED)
  set(FLM_ENGINE_LIBRARY_DIR "${FLM_SOURCE_DIR}/lib/hrx")
  target_link_libraries(gemma4_open_model_extension PRIVATE hrx::hrx)
elseif(FastFlowLMModelExtension_NPU_RUNTIME STREQUAL "xrt")
  set(FLM_ENGINE_LIBRARY_DIR "${FLM_SOURCE_DIR}/lib/xrt")
  if(WIN32)
    find_library(XRT_COREUTIL_LIBRARY NAMES xrt_coreutil
      HINTS "${XRT_LIBRARY_DIR}" REQUIRED NO_DEFAULT_PATH)
    find_library(AIEBU_LIBRARY NAMES aiebu_static
      HINTS "${FLM_ENGINE_LIBRARY_DIR}" REQUIRED NO_DEFAULT_PATH)
  else()
    find_library(XRT_COREUTIL_LIBRARY NAMES xrt_coreutil REQUIRED)
    find_library(AIEBU_LIBRARY NAMES aiebu
      HINTS "${FLM_ENGINE_LIBRARY_DIR}" REQUIRED)
  endif()
  target_link_libraries(gemma4_open_model_extension PRIVATE
    "${AIEBU_LIBRARY}" "${XRT_COREUTIL_LIBRARY}")
else()
  message(FATAL_ERROR
    "Unsupported FastFlowLM runtime: ${FastFlowLMModelExtension_NPU_RUNTIME}")
endif()

find_library(Q4_NPU_EXPRESS_LIBRARY NAMES q4_npu_eXpress
  HINTS "${FLM_ENGINE_LIBRARY_DIR}" REQUIRED)
target_link_libraries(gemma4_open_model_extension PRIVATE
  "${Q4_NPU_EXPRESS_LIBRARY}")
if(WIN32)
  find_library(MHA_LIBRARY NAMES mha
    HINTS "${FLM_ENGINE_LIBRARY_DIR}" REQUIRED)
  target_link_libraries(gemma4_open_model_extension PRIVATE "${MHA_LIBRARY}")
endif()

file(MAKE_DIRECTORY "${BUNDLE_DIR}/catalog")
if(WIN32)
  file(COPY
    "${FLM_ENGINE_LIBRARY_DIR}/q4_npu_eXpress.dll"
    "${FLM_ENGINE_LIBRARY_DIR}/mha.dll"
    DESTINATION "${BUNDLE_DIR}/lib")
else()
  file(COPY "${Q4_NPU_EXPRESS_LIBRARY}" DESTINATION "${BUNDLE_DIR}/lib")
endif()
file(COPY "${FLM_GEMMA4_XCLBIN_DIR}/"
  DESTINATION "${BUNDLE_DIR}/xclbins/Gemma4-E2B-IT-NPU2")
configure_file(model-list.json
  "${BUNDLE_DIR}/catalog/model-list.json" COPYONLY)
file(GENERATE OUTPUT "${BUNDLE_DIR}/extension.json"
  CONTENT
"{
  \"schema_version\": 1,
  \"name\": \"gemma4-open\",
  \"version\": \"1.0.0\",
  \"library\": \"lib/$<TARGET_FILE_NAME:gemma4_open_model_extension>\",
  \"catalog\": \"catalog/model-list.json\"
}
")
