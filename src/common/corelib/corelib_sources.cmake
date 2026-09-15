include("${CMAKE_CURRENT_LIST_DIR}/../models/models_sources.cmake")

set(FLM_CORELIB_AIE4_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/corelib_api.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/corelib_runtime.cpp"
    ${FLM_MODELS_CORELIB_SOURCES})
