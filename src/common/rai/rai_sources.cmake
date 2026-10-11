include("${CMAKE_CURRENT_LIST_DIR}/../models/models_sources.cmake")

set(FLM_RAI_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/corelib_api.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/corelib_model.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/corelib_runtime.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/gguf_file.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/host_ops.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/weight_cache.cpp"
    ${FLM_MODELS_RAI_SOURCES})
