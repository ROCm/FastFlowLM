# Per-model sources, split by implementation. Each model lives in
# common/models/<model>/ with one folder per implementation:
#   <model>/flm/     - the original flm implementation, built into the flm binary
#   <model>/corelib/ - the parallel corelib implementation, built into
#                      flm_corelib_aie4 when FLM_ENABLE_CORELIB_AIE4 is on
# Adding a model means adding the folders, not editing this file.
file(GLOB FLM_MODELS_FLM_SOURCES     "${CMAKE_CURRENT_LIST_DIR}/*/flm/*.cpp")
file(GLOB FLM_MODELS_CORELIB_SOURCES "${CMAKE_CURRENT_LIST_DIR}/*/corelib/*.cpp")
