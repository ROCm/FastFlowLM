# Per-model sources, split by the hardware they run on. Each model lives in
# common/models/<model>/ with one folder per NPU generation:
#   <model>/aie2p/ - FastFlowLM's own NPU kernels, built into the flm binary
#   <model>/aie4/  - the ryzenai-corelib implementation, built into flm_aie4
#                    when FLM_ENABLE_AIE4 is on
# Adding a model means adding the folders, not editing this file.
file(GLOB FLM_MODELS_AIE2P_SOURCES "${CMAKE_CURRENT_LIST_DIR}/*/aie2p/*.cpp")
file(GLOB FLM_MODELS_AIE4_SOURCES  "${CMAKE_CURRENT_LIST_DIR}/*/aie4/*.cpp")
