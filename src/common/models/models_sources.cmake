# Per-model sources, split by where the kernels come from and which silicon they
# run on. Those two axes are independent, so both appear in the path:
#   <model>/<backend>/<platform>/ - e.g. phi4/core/dev04/
# The core sources are built into flm_core when FLM_ENABLE_CORE is on. IRON
# sources are built into flm when FLM_ENABLE_IRON is on.
# FastFlowLM's own flow has no per-model sources here: those engines ship as
# prebuilt libraries under lib/<runtime>/.
# Adding a model means adding the folder, not editing this file.
file(GLOB FLM_MODELS_CORE_SOURCES "${CMAKE_CURRENT_LIST_DIR}/*/core/*/*.cpp")
file(GLOB FLM_MODELS_IRON_SOURCES "${CMAKE_CURRENT_LIST_DIR}/*/iron/*.cpp")
