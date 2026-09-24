# Per-model sources, split by where the kernels come from and which silicon they
# run on. Those two axes are independent, so both appear in the path:
#   <model>/<backend>/<platform>/ - e.g. phi4/rai/aie_next/
# The rai sources are built into flm_rai when FLM_ENABLE_RAI is on.
# FastFlowLM's own flow has no per-model sources here: those engines ship as
# prebuilt libraries under lib/<runtime>/.
# Adding a model means adding the folder, not editing this file.
file(GLOB FLM_MODELS_RAI_SOURCES "${CMAKE_CURRENT_LIST_DIR}/*/rai/*/*.cpp")
