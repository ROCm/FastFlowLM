#!/usr/bin/env bash
# Set up the environment for the MiniCPM-OCR-1.0 test harness.
#
#   export FLM_MODEL_PATH=<folder that holds models/MiniCPM-OCR-1.0-NPU2>
#   source activate.sh
#   make test                    # or: make test LENGTH=64
#
# The harness links the engine that ships in src/lib/xrt/ (built by IRON CI) and
# loads the xclbins from src/xclbins/MiniCPM-OCR-1.0-NPU2/. This script only
# checks that both are there and that FLM_MODEL_PATH points at the weights;
# it never replaces the shipped engine.
#
# model_list.json's "model_path" is "models", so FLM_MODEL_PATH is the PARENT
# of the models folder. Without it the harness looks in ~/.config/flm.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_src="$(cd "$_here/../.." && pwd)"
MODEL_NAME="${MODEL_NAME:-MiniCPM-OCR-1.0-NPU2}"

_fail() { echo "[activate] ERROR: $*" >&2; return 1; }

_main() {
    [ -f "$_src/lib/xrt/libminicpm_ocr_npu.so" ] || { _fail "missing $_src/lib/xrt/libminicpm_ocr_npu.so"; return 1; }
    [ -d "$_src/xclbins/$MODEL_NAME" ] || { _fail "missing $_src/xclbins/$MODEL_NAME"; return 1; }
    if [ -z "${FLM_MODEL_PATH:-}" ]; then
        echo "[activate] FLM_MODEL_PATH is not set: the harness will look in ~/.config/flm/models/$MODEL_NAME"
    elif [ ! -d "$FLM_MODEL_PATH/models/$MODEL_NAME" ]; then
        _fail "no weights at $FLM_MODEL_PATH/models/$MODEL_NAME"; return 1
    else
        export FLM_MODEL_PATH
        echo "[activate] FLM_MODEL_PATH=$FLM_MODEL_PATH"
    fi
    echo "[activate] ready: make test"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "[activate] run it as: source activate.sh" >&2
    exit 1
fi
_main
