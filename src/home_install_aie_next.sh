#!/usr/bin/env bash
#
# home_install_aie_next.sh — fast rebuild-and-run loop for the aie_next build.
#
# Rebuilds the aie_next engine libraries from their source tree (incremental),
# stages them into lib/xrt/aie_next with their headers, builds flm with the
# linux-aie-next preset (its own tree, build_aie_next/) and installs it to a
# user prefix through home_install.sh. Everything is incremental, so a second
# run after a one-line engine change takes seconds.
#
# Usage:
#   ./home_install_aie_next.sh                    # engines + flm + install
#   ./home_install_aie_next.sh run [tag] [args]   # ... then `flm run`
#                                                 #   (tag: qwen3.6-moe:35b-a3b)
#   ./home_install_aie_next.sh --no-engine ...    # use the .so already staged
#   ./home_install_aie_next.sh --no-build ...     # install/run what is built
#
# Environment:
#   FLM_ENGINE_SRC  engine source tree (default: ../../FastFlowLM_IRON beside
#                   this repo); without it the staged .so files are used as-is
#   FLM_CXX         engine compiler (default: g++-13, else g++)
#   FLM_PREFIX      install prefix (default: /scratch/$USER/flm_exe_aie_next,
#                   or ~/flm_exe_aie_next without scratch space)
#   FLM_MODEL_PATH  model root for `run` (default: whatever flm_env.sh sets)
#
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SRC_DIR/.." && pwd)"
ENGINE_SRC="${FLM_ENGINE_SRC:-$REPO_DIR/../FastFlowLM_IRON}"
LIB_DEST="$SRC_DIR/lib/xrt/aie_next"

DEFAULT_PREFIX="/scratch/$USER/flm_exe_aie_next"
[[ -d "/scratch/$USER" ]] || DEFAULT_PREFIX="$HOME/flm_exe_aie_next"
export FLM_PREFIX="${FLM_PREFIX:-$DEFAULT_PREFIX}"

DO_ENGINE=1
DO_BUILD=1
DO_RUN=0
RUN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-engine) DO_ENGINE=0 ;;
        --no-build) DO_BUILD=0; DO_ENGINE=0 ;;
        run) DO_RUN=1; shift; RUN_ARGS=("$@"); break ;;
        -h|--help) grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done
[[ ${#RUN_ARGS[@]} -gt 0 ]] || RUN_ARGS=(qwen3.6-moe:35b-a3b)

log() { echo "[aie_next] $*"; }

# ---- engines ---------------------------------------------------------------
if [[ "$DO_ENGINE" -eq 1 ]]; then
    if [[ ! -d "$ENGINE_SRC/FLM_DLL/detail/aie_next" ]]; then
        log "no engine source at $ENGINE_SRC (set FLM_ENGINE_SRC); using staged libs"
    else
        if [[ -z "${FLM_CXX:-}" ]]; then
            command -v g++-13 >/dev/null && FLM_CXX=g++-13 || FLM_CXX=g++
        fi
        export FLM_CXX
        dll="$ENGINE_SRC/FLM_DLL"
        jobs="$(nproc)"
        log "engines from $dll (FLM_CXX=$FLM_CXX)"
        make -s -C "$dll/detail/q4nx" NPU_PLATFORM=aie_next -j"$jobs"
        for unit in "$dll"/detail/aie_next/*/; do
            [[ -f "$unit/Makefile" ]] || continue
            make -s -C "$unit" NPU_PLATFORM=aie_next -j"$jobs"
        done

        # The engines only; q4nx is shared and already in lib/xrt.
        mkdir -p "$LIB_DEST"
        for so in "$dll"/build/aie_next/lib/*.so; do
            [[ "$(basename "$so")" == libq4_npu_eXpress.so ]] && continue
            install -m 0755 "$so" "$LIB_DEST/"
            log "  + lib/xrt/aie_next/$(basename "$so")"
        done
        # include/models/<m>/aie_next/*.hpp -> include/models/<m>/flm/aie_next/
        for hdr_dir in "$dll"/include/models/*/aie_next; do
            [[ -d "$hdr_dir" ]] || continue
            model="$(basename "$(dirname "$hdr_dir")")"
            mkdir -p "$SRC_DIR/include/models/$model/flm/aie_next"
            for hdr in "$hdr_dir"/*.hpp; do
                if ! cmp -s "$hdr" "$SRC_DIR/include/models/$model/flm/aie_next/$(basename "$hdr")"; then
                    install -m 0644 "$hdr" "$SRC_DIR/include/models/$model/flm/aie_next/"
                    log "  + include/models/$model/flm/aie_next/$(basename "$hdr")"
                fi
            done
        done
    fi
fi

# ---- flm, installed to the prefix --------------------------------------------
# home_install.sh configures (once), builds incrementally, installs and writes
# flm_env.sh. Its own build/ and ~/flm_exe are untouched.
install_args=()
[[ "$DO_BUILD" -eq 1 ]] || install_args+=(--no-build)
PRESET=linux-aie-next BUILD_DIR="$SRC_DIR/build_aie_next" \
    "$SRC_DIR/home_install.sh" ${install_args[@]+"${install_args[@]}"}

# ---- run -------------------------------------------------------------------
if [[ "$DO_RUN" -eq 1 ]]; then
    user_model_path="${FLM_MODEL_PATH:-}"
    # shellcheck disable=SC1091
    source "$FLM_PREFIX/flm_env.sh"
    [[ -n "$user_model_path" ]] && export FLM_MODEL_PATH="$user_model_path"
    log "flm run ${RUN_ARGS[*]}  (FLM_MODEL_PATH=${FLM_MODEL_PATH:-})"
    exec flm run "${RUN_ARGS[@]}"
fi
