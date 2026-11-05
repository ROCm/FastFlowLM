#!/usr/bin/env bash
# Set up the environment for the MiniCPM-V-4.7-1B test harness.
#
#   source activate.sh           # stage whatever engine is already built
#   source activate.sh host      # rebuild the all-host engine first (no NPU)
#   source activate.sh npu       # rebuild the all-NPU engine first  (S4/S5)
#
# then:
#   make test                    # or: make test LENGTH=64
#                                # (the thinking turn generates 4x LENGTH)
#
# Why this is needed: the harness links -lminicpm_ocr_npu out of src/lib/xrt/, but
# that engine is built in the *other* repo (FastFlowLM_IRON) and is deliberately
# not committed here -- it is a build artifact whose host/NPU flavour changes.
# Without staging it you get:
#     /usr/bin/ld: cannot find -lminicpm_ocr_npu: No such file or directory
#
# It also exports FLM_MODEL_PATH so the harness finds the model in /scratch
# rather than ~/.config/flm. model_list.json's "model_path" is "models", and
# model_root_path is FLM_MODEL_PATH/models, so point this at the PARENT of the
# model folders.

# --- settings, override by exporting before sourcing -------------------------
IRON_REPO="${IRON_REPO:-/scratch/michyu/Projects/FastFlowLM_IRON}"
FLM_MODEL_PATH="${FLM_MODEL_PATH:-/scratch/michyu}"
MODEL_NAME="${MODEL_NAME:-MiniCPM-OCR-1.0-NPU2}"

# Host build: every stage on the CPU. Correct but slow (~20 tok/s decode).
# The polarity is opt-OUT -- a plain `make` is all-NPU.
HOST_FLAGS="PREFILL_MM=cpu PREFILL_ATTN=cpu PREFILL_GDN=cpu PREFILL_CONV=cpu DECODE=cpu"

# Tolerate `bash activate.sh` as well as `source activate.sh`; only the latter
# makes the exports stick, so say so rather than silently doing half the job.
_sourced=1
[ "${BASH_SOURCE[0]}" = "${0}" ] && _sourced=0

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_libdir="$(cd "$_here/../../lib/xrt" && pwd)"
_engdir="$IRON_REPO/FLM_DLL/detail/minicpm_ocr_npu"
_built="$IRON_REPO/FLM_DLL/build/lib/libminicpm_ocr_npu.so"

_fail() { echo "[activate] ERROR: $*" >&2; return 1; }

_main() {
    [ -d "$_engdir" ] || { _fail "engine dir not found: $_engdir (set IRON_REPO)"; return 1; }

    case "${1:-}" in
        host) echo "[activate] building all-host engine ..."
              make -C "$_engdir" -j"$(nproc)" $HOST_FLAGS >/tmp/minicpm_ocr_activate_build.log 2>&1 \
                  || { tail -25 /tmp/minicpm_ocr_activate_build.log; _fail "build failed"; return 1; } ;;
        npu)  echo "[activate] building all-NPU engine ..."
              make -C "$_engdir" -j"$(nproc)" >/tmp/minicpm_ocr_activate_build.log 2>&1 \
                  || { tail -25 /tmp/minicpm_ocr_activate_build.log; _fail "build failed"; return 1; } ;;
        "")   ;;
        *)    _fail "unknown argument '$1' (expected: host, npu, or nothing)"; return 1 ;;
    esac

    [ -f "$_built" ] || { _fail "no engine built yet: $_built
         run:  source activate.sh host     (all-CPU, works without an NPU)
         or:   source activate.sh npu      (needs S4/S5 and the NPU box)"; return 1; }

    # Stale-.so check. build/ is gitignored and therefore per-worktree, so
    # several copies of this file exist on disk. AGENTS.md section 8: a stale
    # .so makes experiments lie. Compare against the newest engine source.
    local newest
    newest="$(find "$_engdir" -name '*.cpp' -o -name '*.hpp' | xargs ls -t 2>/dev/null | head -1)"
    if [ -n "$newest" ] && [ "$newest" -nt "$_built" ]; then
        echo "[activate] WARNING: engine is OLDER than $(basename "$newest")."
        echo "[activate]          rebuild with: source activate.sh host|npu"
    fi

    install -m 755 "$_built" "$_libdir/libminicpm_ocr_npu.so" || { _fail "could not stage into $_libdir"; return 1; }
    echo "[activate] staged $(basename "$_built") -> src/lib/xrt/  ($(date -r "$_built" '+%H:%M'))"

    # Report which flavour was staged, so a surprising tok/s is explained before
    # it is chased. A -D define leaves no trace in the binary, so test for the
    # "layer.xclbin" literal instead: its register_xclbin() call sits inside
    # #if MINICPM_OCR_DECODE_ON_NPU, so the host build does not contain it.
    # (Do not test lm_head.xclbin -- that string appears in both, from the
    # shared detail/lm_head objects linked into every engine.)
    if strings "$_libdir/libminicpm_ocr_npu.so" | grep -q 'layer\.xclbin'; then
        echo "[activate] flavour: NPU (expect ~44-47 tok/s decode in make test)"
        # The harness is built with -DDEV_BUILD, which hardwires LM_Config's
        # exec_path to "../../../" relative to its cwd -- i.e. src/ -- so it reads
        # src/xclbins/<model>/ and IGNORES FLM_XCLBIN_PATH. Check that directory.
        local xdir="$_here/../../xclbins/$MODEL_NAME"
        local x missing=""
        for x in layer lm_head mm attn conv GateDeltaNet_prefill; do
            [ -f "$xdir/$x.xclbin" ] || missing="$missing $x.xclbin"
        done
        if [ -n "$missing" ]; then
            echo "[activate] WARNING: missing in src/xclbins/$MODEL_NAME/:$missing"
            echo "[activate]          the text tower reuses Qwen3.5-0.8B's bitstreams unchanged:"
            echo "[activate]          mkdir -p $xdir && cp $_here/../../xclbins/Qwen3.5-0.8B-NPU2/{layer,lm_head,mm,attn,conv,GateDeltaNet_prefill}.xclbin $xdir/"
        fi
    else
        echo "[activate] flavour: ALL-HOST -- decode will be ~17-20 tok/s, NOT the NPU number."
        echo "[activate]          for NPU numbers:  source activate.sh npu"
    fi

    local mdir="$FLM_MODEL_PATH/models/$MODEL_NAME"
    [ -d "$mdir" ] || echo "[activate] WARNING: model folder not found: $mdir"
    [ -f "$mdir/model.q4nx" ] || echo "[activate] WARNING: no model.q4nx in $mdir"

    export FLM_MODEL_PATH
    echo "[activate] FLM_MODEL_PATH=$FLM_MODEL_PATH  (resolves $FLM_MODEL_PATH/models/$MODEL_NAME)"

    if [ "$_sourced" = "0" ]; then
        echo "[activate] NOTE: run with 'source activate.sh' -- executed directly, FLM_MODEL_PATH will not persist."
    else
        echo "[activate] ready:  make test            # add LENGTH=64 for a quicker run"
    fi
}

_main "${1:-}"
_activate_rc=$?
unset -f _main _fail
unset _here _libdir _engdir _built _sourced
return $_activate_rc 2>/dev/null || exit $_activate_rc
