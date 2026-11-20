#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#
# Self-contained installer for the smolvla-npu package.
#
# Creates a python 3.12 venv, installs the prebuilt extension wheel and its
# CPU-only dependencies, and runs the environment check. No compiler, no build
# step, no source -- the NPU extension ships prebuilt.
#
# The model directory (config.json, model.safetensors) is NOT part of
# this package: it is several GB and is supplied separately. It is also not
# needed to install -- nothing here reads it. Point at your copy when you run:
#   smolvla-verify --model_path /path/to/SmolVLA
#
# Usage:
#   ./setup.sh                                  # install into ./.venv
#   ./setup.sh --venv /opt/smolvla-env          # install elsewhere
#   ./setup.sh --python /usr/bin/python3.12     # pick the interpreter
#   ./setup.sh --check-only                     # diagnose an existing install
#
# Then:
#   source activate.sh
#   smolvla-verify --model_path /path/to/SmolVLA --dataset lerobot/libero --steps 4
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN=""
CHECK_ONLY=0
OFFLINE=0
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

die()  { printf '\nerror: %s\n' "$*" >&2; exit 1; }
info() { printf '\n==> %s\n' "$*"; }

# Print the header comment as the help text: one copy of the usage, so the two
# cannot drift. Stops at the first non-comment line.
usage() { sed -n '3,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d; s/^#\{1,\} \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)        VENV_DIR="$2"; shift 2 ;;
        --python)      PYTHON_BIN="$2"; shift 2 ;;
        --check-only)  CHECK_ONLY=1; shift ;;
        # For an air-gapped host: pass a directory of pre-downloaded wheels
        # (pip download -r requirements-runtime.txt -d wheels/) and pip will
        # resolve entirely from it.
        --offline)     OFFLINE=1; shift ;;
        -h|--help)     usage ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
done

# --------------------------------------------------------------------------
# interpreter
# --------------------------------------------------------------------------
# The wheel carries a prebuilt cp312 extension, so the interpreter is not a
# preference -- 3.13 would install the wheel happily and then fail to import it.
# Check before creating anything.
find_python() {
    if [[ -n "$PYTHON_BIN" ]]; then
        command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "no such interpreter: $PYTHON_BIN"
        echo "$PYTHON_BIN"; return
    fi
    for cand in python3.12 python3 python; do
        if command -v "$cand" >/dev/null 2>&1 && \
           "$cand" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
            echo "$cand"; return
        fi
    done
    die "python 3.12 not found. The bundled NPU extension is built for CPython 3.12
       and will not import under any other version.
         Ubuntu: sudo apt install python3.12 python3.12-venv
       Then re-run, or pass --python /path/to/python3.12"
}

if [[ $CHECK_ONLY -eq 0 ]]; then
    PY="$(find_python)"
    PYVER="$("$PY" -c 'import platform; print(platform.python_version())')"
    [[ "$PYVER" == 3.12.* ]] || die "found python $PYVER, need 3.12.x (see --python)"
    info "using $(command -v "$PY") ($PYVER)"
fi

# --------------------------------------------------------------------------
# wheel
# --------------------------------------------------------------------------
find_wheel() {
    local w
    # Newest first, so a re-issued build wins over the one beside it.
    w="$(ls -1t "$SCRIPT_DIR"/dist/smolvla_npu-*.whl "$SCRIPT_DIR"/smolvla_npu-*.whl 2>/dev/null | head -1 || true)"
    [[ -n "$w" ]] || die "no smolvla_npu wheel found next to this script or in dist/.
       The package archive should contain one; re-extract it."
    echo "$w"
}

# --------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------
if [[ $CHECK_ONLY -eq 0 ]]; then
    WHEEL="$(find_wheel)"
    info "wheel: $(basename "$WHEEL")"

    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        info "creating venv at $VENV_DIR"
        "$PY" -m venv "$VENV_DIR" 2>/dev/null || die "venv creation failed.
       On Debian/Ubuntu the venv module ships separately:
         sudo apt install python3.12-venv"
    else
        info "reusing existing venv at $VENV_DIR"
    fi

    VPIP="$VENV_DIR/bin/pip"
    "$VPIP" install --quiet --upgrade pip setuptools wheel

    PIP_ARGS=()
    if [[ $OFFLINE -eq 1 ]]; then
        [[ -d "$SCRIPT_DIR/wheels" ]] || die "--offline needs a wheels/ directory next to this script"
        PIP_ARGS+=(--no-index --find-links "$SCRIPT_DIR/wheels")
        info "offline install from $SCRIPT_DIR/wheels"
    else
        # torch/torchvision +cpu live only on the PyTorch index. Without this,
        # pip resolves the default CUDA build and pulls in ~3 GB of unused
        # nvidia-* wheels.
        PIP_ARGS+=(--extra-index-url "$TORCH_CPU_INDEX")
    fi

    info "installing dependencies (CPU-only torch; this takes a few minutes)"
    "$VPIP" install "${PIP_ARGS[@]}" -r "$SCRIPT_DIR/requirements-runtime.txt" \
        || die "dependency install failed. If this is a network/index problem, see
       --offline in --help for installing from a local wheel directory."

    info "installing smolvla-npu"
    # --no-deps: the wheel declares none by design (see pyproject.toml); the
    # runtime set above is the single source of truth for versions.
    "$VPIP" install --force-reinstall --no-deps "$WHEEL"
fi

[[ -x "$VENV_DIR/bin/python" ]] || die "no venv at $VENV_DIR -- run without --check-only first"

# --------------------------------------------------------------------------
# activate.sh
# --------------------------------------------------------------------------
# Written rather than shipped, so it carries this install's absolute paths.
if [[ $CHECK_ONLY -eq 0 ]]; then
    ACTIVATE="$SCRIPT_DIR/activate.sh"
    {
        echo "# Generated by setup.sh -- source this file, do not execute it."
        echo "#   source activate.sh"
        echo "SMOLVLA_ENV_DIR=\"\$(cd \"\$(dirname \"\${BASH_SOURCE[0]}\")\" && pwd)\""
        echo "source \"$VENV_DIR/bin/activate\""
        echo
        echo "# XRT is linked directly by the extension. Adding its lib dir is harmless"
        echo "# when XRT was installed system-wide and necessary when it was not."
        echo "if [ -d /opt/xilinx/xrt/lib ]; then"
        echo "    export LD_LIBRARY_PATH=\"/opt/xilinx/xrt/lib:\${LD_LIBRARY_PATH:-}\""
        echo "fi"
        echo
        echo "# The model directory is a run-time argument: pass --model_path to"
        echo "# smolvla-verify. Uncomment to default it for every run in this shell."
        echo "#   export SMOLVLA_MODEL_PATH=/path/to/SmolVLA"
        echo
        echo "# HF_HOME keeps dataset/checkpoint downloads inside this install rather"
        echo "# than the home directory, which is often the smaller filesystem."
        echo "export HF_HOME=\"\${HF_HOME:-\$SMOLVLA_ENV_DIR/.hf_cache}\""
    } > "$ACTIVATE"
    chmod +x "$ACTIVATE"
    info "wrote $ACTIVATE"
fi

# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
# The model directory is deliberately not checked here: it is a run-time
# argument, not an install-time one, so an install on a machine that does not
# have it yet is still a complete install. `smolvla-doctor --model-path DIR`
# validates a copy once you have one.
info "checking the environment"
set +e
"$VENV_DIR/bin/smolvla-doctor"
DOCTOR_RC=$?
set -e

echo
if [[ $DOCTOR_RC -ne 0 ]]; then
    echo "Install completed, but the checks above found blocking problems."
    echo "Fix those, then re-check with:  ./setup.sh --check-only"
    exit $DOCTOR_RC
fi

cat <<EOF
Done.

  source $SCRIPT_DIR/activate.sh
  smolvla-verify --model_path /path/to/SmolVLA --dataset lerobot/libero --steps 4

The model directory is not part of this package -- point --model_path at your
copy. To check one: smolvla-doctor --model-path /path/to/SmolVLA
smolvla-verify --help lists the datasets this has been run against.
EOF
