#!/usr/bin/env bash
#
# home_install.sh — replicate `cmake --install` into a user-writable prefix
# (no sudo required). Produces the exact same layout under $FLM_PREFIX as a
# system install would under /opt/fastflowlm, and emits an env script that
# points the relocated binary at its data files and runtime libraries.
#
# Usage:
#   ./home_install.sh                # build (if needed) + install to ~/flm_exe
#   ./home_install.sh --no-build     # install an existing build/ tree only
#   ./home_install.sh --rai          # ryzenai-corelib backend, to scratch
#   FLM_PREFIX=/path ./home_install.sh
#
# --rai installs to /scratch/$USER/flm_exe_rai rather than ~/flm_exe: a rai
# build and a stock one would otherwise overwrite each other, and the corelib
# and DynamicDispatch libraries staged into a rai prefix are far too large for a
# home directory. Falls back to ~/flm_exe_rai with no scratch space, and
# FLM_PREFIX overrides either default.
#
# --rai selects the linux-rai-on preset. Point that build at a corelib prefix
# with RYZENAI_CORELIB_ROOT=/path (once -- CMake caches it), or drop the headers
# into include/ and libryzenai_corelib.so into lib/. An explicit PRESET= in the
# environment overrides --rai.
#
set -euo pipefail

# ---- configuration ---------------------------------------------------------
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SRC_DIR/.." && pwd)"
# The default prefix depends on --rai, which is not parsed yet, so the decision
# is deferred to just after the argument loop. Only the environment value is
# captured here so that an explicit FLM_PREFIX keeps overriding both defaults.
FLM_PREFIX_FROM_ENV="${FLM_PREFIX:-}"
# Every preset in CMakePresets.json puts its binaryDir at ${sourceDir}/build,
# and sourceDir is this directory -- not the repository root. Defaulting to the
# root instead left `cmake --build --preset` and this script configuring two
# separate trees of the same project, doubling the build and making --no-build
# miss a tree that had just been built. A PRESET with a different binaryDir
# (linux-rai, linux-snap) needs BUILD_DIR set to match it.
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build}"
# An explicit PRESET in the environment always wins; --rai only changes the
# default, which is why the environment value is remembered separately.
PRESET_FROM_ENV="${PRESET:-}"
PRESET="${PRESET_FROM_ENV:-linux-default}"
# Where XRT (the AMD NPU runtime) is installed. Override if non-standard.
XRT_DIR="${XRT_DIR:-/opt/xilinx/xrt}"

DO_BUILD=1
WANT_RAI=0
for arg in "$@"; do
    case "$arg" in
        --no-build) DO_BUILD=0 ;;
        --rai) WANT_RAI=1 ;;
        -h|--help)
            grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [[ "$WANT_RAI" -eq 1 && -z "$PRESET_FROM_ENV" ]]; then
    PRESET="linux-rai-on"
fi

# A rai install and a stock one differ in the binary and in the libraries staged
# beside it, but occupy identical paths, so sharing a prefix means whichever ran
# last silently replaces the other. Giving rai its own default keeps both usable
# and lets the two env scripts be sourced independently.
#
# That default lives on scratch rather than under $HOME: a rai install also
# carries libryzenai_corelib.so and the DynamicDispatch core it links against,
# which together dwarf the rest of the tree and do not belong in a home
# directory. $HOME is the fallback for a machine with no scratch space.
RAI_PREFIX_DEFAULT="/scratch/$USER/flm_exe_rai"
[[ -d "/scratch/$USER" ]] || RAI_PREFIX_DEFAULT="$HOME/flm_exe_rai"
if [[ -n "$FLM_PREFIX_FROM_ENV" ]]; then
    FLM_PREFIX="$FLM_PREFIX_FROM_ENV"
elif [[ "$WANT_RAI" -eq 1 ]]; then
    FLM_PREFIX="$RAI_PREFIX_DEFAULT"
else
    FLM_PREFIX="$HOME/flm_exe"
fi

echo "[home_install] repo:    $REPO_DIR"
echo "[home_install] prefix:  $FLM_PREFIX"
echo "[home_install] build:   $BUILD_DIR"
echo "[home_install] preset:  $PRESET"

# ---- build (optional) ------------------------------------------------------
# A configure that fails part way still writes CMakeCache.txt, but never gets
# as far as emitting build.ninja. Treating the cache alone as "already
# configured" therefore wedged the script permanently: every later run skipped
# the configure it needed and died inside the build step with
# "ninja: error: loading 'build.ninja'", which says nothing about the real
# fault. Key the decision on the generator file, which only exists once a
# configure has actually succeeded.
needs_configure() {
    [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]] && return 0
    # Ninja and Makefile cover the generators these presets select; anything
    # else is left alone rather than guessed at.
    [[ -f "$BUILD_DIR/build.ninja" || -f "$BUILD_DIR/Makefile" ]] || return 0
    # Reusing a cache configured the other way round would silently install a
    # non-rai build for --rai (or the reverse). Only checked when the preset was
    # not chosen by hand, since a custom preset's intent is not ours to infer.
    if [[ -z "$PRESET_FROM_ENV" ]]; then
        local cached=OFF
        grep -q '^FLM_ENABLE_RAI:BOOL=ON' "$BUILD_DIR/CMakeCache.txt" && cached=ON
        local wanted=OFF
        [[ "$WANT_RAI" -eq 1 ]] && wanted=ON
        [[ "$cached" == "$wanted" ]] || return 0
    fi
    return 1
}

if [[ "$DO_BUILD" -eq 1 ]]; then
    if needs_configure; then
        echo "[home_install] configuring (preset: $PRESET) ..."
        configure_args=()
        if [[ -n "${RYZENAI_CORELIB_ROOT:-}" ]]; then
            configure_args+=(-DRYZENAI_CORELIB_ROOT="$RYZENAI_CORELIB_ROOT")
        fi
        # --fresh discards whatever a previous failed or differently-configured
        # run left behind. Without it CMake reloads those entries and the
        # reconfigure inherits the state it is meant to replace.
        fresh_args=()
        [[ -f "$BUILD_DIR/CMakeCache.txt" ]] && fresh_args+=(--fresh)
        if ! cmake -S "$SRC_DIR" -B "$BUILD_DIR" --preset "$PRESET" \
                   ${fresh_args[@]+"${fresh_args[@]}"} \
                   ${configure_args[@]+"${configure_args[@]}"}; then
            echo >&2
            echo "[home_install] ERROR: configure failed." >&2
            if [[ "$WANT_RAI" -eq 1 || "$PRESET" == *rai* ]]; then
                echo "               A rai build needs the ryzenai-corelib headers. If the error" >&2
                echo "               above is about RYZENAI_CORELIB_INCLUDE_DIR, point the build at" >&2
                echo "               the corelib prefix:" >&2
                echo >&2
                echo "                 RYZENAI_CORELIB_ROOT=/path/to/ryzenai-corelib \\" >&2
                echo "                   $0 ${*}" >&2
                echo >&2
                echo "               CMake caches it, so this is only needed once per build dir." >&2
            fi
            exit 1
        fi
    fi
    echo "[home_install] building ..."
    cmake --build "$BUILD_DIR"
else
    echo "[home_install] --no-build: skipping configure/build"
fi

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "[home_install] ERROR: build dir '$BUILD_DIR' does not exist. Run without --no-build first." >&2
    exit 1
fi

# ---- install into the home prefix -----------------------------------------
# This runs the same install rules as a system install, just relocated.
# It populates:
#   $FLM_PREFIX/bin/flm
#   $FLM_PREFIX/lib/flm/*.so
#   $FLM_PREFIX/share/flm/model_list.json
#   $FLM_PREFIX/share/flm/xclbins/
echo "[home_install] installing to $FLM_PREFIX ..."
cmake --install "$BUILD_DIR" --prefix "$FLM_PREFIX"

# ---- stage the rai (ryzenai-corelib) runtime -------------------------------
# A corelib-backed build links libryzenai_corelib.so, which no install rule
# covers: CMake only globs lib/<runtime> for engine libraries, and corelib is
# not one of those. Copying it alone is still not enough -- it reaches
# libdyn_dispatch_core.so through an absolute RUNPATH into the out-of-tree
# DynamicDispatch prefix it was built against, so the install would break as
# soon as that directory moved.
#
# Copy corelib together with every non-system library it pulls in, leaving the
# prefix self-contained. The dynamic loader searches LD_LIBRARY_PATH (set by the
# env script below) before DT_RUNPATH, so these copies take precedence over the
# build-time paths baked into the binaries.
CACHE_FILE="$BUILD_DIR/CMakeCache.txt"
if [[ -f "$CACHE_FILE" ]] && grep -q '^FLM_ENABLE_RAI:BOOL=ON' "$CACHE_FILE"; then
    # HRX puts engine libs in lib/flm; the portable and XRT layouts use lib/.
    if grep -q '^FLM_USE_HRX:BOOL=ON' "$CACHE_FILE"; then
        RAI_LIB_DEST="$FLM_PREFIX/lib/flm"
    else
        RAI_LIB_DEST="$FLM_PREFIX/lib"
    fi

    CORELIB="$(sed -n 's/^RYZENAI_CORELIB_LIBRARY:FILEPATH=//p' "$CACHE_FILE")"
    if [[ -z "$CORELIB" || ! -f "$CORELIB" ]]; then
        echo "[home_install] ERROR: rai build, but RYZENAI_CORELIB_LIBRARY in" >&2
        echo "               $CACHE_FILE does not point at a readable file." >&2
        exit 1
    fi

    echo "[home_install] rai build detected; staging corelib into $RAI_LIB_DEST"
    mkdir -p "$RAI_LIB_DEST"
    install -m 0755 "$CORELIB" "$RAI_LIB_DEST/"
    # ldd resolves the whole transitive chain, so filtering out the system
    # directories here leaves exactly the libraries that ship with the build.
    DD_CORE_DIR=""
    while read -r _soname _arrow dep _addr; do
        case "$dep" in
            /lib/*|/lib64/*|/usr/lib/*|/usr/lib64/*|"") continue ;;
        esac
        [[ -f "$dep" ]] || continue
        if [[ "$(basename "$dep")" == libdyn_dispatch_core.so* ]]; then
            DD_CORE_DIR="$(dirname "$dep")"
        fi
        echo "[home_install]   + $(basename "$dep")"
        install -m 0755 "$dep" "$RAI_LIB_DEST/"
    done < <(ldd "$CORELIB" | grep '=>' || true)

    # libdyn_bins.so is invisible to the ldd walk above: nothing declares it as
    # a DT_NEEDED. DynamicDispatch dlopen()s it at runtime, from the directory
    # holding libdyn_dispatch_core.so (it dladdr()s itself to find that dir),
    # and when the file is absent Transaction::load_large_txn_ops_dll() just
    # returns -- no warning, no error.
    #
    # It only exists when DD was configured with DD_MDS_IN_BINS_DLL=ON, which
    # moves every transaction/mds/*.elf -- i.e. the whole AIE4 kernel set -- out
    # of the static transaction lib and into this library. Skipping it leaves
    # the AIE4 shape table empty, and the first matmul dies with
    #   Target Shape (K: 3072, N: 3072, Gs: 64) ... not supported in this
    #   supported shape list. Shape list size: 0
    # which reads like an unsupported model rather than a missing file.
    if [[ -n "$DD_CORE_DIR" && -f "$DD_CORE_DIR/libdyn_bins.so" ]]; then
        echo "[home_install]   + libdyn_bins.so (dlopen'd AIE4 kernel package)"
        install -m 0755 "$DD_CORE_DIR/libdyn_bins.so" "$RAI_LIB_DEST/"
    fi
fi

# ---- emit the environment script ------------------------------------------
ENV_SCRIPT="$FLM_PREFIX/flm_env.sh"
echo "[home_install] writing env script: $ENV_SCRIPT"
cat > "$ENV_SCRIPT" <<EOF
#!/usr/bin/env bash
# flm_env.sh — initialize the environment for the relocated FastFlowLM install.
# Source this before running flm:   source "$ENV_SCRIPT"
#
# The binary was compiled with CMAKE_INSTALL_PREFIX baked in (e.g. /opt/fastflowlm),
# so when relocated it relies on these env vars to locate its data and libraries.

FLM_PREFIX="$FLM_PREFIX"
XRT_DIR="$XRT_DIR"

# 1. Data files (consumed by utils::find_model_list / utils::find_xclbin_path).
export FLM_CONFIG_PATH="\$FLM_PREFIX/share/flm/model_list.json"
export FLM_XCLBIN_PATH="\$FLM_PREFIX/share/flm"

# 2. Runtime libraries. The flm binary already has RPATH \$ORIGIN/../lib/flm for
#    its bundled .so files, but XRT's libs (libxrt_coreutil.so, etc.) are found
#    via XRT's own setup or LD_LIBRARY_PATH.
if [[ -f "\$XRT_DIR/setup.sh" ]]; then
    # XRT's setup.sh exports XILINX_XRT, PATH and LD_LIBRARY_PATH for the NPU.
    source "\$XRT_DIR/setup.sh"
else
    echo "[flm_env] WARNING: \$XRT_DIR/setup.sh not found; falling back to LD_LIBRARY_PATH" >&2
    export LD_LIBRARY_PATH="\$XRT_DIR/lib:\${LD_LIBRARY_PATH:-}"
fi
# Belt-and-suspenders: also expose the bundled libs explicitly. Both engine
# directories are listed because the layout depends on the runtime backend
# (HRX uses lib/flm, XRT and portable builds use lib), and a rai build stages
# libryzenai_corelib.so plus its DynamicDispatch dependency into whichever one
# applies. This has to precede DT_RUNPATH, which still points at the machine
# the libraries were built on.
export LD_LIBRARY_PATH="\$FLM_PREFIX/lib:\$FLM_PREFIX/lib/flm:\${LD_LIBRARY_PATH:-}"

# 3. Put flm on PATH.
export PATH="\$FLM_PREFIX/bin:\$PATH"
export FLM_MODEL_PATH="/scratch/alfxu"

echo "[flm_env] FastFlowLM environment ready (prefix: \$FLM_PREFIX)"
EOF
chmod +x "$ENV_SCRIPT"

echo
echo "[home_install] Done."
echo "  Run:    source \"$ENV_SCRIPT\" && flm --help"
echo "  Or:     source \"$ENV_SCRIPT\" && flm run <model>"
