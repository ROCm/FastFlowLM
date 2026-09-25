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

# XRT does not reach its own plugins through DT_NEEDED. It dlopens libxrt_core,
# the libxrt_driver_xdna NPU driver and the rest at run time from a path it
# builds as $XILINX_XRT/lib/x86_64-linux-gnu/<lib>, and when XILINX_XRT is unset
# it guesses that root three directories above wherever libxrt_coreutil happened
# to be loaded from. A rai install bundles no XRT for that guess to land on, so
# it names the root instead, and accepts a candidate only if the directory XRT
# will actually dlopen from exists -- that way a wrong answer surfaces at
# install time instead of at the first NPU call.
#
# A stock install is left to the guess, exactly as it always was: its prefix
# carries its own XRT in the layout XRT expects, and the guess lands on it.
# Naming a root there, or putting a second XRT ahead of it on LD_LIBRARY_PATH,
# is what made a stock prefix report "No such library
# /home/$USER/lib/x86_64-linux-gnu/libxrt_core.so.2" and then "No such device
# with index '0'" on a machine whose NPU was fine.
detect_xrt_root() {
    local cand
    for cand in "${XILINX_XRT:-}" "$XRT_DIR" /usr/local /usr; do
        [[ -n "$cand" ]] || continue
        if [[ -f "$cand/lib/x86_64-linux-gnu/libxrt_core.so.2" ]]; then
            printf '%s\n' "$cand"
            return 0
        fi
    done
    return 1
}

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

# Empty unless this is a rai install; see detect_xrt_root above for why only
# that half wants an answer.
XRT_ROOT=""
[[ "$WANT_RAI" -eq 1 ]] && XRT_ROOT="$(detect_xrt_root || true)"

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

# An earlier split shipped a second catalog, model_list_rai.json, beside this
# one; a prefix that still holds it would answer "what can this install run"
# twice, with the stale copy winning nothing but confusion.
rm -f "$FLM_PREFIX/share/flm/model_list_rai.json"

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
#
# Keyed on --rai rather than on the cache: a stock install has to be the stock
# install main ships, and reading the flag out of whichever build tree happened
# to be lying around made a plain ./home_install.sh drop libryzenai_corelib.so
# and a 400 MB libdyn_dispatch_core.so into ~/flm_exe, where the bundled XRT
# then had company it could not cope with. A tree that disagrees with the mode
# gets a warning instead.
CACHE_FILE="$BUILD_DIR/CMakeCache.txt"
if [[ "$WANT_RAI" -eq 0 ]]; then
    if [[ -f "$CACHE_FILE" ]] && grep -q '^FLM_ENABLE_RAI:BOOL=ON' "$CACHE_FILE"; then
        echo "[home_install] WARNING: $BUILD_DIR was configured with FLM_ENABLE_RAI=ON," >&2
        echo "[home_install]          but this is a stock install: no corelib runtime is" >&2
        echo "[home_install]          staged and the env script is the stock one. Re-run" >&2
        echo "[home_install]          with --rai for a corelib install." >&2
    fi
elif [[ -f "$CACHE_FILE" ]] && grep -q '^FLM_ENABLE_RAI:BOOL=ON' "$CACHE_FILE"; then
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
    # moves every transaction/mds/*.elf -- i.e. the whole AIE_NEXT kernel set -- out
    # of the static transaction lib and into this library. Skipping it leaves
    # the AIE_NEXT shape table empty, and the first matmul dies with
    #   Target Shape (K: 3072, N: 3072, Gs: 64) ... not supported in this
    #   supported shape list. Shape list size: 0
    # which reads like an unsupported model rather than a missing file.
    if [[ -n "$DD_CORE_DIR" && -f "$DD_CORE_DIR/libdyn_bins.so" ]]; then
        echo "[home_install]   + libdyn_bins.so (dlopen'd AIE_NEXT kernel package)"
        install -m 0755 "$DD_CORE_DIR/libdyn_bins.so" "$RAI_LIB_DEST/"
    fi
else
    echo "[home_install] ERROR: --rai, but $BUILD_DIR was not configured with" >&2
    echo "               FLM_ENABLE_RAI=ON, so there is no corelib runtime to" >&2
    echo "               stage. Drop --no-build, or point BUILD_DIR at a rai tree." >&2
    exit 1
fi

# ---- emit the environment script ------------------------------------------
if [[ "$WANT_RAI" -eq 1 ]]; then
    if [[ -n "$XRT_ROOT" ]]; then
        echo "[home_install] XRT runtime root: $XRT_ROOT"
    else
        echo "[home_install] WARNING: found no XRT install providing" >&2
        echo "[home_install]          lib/x86_64-linux-gnu/libxrt_core.so.2; the NPU will be" >&2
        echo "[home_install]          unavailable unless \$XRT_DIR/setup.sh supplies it." >&2
    fi
fi

# Section 2 of the env script is the one place the two installs genuinely
# differ, so it is composed here instead of branched on inside the generated
# file. The stock text is main's, to the character: a stock prefix carries its
# own XRT in the layout XRT expects and reaches it through the binary's RUNPATH,
# and every extra hint that was added for rai -- the XILINX_XRT pin, $FLM_PREFIX
# /lib ahead of it on LD_LIBRARY_PATH -- only got in its way. A rai prefix has
# the opposite need: no XRT of its own, and corelib plus DynamicDispatch staged
# beside the binary.
if [[ "$WANT_RAI" -eq 1 ]]; then
    XRT_ENV_BLOCK="$(cat <<RAI_ENV
XRT_ROOT="$XRT_ROOT"

# 2. Runtime libraries. The flm binary already has RPATH \$ORIGIN/../lib/flm for
#    its bundled .so files, but XRT's libs (libxrt_coreutil.so, etc.) are found
#    via XRT's own setup or LD_LIBRARY_PATH.
#
#    XILINX_XRT has to be exported, not merely inherited. XRT dlopens libxrt_core
#    and the libxrt_driver_xdna NPU plugin from \$XILINX_XRT/lib/x86_64-linux-gnu,
#    and when the variable is unset it guesses that root three directories above
#    wherever libxrt_coreutil was loaded from. A rai build bundles no XRT for the
#    guess to land on, so pin the root home_install.sh resolved -- it accepted
#    that root only because the plugin directory exists -- and put the same
#    directory on LD_LIBRARY_PATH so the linked-in libxrt_coreutil comes from
#    that install too. Without the pin nothing reports a missing root: the NPU
#    just comes up with no driver, corelib says "no AIE_NEXT hw_context
#    (unordered_map::at)" and xrt::device(0) says "No such library
#    .../libxrt_core.so.2".
if [[ -f "\$XRT_DIR/setup.sh" ]]; then
    # A real XRT install knows its own layout; let it speak for itself.
    # setup.sh exports XILINX_XRT, PATH and LD_LIBRARY_PATH for the NPU.
    source "\$XRT_DIR/setup.sh"
elif [[ -n "\$XRT_ROOT" ]]; then
    export XILINX_XRT="\$XRT_ROOT"
    export LD_LIBRARY_PATH="\$XRT_ROOT/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}"
else
    echo "[flm_env] WARNING: no XRT runtime found at install time and" >&2
    echo "[flm_env]          \$XRT_DIR/setup.sh is missing; the NPU will be unavailable." >&2
fi
# Belt-and-suspenders: also expose the bundled libs explicitly. Both engine
# directories are listed because the layout depends on the runtime backend
# (HRX uses lib/flm, XRT and portable builds use lib), and a rai build stages
# libryzenai_corelib.so plus its DynamicDispatch dependency into whichever one
# applies. This has to precede DT_RUNPATH, which still points at the machine
# the libraries were built on.
export LD_LIBRARY_PATH="\$FLM_PREFIX/lib:\$FLM_PREFIX/lib/flm:\${LD_LIBRARY_PATH:-}"
RAI_ENV
)"
else
    XRT_ENV_BLOCK="$(cat <<'STOCK_ENV'
# 2. Runtime libraries. The flm binary already has RPATH $ORIGIN/../lib/flm for
#    its bundled .so files, but XRT's libs (libxrt_coreutil.so, etc.) are found
#    via XRT's own setup or LD_LIBRARY_PATH.
if [[ -f "$XRT_DIR/setup.sh" ]]; then
    # XRT's setup.sh exports XILINX_XRT, PATH and LD_LIBRARY_PATH for the NPU.
    source "$XRT_DIR/setup.sh"
else
    echo "[flm_env] WARNING: $XRT_DIR/setup.sh not found; falling back to LD_LIBRARY_PATH" >&2
    export LD_LIBRARY_PATH="$XRT_DIR/lib:${LD_LIBRARY_PATH:-}"
fi
# Belt-and-suspenders: also expose the bundled libs explicitly.
export LD_LIBRARY_PATH="$FLM_PREFIX/lib/flm:${LD_LIBRARY_PATH:-}"
STOCK_ENV
)"
fi

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

$XRT_ENV_BLOCK

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
