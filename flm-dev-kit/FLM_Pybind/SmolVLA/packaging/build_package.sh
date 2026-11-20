#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#
# Build the shippable smolvla-npu archive. Run this from the source tree; the
# archive it produces is what you send.
#
# It copies the freshly built extension and the NPU bitstreams in, builds a
# wheel, and tars the wheel together with setup.sh / requirements-runtime.txt /
# README. No C++ source, headers or Makefile go into the archive -- only the
# compiled .so, the xclbins it loads, and the python harness.
#
# Usage:
#   ./build_package.sh                     # build .so if needed, then package
#   ./build_package.sh --skip-build        # package whatever .so is already there
#   ./build_package.sh --offline-wheels    # also vendor every dependency wheel
#   ./build_package.sh --refresh-harness   # re-copy verify_smolvla.py from the
#                                          # source tree, overwriting the
#                                          # packaged copy and its docstring
#
# The result is dist/smolvla-npu-<version>.tar.gz.
#
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$PKG_DIR/.." && pwd)"

SKIP_BUILD=0
OFFLINE_WHEELS=0
REFRESH_HARNESS=0
BUILD_PY="$SRC_DIR/.venv/bin/python"
EXT_NAME="smolvla_npu.cpython-312-x86_64-linux-gnu.so"
XCLBINS=(vision.xclbin text.xclbin expert.xclbin)
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

die()  { printf '\nerror: %s\n' "$*" >&2; exit 1; }
info() { printf '\n==> %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)      SKIP_BUILD=1; shift ;;
        --offline-wheels)  OFFLINE_WHEELS=1; shift ;;
        --refresh-harness) REFRESH_HARNESS=1; shift ;;
        -h|--help)         sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -x "$BUILD_PY" ]] || die "no build venv at $BUILD_PY (run the source-tree setup first)"

# --------------------------------------------------------------------------
# 1. the extension
# --------------------------------------------------------------------------
# PROFILE=0: with profiling on, every forward() prints a per-stage timing table,
# which buries the harness output. The shipped module should be quiet.
# DIST=1:    hidden visibility + NDEBUG + strip, so the binary does not carry
#            internal symbol names or the source paths that assert() embeds.
# Both differ from the local defaults, and objects are keyed by source path
# only, so a stale build/ from a normal `make python` would be silently reused.
# Clean first.
if [[ $SKIP_BUILD -eq 0 ]]; then
    info "building the extension (PROFILE=0 DIST=1)"
    make -C "$SRC_DIR" clean >/dev/null 2>&1 || true
    make -C "$SRC_DIR" python PROFILE=0 DIST=1 PYTHON="$BUILD_PY" >/dev/null \
        || die "extension build failed"
fi

[[ -f "$SRC_DIR/$EXT_NAME" ]] || die "extension not found: $SRC_DIR/$EXT_NAME"
info "extension: $(du -h "$SRC_DIR/$EXT_NAME" | cut -f1)"
cp "$SRC_DIR/$EXT_NAME" "$PKG_DIR/smolvla_npu_pkg/$EXT_NAME"

# --------------------------------------------------------------------------
# 1b. the bitstreams
# --------------------------------------------------------------------------
# They are tracked in the source tree (see ../.gitignore) because they belong
# to the kernel build, not to any model directory. The extension locates them
# beside itself at run time, so they have to land inside the package dir.
info "bitstreams"
mkdir -p "$PKG_DIR/smolvla_npu_pkg/XCLBINS"
for x in "${XCLBINS[@]}"; do
    [[ -f "$SRC_DIR/XCLBINS/$x" ]] || die "bitstream not found: $SRC_DIR/XCLBINS/$x"
    cp "$SRC_DIR/XCLBINS/$x" "$PKG_DIR/smolvla_npu_pkg/XCLBINS/$x"
    printf '    %s (%s)\n' "$x" "$(du -h "$SRC_DIR/XCLBINS/$x" | cut -f1)"
done

# The harness is developed in the source tree and copied in. Its docstring
# differs on purpose -- the packaged one documents the installed entry points
# (source activate.sh / smolvla-verify) rather than `python verify_smolvla.py`
# -- so an unconditional copy would clobber that text on every build. Copy only
# when the packaged file is absent or --refresh-harness was passed, and compare
# the code below either way: a silently stale copy ships a harness missing
# whatever was just added to the source tree, and the failure only shows up on
# the recipient's machine as an unrecognized argument.
info "harness"
HARNESS_SRC="$SRC_DIR/verify_smolvla.py"
HARNESS_PKG="$PKG_DIR/smolvla_npu_pkg/verify_smolvla.py"
[[ -f "$HARNESS_SRC" ]] || die "harness not found: $HARNESS_SRC"

if [[ ! -f "$HARNESS_PKG" || $REFRESH_HARNESS -eq 1 ]]; then
    cp "$HARNESS_SRC" "$HARNESS_PKG"
    echo "    copied verify_smolvla.py from the source tree."
    echo "    NOTE: this overwrote the packaged docstring with the source-tree"
    echo "          one, which documents 'python verify_smolvla.py' rather than"
    echo "          the installed 'smolvla-verify'. Re-edit it before shipping."
else
    # Everything except the module docstring must match, or the packaged copy is
    # behind the source tree.
    if ! "$BUILD_PY" - "$HARNESS_SRC" "$HARNESS_PKG" <<'PY'
import ast, sys

def body(path):
    src = open(path).read()
    tree = ast.parse(src)
    if ast.get_docstring(tree) is not None:
        tree.body = tree.body[1:]          # drop the deliberate difference
    return ast.dump(tree)

sys.exit(0 if body(sys.argv[1]) == body(sys.argv[2]) else 1)
PY
    then
        die "packaged verify_smolvla.py is out of date with $HARNESS_SRC
       (they differ beyond the module docstring). Re-run with --refresh-harness
       to overwrite the packaged copy, then restore its docstring."
    fi
    echo "    packaged verify_smolvla.py matches the source tree."
fi

# --------------------------------------------------------------------------
# 2. the wheel
# --------------------------------------------------------------------------
info "building the wheel"
rm -rf "$PKG_DIR/dist" "$PKG_DIR/build" "$PKG_DIR"/*.egg-info
"$BUILD_PY" -m pip install --quiet --upgrade build >/dev/null 2>&1 || true
( cd "$PKG_DIR" && "$BUILD_PY" -m build --wheel --no-isolation ) >/dev/null \
    || die "wheel build failed (need: pip install build setuptools wheel)"

WHEEL="$(ls -1t "$PKG_DIR"/dist/*.whl | head -1)"
info "wheel: $(basename "$WHEEL") ($(du -h "$WHEEL" | cut -f1))"

# Guard against shipping an archive whose wheel is missing the extension or a
# bitstream -- the wheel builds fine without either. A missing extension fails
# at import; a missing bitstream gets all the way to NPU bring-up before it
# does, which is a far worse place to find out.
"$BUILD_PY" - "$WHEEL" "$EXT_NAME" "${XCLBINS[@]}" <<'PY' || die "wheel is incomplete (see above)"
import sys, zipfile
wheel, required = sys.argv[1], sys.argv[2:]
names = zipfile.ZipFile(wheel).namelist()
missing = [r for r in required if not any(n.endswith(r) for n in names)]
if missing:
    print("wheel is missing: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PY

# --------------------------------------------------------------------------
# 3. optional: vendored dependency wheels for an air-gapped target
# --------------------------------------------------------------------------
if [[ $OFFLINE_WHEELS -eq 1 ]]; then
    info "downloading dependency wheels (several GB, slow)"
    rm -rf "$PKG_DIR/wheels"
    "$BUILD_PY" -m pip download \
        --extra-index-url "$TORCH_CPU_INDEX" \
        -r "$PKG_DIR/requirements-runtime.txt" \
        -d "$PKG_DIR/wheels" >/dev/null || die "wheel download failed"
    info "vendored $(ls -1 "$PKG_DIR/wheels" | wc -l) wheels ($(du -sh "$PKG_DIR/wheels" | cut -f1))"
fi

# --------------------------------------------------------------------------
# 4. the archive
# --------------------------------------------------------------------------
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$PKG_DIR/pyproject.toml" | head -1)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
OUT="smolvla-npu-$VERSION"
mkdir -p "$STAGE/$OUT/dist"

cp "$WHEEL" "$STAGE/$OUT/dist/"
cp "$PKG_DIR/setup.sh" "$PKG_DIR/requirements-runtime.txt" "$STAGE/$OUT/"
cp "$PKG_DIR/README.md" "$STAGE/$OUT/" 2>/dev/null || true
chmod +x "$STAGE/$OUT/setup.sh"
[[ $OFFLINE_WHEELS -eq 1 ]] && cp -r "$PKG_DIR/wheels" "$STAGE/$OUT/"

mkdir -p "$PKG_DIR/dist"
ARCHIVE="$PKG_DIR/dist/$OUT.tar.gz"
tar -czf "$ARCHIVE" -C "$STAGE" "$OUT"

# --------------------------------------------------------------------------
# 5. leak check
# --------------------------------------------------------------------------
# The whole point of shipping a wheel is that the C++ stays here. Fail loudly
# rather than discover a stray header after the archive has been sent.
#
# Bitstreams used to be listed here too, back when they travelled with the
# model directory and had no business in the archive. They are payload now --
# the extension will not bring up the NPU without them -- so only source counts
# as a leak.
info "checking the archive for source files"
LEAKS="$(tar -tzf "$ARCHIVE" | grep -E '\.(cpp|hpp|h|cc|cxx)$|Makefile' || true)"
if [[ -n "$LEAKS" ]]; then
    echo "$LEAKS" >&2
    die "archive contains source files (listed above)"
fi
echo "    no source files."

# A DIST=1 build exports only the module init; anything else means the build
# silently fell back to the normal flags (a stale build/ tree is the usual
# cause) and the binary is carrying internal names and assert() source paths.
SHIPPED="$PKG_DIR/smolvla_npu_pkg/$EXT_NAME"
INTERNAL="$(nm -D --defined-only "$SHIPPED" 2>/dev/null | grep -c 'SmolVLA\|cpu_func\|npu_app' || true)"
SRCPATHS="$(strings -a "$SHIPPED" | grep -E '(^|/)src/.*\.(cpp|hpp)' | sort -u || true)"
if [[ "$INTERNAL" -gt 0 || -n "$SRCPATHS" ]]; then
    [[ -n "$SRCPATHS" ]] && echo "$SRCPATHS" >&2
    die "the extension still exports $INTERNAL internal symbol(s) and embeds the
       source paths above. It was not built with DIST=1 -- run without
       --skip-build, or: make -C $SRC_DIR clean && make -C $SRC_DIR python PROFILE=0 DIST=1"
fi
echo "    no internal symbols or source paths in the extension."

info "done: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
cat <<EOF

Send that one file. The recipient runs:

    tar xzf $OUT.tar.gz && cd $OUT
    ./setup.sh
    source activate.sh
    smolvla-verify --model_path /path/to/SmolVLA --dataset lerobot/libero --steps 4

The model directory is theirs to supply at run time; the install does not need
it. Only config.json and model.safetensors -- the bitstreams are in the wheel.
EOF
