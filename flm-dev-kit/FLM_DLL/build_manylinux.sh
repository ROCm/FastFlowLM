#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Build all FLM engine .so inside the TheRock manylinux_2_28 container against the
# pinned public HRX package. Produces glibc<=2.28 Boost-free engine libraries.
#
# usage: bash flm-dev-kit/FLM_DLL/build_manylinux.sh     (on a host with Docker; CI runs exactly this)
#
# On the host it fetches the pinned HRX package if missing, then re-runs itself in the pinned
# container as the calling user, with the repo mounted at /work, so build/ is not left owned by root.
# Inside (FLM_MANYLINUX_INNER=1) it builds. To run it in a container you started yourself, set
# FLM_MANYLINUX_INNER=1 there.
#
# Environment:
#   MANYLINUX_IMAGE  container image (default: the pin below)
#   SKIP_ABI_CHECK=1 skip the closing check_abi.sh run (the CI action runs it as its own step)
set -uo pipefail

# The same image as the SDK's manylinux_2_28 FLM job, pinned by digest so CI and that link agree.
MANYLINUX_IMAGE="${MANYLINUX_IMAGE:-ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:702a5133851e6d1daf1207d2c9fbb01c2667914a5b6dc5a01faeb3ce66ea6421}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# flm-dev-kit/FLM_DLL -> FastFlowLM repo root, which owns hrx-integration/
root="$(cd "$here/../.." && pwd)"
[[ -f $root/hrx-integration/hrx-release.env ]] || { echo "FATAL: $root/hrx-integration/hrx-release.env not found" >&2; exit 2; }

# Derive the extracted-package dir from the pinned release (hrx-release.env) so
# this tracks the pin instead of hardcoding a version.
# shellcheck disable=SC1091
source "$root/hrx-integration/hrx-release.env"
release_dir="hrx-integration/.hrx-release/${HRX_RELEASE_ASSET%.tar.zst}"

if [[ ${FLM_MANYLINUX_INNER:-0} != 1 ]]; then
  command -v docker >/dev/null 2>&1 || {
    echo "FATAL: docker not found. Run this on a host with Docker, or set FLM_MANYLINUX_INNER=1 inside the manylinux_2_28 container." >&2
    exit 2
  }
  if [[ ! -f $root/$release_dir/lib/libhrx.so ]]; then
    bash "$root/hrx-integration/fetch-hrx-release.sh" || exit
  fi
  echo "== image: $MANYLINUX_IMAGE"
  exec docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e FLM_MANYLINUX_INNER=1 \
    -e SKIP_ABI_CHECK="${SKIP_ABI_CHECK:-0}" \
    -v "$root:/work" \
    -w /work \
    "$MANYLINUX_IMAGE" \
    bash /work/flm-dev-kit/FLM_DLL/build_manylinux.sh
fi

# gcc-toolset-13 ships as g++/gcc; the engine Makefiles hardcode g++-13/gcc-13. Alias them in a
# private dir on PATH rather than /usr/local/bin, which a non-root user cannot write.
shim="$(mktemp -d)"
trap 'rm -rf "$shim"' EXIT
ln -s "$(command -v g++)" "$shim/g++-13"
ln -s "$(command -v gcc)" "$shim/gcc-13"
export PATH="$shim:$PATH"
hash -r

export HRX_RELEASE_DIR="$root/$release_dir"
echo "== toolchain =="; g++-13 --version | head -1; echo "== HRX_RELEASE_DIR: $HRX_RELEASE_DIR"
test -f "$HRX_RELEASE_DIR/lib/libhrx.so" || { echo "FATAL: libhrx.so missing"; exit 2; }

# The engine Makefiles select the HRX path only when this is exactly 1.
# Without it, build.sh produces the XRT path (which cannot link in this
# container — no /opt/xilinx/xrt).
export FLM_USE_HRX=1
export HRX_HOME="${HRX_HOME:-$HRX_RELEASE_DIR}"

cd "$here"
bash build.sh
rc=$?
echo "== build.sh exit: $rc =="
echo "== produced libs =="; ls -1 build/lib/*.so 2>/dev/null | wc -l; ls -1 build/lib/*.so 2>/dev/null

# The header promises glibc<=2.28; enforce it so a toolchain that is not the manylinux_2_28 one
# (or an engine that pulls in a newer symbol version) fails here instead of in the SDK's FLM link.
if [[ $rc -eq 0 && ${SKIP_ABI_CHECK:-0} != 1 ]]; then
  echo "== ABI ceiling =="; bash check_abi.sh build/lib || rc=$?
fi
exit $rc
