# syntax=docker/dockerfile:1
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
#
# aie-build: the dev kit's kernel (xclbin) build environment. In it,
# "make bitstream [MODEL_TYPE=...]" in any flm-dev-kit/FLM_Xclbin build unit
# works with no setup.sh: the environment below is what setup.sh sets up with
# MLIR_AIE_DIR=/opt/mlir-aie and PEANO_INSTALL_DIR=/opt/peano/llvm-aie.
#
#   /opt/mlir-aie/install   mlir-aie built from source (aiecc.py, aie-opt, the
#                           Python bindings)
#   /opt/mlir-aie/ironenv   its venv (Python 3.12: the IRON Python, lit, cmake)
#   /opt/mlir-aie/utils     its scripts, so setup.sh can still be sourced
#   /opt/peano/llvm-aie     Peano (llvm-aie), the AIE core compiler
#   /usr/bin/xclbinutil     XRT's, which aiecc.py runs to package the xclbin
#
# Every version that reaches the kernels comes from
# flm-dev-kit/FLM_Xclbin/toolchain.lock, passed as build args (each ARG below without a default is a key there; the build
# stops if one is missing). .github/workflows/devkit-aie-image.yml builds and
# publishes it as ghcr.io/rocm/fastflowlm/aie-build. Locally, from the repo
# root:
#
#   docker build -f docker/aie-build.Dockerfile -t aie-build \
#     $(grep -E '^[A-Z0-9_]+=' flm-dev-kit/FLM_Xclbin/toolchain.lock | sed 's/^/--build-arg /') docker
#
# The build context is docker/: nothing from the repository goes into the
# image. Building mlir-aie takes about 10 minutes on 32 cores and about an hour
# on a 4-core runner; the stages keep it cached apart from Peano and XRT, so
# bumping those does not rebuild it. No Chess (Vitis AIE Essentials is
# licensed): ENABLE_CHESSCC=1 builds stop in common.mk's AIETOOLS_CHECK.
#
# apt: the packages that build mlir-aie (g++-13, lld) or run the IRON Python
# (python3.12) and xclbinutil (libxrt-utils, libxrt2) are pinned to exact
# versions from the lock; the rest (build-essential, git, curl, ninja, make,
# zstd and the other dependencies) resolve to the current Ubuntu 24.04 archive
# when the image builds. Ubuntu keeps only the newest version of a package in
# noble-updates, so after an update the build stops at the pin until the lock
# is bumped. The image's xclbinutil can package what aiecc.py makes, but there
# is no libxrt-dev and no XILINX_XRT: "make bitstream", "make kernel" and
# "make link" work, the host programs (run.exe, "make host", plain "make")
# do not.

ARG UBUNTU_IMAGE

# ---------------------------------------------------------------------------
# Peano: the wheel unpacked to /opt/peano, as pip --no-deps --target does
# (README.md step 2), from its URL and checked against its sha256.
FROM ${UBUNTU_IMAGE} AS peano

ARG LLVM_AIE_VERSION
ARG LLVM_AIE_WHEEL_URL
ARG LLVM_AIE_WHEEL_SHA256

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN for v in LLVM_AIE_VERSION LLVM_AIE_WHEEL_URL LLVM_AIE_WHEEL_SHA256; do \
        [[ -n "${!v:-}" ]] || { echo "build arg $v is not set: pass flm-dev-kit/FLM_Xclbin/toolchain.lock" >&2; exit 1; }; \
    done \
    && [[ "$LLVM_AIE_WHEEL_URL" == *"llvm_aie-${LLVM_AIE_VERSION}-"* ]] \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/* \
    && whl="/tmp/${LLVM_AIE_WHEEL_URL##*/}" \
    && curl -fsSL --retry 5 --retry-all-errors -o "$whl" "$LLVM_AIE_WHEEL_URL" \
    && echo "${LLVM_AIE_WHEEL_SHA256}  ${whl}" | sha256sum -c - \
    && unzip -q "$whl" -d /opt/peano \
    && rm "$whl" \
    && test -x /opt/peano/llvm-aie/bin/clang++

# ---------------------------------------------------------------------------
# mlir-aie, built from source exactly as flm-dev-kit/FLM_Xclbin/README.md
# step 1 does, except that the MLIR wheel and the mlir_aie wheel are fetched by
# URL and checked against their sha256, and the build is given the pinned
# Peano's final path rather than installing an older Peano into the venv.
FROM ${UBUNTU_IMAGE} AS mlir-aie

ARG MLIR_AIE_REPO
ARG MLIR_AIE_REF
ARG MLIR_AIE_SHA
ARG MLIR_WHEEL_VERSION
ARG MLIR_WHEEL_URL
ARG MLIR_WHEEL_SHA256
ARG MLIR_AIE_WHEEL_VERSION
ARG MLIR_AIE_WHEEL_URL
ARG MLIR_AIE_WHEEL_SHA256
ARG GCC13_VERSION
ARG LLD_VERSION
ARG LLD18_VERSION
ARG PYTHON312_VERSION
ARG PIP_VERSION
ARG ZLIB1G_DEV_VERSION

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN for v in MLIR_AIE_REPO MLIR_AIE_REF MLIR_AIE_SHA MLIR_WHEEL_VERSION MLIR_WHEEL_URL \
             MLIR_WHEEL_SHA256 MLIR_AIE_WHEEL_VERSION MLIR_AIE_WHEEL_URL MLIR_AIE_WHEEL_SHA256 \
             GCC13_VERSION LLD_VERSION LLD18_VERSION PYTHON312_VERSION PIP_VERSION ZLIB1G_DEV_VERSION; do \
        [[ -n "${!v:-}" ]] || { echo "build arg $v is not set: pass flm-dev-kit/FLM_Xclbin/toolchain.lock" >&2; exit 1; }; \
    done

# The host compiler and linker of the mlir-aie build (its script links with
# lld whenever "lld" is on PATH) and ironenv's Python, at the lock's versions.
# zlib1g-dev: the MLIR wheel's LLVM exports link ZLIB::ZLIB.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git ninja-build unzip \
        "g++-13=${GCC13_VERSION}" "gcc-13=${GCC13_VERSION}" \
        "lld=${LLD_VERSION}" "lld-18=${LLD18_VERSION}" \
        "python3.12=${PYTHON312_VERSION}" "python3.12-dev=${PYTHON312_VERSION}" \
        "python3.12-venv=${PYTHON312_VERSION}" \
        "zlib1g-dev=${ZLIB1G_DEV_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && [[ "$(c++ -dumpfullversion)" == 13.* ]] \
    && command -v lld

WORKDIR /opt/mlir-aie

# The fork at the pinned commit, with its submodules. The commit must still be
# on MLIR_AIE_REF: if it is not, the branch was rewritten (decision: its owner
# does not), which is worth a look before trusting the pin.
RUN git init -q . \
    && git remote add origin "$MLIR_AIE_REPO" \
    && git fetch -q --depth 1 origin "$MLIR_AIE_SHA" \
    && git checkout -q --detach FETCH_HEAD \
    && [[ "$(git rev-parse HEAD)" == "$MLIR_AIE_SHA" ]] \
    && git fetch -q --filter=blob:none origin "refs/heads/${MLIR_AIE_REF}:refs/remotes/origin/${MLIR_AIE_REF}" \
    && git merge-base --is-ancestor "$MLIR_AIE_SHA" "origin/${MLIR_AIE_REF}" \
    && git submodule update -q --init --recursive --depth 1 \
    && got="$(utils/clone-llvm.sh --get-wheel-version)" \
    && { [[ "$got" == "$MLIR_WHEEL_VERSION" ]] \
         || { echo "MLIR_WHEEL_VERSION is $MLIR_WHEEL_VERSION but utils/clone-llvm.sh at $MLIR_AIE_SHA wants $got" >&2; exit 1; }; }

# ironenv: the fork's pinned Python requirements, mlir-python-extras (pinned
# by commit in requirements_extras.txt) and the prebuilt mlir_aie wheel.
RUN mkdir -p /tmp/wheels \
    && whl="/tmp/wheels/${MLIR_AIE_WHEEL_URL##*/}" \
    && curl -fsSL --retry 5 --retry-all-errors -o "$whl" "$MLIR_AIE_WHEEL_URL" \
    && echo "${MLIR_AIE_WHEEL_SHA256}  ${whl}" | sha256sum -c - \
    && python3.12 -m venv ironenv \
    && source ironenv/bin/activate \
    && python3 -m pip install -q "pip==${PIP_VERSION}" \
    && python3 -m pip install -q -r python/requirements.txt \
    && HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -q -r python/requirements_extras.txt \
    && python3 -m pip install -q "$whl" \
    && [[ "$(python3 -m pip show mlir_aie | sed -n 's/^Version: //p')" == "$MLIR_AIE_WHEEL_VERSION" ]] \
    && rm -rf /tmp/wheels

# The source build, against the MLIR wheel unpacked where the script itself
# would put it, with the pinned Peano in place (its path is recorded as aiecc's
# default when PEANO_INSTALL_DIR is unset). On failure the CMake and ninja logs
# are printed: the script tees them into build/. The script can
# report success after a failure, so the result is checked, and everything
# but install/ is removed in this layer.
COPY --from=peano /opt/peano /opt/peano
RUN mkdir -p my_install \
    && whl="my_install/${MLIR_WHEEL_URL##*/}" \
    && curl -fsSL --retry 5 --retry-all-errors -o "$whl" "$MLIR_WHEEL_URL" \
    && echo "${MLIR_WHEEL_SHA256}  ${whl}" | sha256sum -c - \
    && unzip -q "$whl" -d my_install \
    && rm "$whl" \
    && source ironenv/bin/activate \
    && { bash ./utils/build-mlir-aie-from-wheels.sh my_install/mlir build install /opt/peano/llvm-aie \
             > /tmp/mlir-aie-build.log 2>&1 \
         || { tail -n 100 /tmp/mlir-aie-build.log; \
              for f in build/cmake.log build/CMakeFiles/CMakeError.log build/mlir-aie-ninja.log; do \
                  [[ -f "$f" ]] && { echo "==== $f"; tail -n 150 "$f"; }; done; \
              exit 1; }; } \
    && tail -n 5 /tmp/mlir-aie-build.log \
    && test -x install/bin/aiecc.py \
    && test -x install/bin/aie-opt \
    && test -f install/python/aie/compiler/aiecc/configure.py \
    && rm -rf build my_install /tmp/mlir-aie-build.log

# ---------------------------------------------------------------------------
# The image: the two trees above, xclbinutil, and what the kernel Makefiles
# run for "make bitstream", "make kernel" and "make link" (make, python3,
# grep/sed/awk). g++ and cmake are there for scripts that expect them; the
# host programs (run.exe) do not build here (no libxrt-dev, no XILINX_XRT,
# and host.mk compiles with -march=native).
FROM ${UBUNTU_IMAGE}

ARG MLIR_AIE_REPO
ARG MLIR_AIE_SHA
ARG LLVM_AIE_VERSION
ARG XRT_PPA
ARG XRT_UTILS_VERSION
ARG PYTHON312_VERSION

LABEL org.opencontainers.image.description="FastFlowLM dev kit: NPU kernel (xclbin) build environment (mlir-aie, Peano, xclbinutil)"
LABEL org.opencontainers.image.source="https://github.com/ROCm/FastFlowLM"
LABEL com.amd.flm.mlir-aie.repo="${MLIR_AIE_REPO}" \
      com.amd.flm.mlir-aie.sha="${MLIR_AIE_SHA}" \
      com.amd.flm.llvm-aie.version="${LLVM_AIE_VERSION}" \
      com.amd.flm.xrt-utils.version="${XRT_UTILS_VERSION}"

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# xclbinutil comes with libxrt-utils from the PPA, with the libxrt2 of the
# same build; software-properties-common is only needed to add it. Python is
# the version ironenv (copied below) was made with.
RUN for v in MLIR_AIE_SHA LLVM_AIE_VERSION XRT_PPA XRT_UTILS_VERSION PYTHON312_VERSION; do \
        [[ -n "${!v:-}" ]] || { echo "build arg $v is not set: pass flm-dev-kit/FLM_Xclbin/toolchain.lock" >&2; exit 1; }; \
    done \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
        "python3.12=${PYTHON312_VERSION}" \
    && add-apt-repository -y "$XRT_PPA" \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl git make zstd \
        "python3.12=${PYTHON312_VERSION}" \
        "libxrt-utils=${XRT_UTILS_VERSION}" "libxrt2=${XRT_UTILS_VERSION}" \
    && apt-get purge -y --auto-remove software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && [[ "$(dpkg-query -W -f='${Version}' python3.12)" == "$PYTHON312_VERSION" ]] \
    && command -v xclbinutil

COPY --from=peano /opt/peano /opt/peano
COPY --from=mlir-aie /opt/mlir-aie/install /opt/mlir-aie/install
COPY --from=mlir-aie /opt/mlir-aie/ironenv /opt/mlir-aie/ironenv
COPY --from=mlir-aie /opt/mlir-aie/utils /opt/mlir-aie/utils

# What setup.sh leaves behind: the venv activated, then mlir-aie's
# env_setup.sh with both folders. Peano stays off PATH, as env_setup.sh keeps
# it; the Makefiles call ${PEANO_INSTALL_DIR}/bin/clang++.
ENV MLIR_AIE_DIR=/opt/mlir-aie \
    MLIR_AIE_INSTALL_DIR=/opt/mlir-aie/install \
    PEANO_INSTALL_DIR=/opt/peano/llvm-aie \
    VIRTUAL_ENV=/opt/mlir-aie/ironenv \
    PATH=/opt/mlir-aie/install/bin:/opt/mlir-aie/ironenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/opt/mlir-aie/install/python \
    LD_LIBRARY_PATH=/opt/mlir-aie/install/lib

# The toolchain works and is the pinned one. Peano: it must target aie2p, and
# its banner must name the pinned commit (parsed from the banner line, not
# grepped from the whole output, whose InstalledDir would match anything).
RUN [[ "$(command -v python3)" == /opt/mlir-aie/ironenv/bin/python3 ]] \
    && python3 -c "import aie.iron, aie.extras, aie.compiler.aiecc.main" \
    && aiecc.py --help > /dev/null \
    && aie-opt --version \
    && xclbinutil --version | head -n 3 \
    && "$PEANO_INSTALL_DIR/bin/clang++" --target=aie2p-none-unknown-elf -E -x c++ /dev/null > /dev/null \
    && got="$("$PEANO_INSTALL_DIR/bin/clang++" --version 2>&1 \
              | sed -nE 's#.*github\.com/Xilinx/llvm-aie ([0-9a-f]+).*#\1#p' | head -n 1)" \
    && { [[ -n "$got" && "$got" == "${LLVM_AIE_VERSION##*+}"* ]] \
         || { echo "Peano is '${got:-unknown}', toolchain.lock pins ${LLVM_AIE_VERSION}" >&2; exit 1; }; }

# Run as an unprivileged user by default; nothing in a kernel build needs root.
# A GitHub Actions job that uses this image as its container needs root for
# the runner's workspace mounts: give that job "options: --user root".
# aie-build-check.sh runs it with the caller's uid (docker run -u).
RUN useradd --create-home --uid 1001 --shell /bin/bash aie \
    && mkdir -p /workspace && chown aie:aie /workspace

WORKDIR /workspace
USER aie

CMD ["/bin/bash"]
