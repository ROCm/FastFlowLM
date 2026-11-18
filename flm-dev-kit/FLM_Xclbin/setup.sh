#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Kernel build environment (Peano). Source it after setting, if the defaults do not fit:
#   MLIR_AIE_DIR       mlir-aie folder with ironenv/ and install/ (default: ~/mlir-aie)
#   PEANO_INSTALL_DIR  the llvm-aie (Peano) to build with; see README.md for the pinned version
#   XILINX_XRT_SETUP   XRT's setup.sh (default: /opt/xilinx/xrt/setup.sh, skipped if missing)
# Chess only (ENABLE_CHESSCC=1): AIETOOLS_ROOT=<Vitis AIE Essentials>, LM_LICENSE_FILE=<license>.

MLIR_AIE_DIR="${MLIR_AIE_DIR:-$HOME/mlir-aie}"

if [ -n "${AIETOOLS_ROOT:-}" ]; then
    export AIETOOLS_ROOT
    export PATH="$PATH:$AIETOOLS_ROOT/bin"
fi
if [ -n "${LM_LICENSE_FILE:-}" ]; then
    export LM_LICENSE_FILE
fi

XILINX_XRT_SETUP="${XILINX_XRT_SETUP:-/opt/xilinx/xrt/setup.sh}"
if [ -f "$XILINX_XRT_SETUP" ]; then
    source "$XILINX_XRT_SETUP"
fi

source "$MLIR_AIE_DIR/ironenv/bin/activate"
if [ -n "${PEANO_INSTALL_DIR:-}" ]; then
    source "$MLIR_AIE_DIR/utils/env_setup.sh" "$MLIR_AIE_DIR/install" "$PEANO_INSTALL_DIR"
else
    source "$MLIR_AIE_DIR/utils/env_setup.sh" "$MLIR_AIE_DIR/install"
fi
