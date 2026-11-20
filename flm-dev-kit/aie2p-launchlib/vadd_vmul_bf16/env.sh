#!/bin/bash
# env.sh -- run a command with the AIE2P toolchain on PATH.
#
#   ./env.sh make            ./env.sh aiecc.py --version
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# From the caller's environment, or from ../env.local.sh (ignored by version
# control: a few lines of VAR=value for one machine) when it exists:
#   MLIR_AIE_DIR  required: an mlir-aie build with install/ (aiecc.py, aie-translate)
#                 and ironenv/
#   PEANO_DIR     llvm-aie (Peano); default the one of that ironenv
#   IRON_PYPATH   the python path for the IRON scripts; default that ironenv's
#                 site-packages. (If it holds a stray dataclasses.py that breaks the
#                 system python 3.12, point this at a folder of links to the rest.)
# XRT comes from the system paths (/usr/include/xrt, /usr/lib/x86_64-linux-gnu).

LOCAL=$(dirname "$0")/../env.local.sh
[ -f "$LOCAL" ] && . "$LOCAL"
if [ -z "${MLIR_AIE_DIR:-}" ]; then
    echo "env.sh: set MLIR_AIE_DIR (an mlir-aie build), in the environment or in $LOCAL" >&2
    exit 2
fi
PEANO_DIR=${PEANO_DIR:-$MLIR_AIE_DIR/ironenv/lib/python3.12/site-packages/llvm-aie}
IRON_PYPATH=${IRON_PYPATH:-$MLIR_AIE_DIR/ironenv/lib/python3.12/site-packages}

export MLIR_AIE_DIR PEANO_DIR
export PEANO_INSTALL_DIR=$PEANO_DIR
export PYTHONPATH=$IRON_PYPATH:$MLIR_AIE_DIR/install/python
# /usr/bin before the rest: python3 must be the system 3.12 the IRON bindings
# were built for (aiecc.py runs `env python3` too).
export PATH=$MLIR_AIE_DIR/install/bin:$PEANO_DIR/bin:/usr/bin:$PATH
exec "$@"
