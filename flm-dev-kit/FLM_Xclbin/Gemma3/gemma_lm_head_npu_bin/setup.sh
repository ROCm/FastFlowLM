#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

export AIETOOLS_ROOT=/tools/ryzen_ai-1.3.0/vitis_aie_essentials
export PATH=$PATH:${AIETOOLS_ROOT}/bin
export LM_LICENSE_FILE=/opt/Xilinx.lic

MLIR_AIE_BUILD_DIR=~/mlir-aie

source /opt/xilinx/xrt/setup.sh

source ${MLIR_AIE_BUILD_DIR}/utils/env_setup.sh
