<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# FastFlowLM xclbin projects

## Environment setup

FastFlowLM requires mlir-aie. Before doing anything, make sure you have at least **XRT version 2.19.0** installed. If you do not have the driver installed, see Step 4.

* **Optional**: Use [pyenv](https://github.com/pyenv/pyenv) to install a customize manager 3.12 python version


* **Step 0**: Make sure you have python >= 3.12

  ```bash
  python3 --version
  ```

  If not, install it with:

  ```bash
  sudo apt install python3.12 python3.12-venv
  ```

* **Step 1**: Clone the mlir-aie repo to a place where you want to store the tool. The home directory is used in this tutorial:

  ```bash
  cd ~
  git clone https://github.com/ngdxzy/mlir-aie.git
  ```

* **Step 2**: FLM builds on this checkpoint, so create a new branch at the given commit hash. Also fetch submodules:

  ```bash
  cd mlir-aie
  git fetch origin flm_build
  git checkout flm_build
  git submodule update --init --recursive
  ```

* **Step 3**: Install some system-level dependencies:

  ```bash
  sudo apt install \
  build-essential clang clang-14 lld lld-14 cmake ninja-build python3-venv python3-pip
  ```

* **Step 4**: Install AIE tools

  * Install Vitis™ AIE Essentials from [Ryzen AI Software 1.3 Early Access](https://account.amd.com/en/member/ryzenai-sw-ea.html#tabs-a5e122f973-item-4757898120-tab). We assume you use the installation directory `/tools/ryzen_ai-1.3.0/vitis_aie_essentials`.

    > This is an early access lounge; you must register and be granted access.

  * Download the VAIML installer for Linux-based compilation: `ryzen_ai-1.3.0ea1.tgz`

  * Extract the required tools:

    ```bash
    tar -xzvf ryzen_ai-1.3.0ea1.tgz
    cd ryzen_ai-1.3.0
    mkdir vitis_aie_essentials
    mv vitis_aie_essentials*.whl vitis_aie_essentials
    cd vitis_aie_essentials
    unzip vitis_aie_essentials*.whl
    ```

  * Set up an AI Engine license.

  * Get the local license (`2025 AI Engine Tools License`) for AI Engine tools from
    [https://www.xilinx.com/getlicense](https://www.xilinx.com/getlicense).

  * Download and copy your license file (`Xilinx.lic`) to your preferred location, e.g. `/opt/Xilinx.lic`.

  * Set up your environment using the following script for Vitis™ AI Engine tools:

    ```bash
    #!/bin/bash
    #################################################################################
    # Setup Vitis AIE Essentials
    #################################################################################
    export AIETOOLS_ROOT=/tools/ryzen_ai-1.3.0/vitis_aie_essentials
    export PATH=$PATH:${AIETOOLS_ROOT}/bin
    export LM_LICENSE_FILE=/opt/Xilinx.lic
    ```

  * To install the driver, you have two choices:

    1. Run the script that comes with mlir-aie directly:

       ```bash
       bash ./utils/build_drivers.sh
       sudo reboot
       ```

    2. New drivers are not good for debugging for some reason, so the same script is copied from the mlir-aie repo (commit: `f66b515`) into the current repo. Simply replace the script and run the same commands.

       > If you use this method, you might encounter the NPU name inconsistency problem. Old XRT uses **NPU Kracken** as the device name, but it was later changed to **Krackan**. If you see errors such as `NO NPU FOUND` or `DEVICE NOT SUPPORTED`, you can manually fix the name check in the script or source.

* **Step 5**: Create an Iron Python virtual environment:

  ```bash
  python3.12 -m venv ironenv 
  source ironenv/bin/activate
  python3 -m pip install --upgrade pip
  ```

* **Step 6**: Install Python dependencies and mlir-aie. A specific version of mlir-aie is required.

  ```bash
  python3 -m pip install -r python/requirements.txt
  HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
  python3 -m pip install mlir_aie==0.0.1.2025101504+ae29ec5 -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-2/


  # for some utilities
  pip install sympy
  # This installs the pre-commit hooks defined in .pre-commit-config.yaml
  pre-commit install
  ```

* **Step 7**: Build llvm-aie; this should take a few minutes:

  ```bash
  bash ./utils/build-mlir-aie-from-wheels.sh
  ```

  After this, all required tools are installed. If you run `pip list | grep aie`, you should see (the date may be different but commit number shall match):

  ```bash
  $ pip list | grep aie
  aie-python-extras 0.0.8.2026061909+ba2e6fe
  llvm-aie          21.0.0.2026061101+50bd3557
  mlir-aie          0.0.1.2025101504+ae29ec5
  ```

* **Step 8**: Set up environment variables:

  ```bash
  source utils/env_setup.sh install
  ```

  The current session is now ready to compile xclbins. For future sessions, you can create a `setup.sh` script as follows:

  ```bash
  #!/bin/bash

  export AIETOOLS_ROOT=/tools/ryzen_ai-1.3.0/vitis_aie_essentials
  export PATH=$PATH:${AIETOOLS_ROOT}/bin
  export LM_LICENSE_FILE=/opt/Xilinx.lic

  MLIR_AIE_BUILD_DIR=~/mlir-aie

  source /opt/xilinx/xrt/setup.sh

  source ${MLIR_AIE_BUILD_DIR}/ironenv/bin/activate
  source ${MLIR_AIE_BUILD_DIR}/utils/env_setup.sh
  source ${MLIR_AIE_BUILD_DIR}/utils/env_setup.sh ${MLIR_AIE_BUILD_DIR}/install
  ```

  This file is also provided in this repo. Make the necessary changes if your paths are different. Run ```source setup.sh``` to activate the building environment.

## Test the build

* **Step 1**: Clone the **FLM_Xclbin** repo: 

  ```bash
  git clone https://github.com/FastFlowLM/FLM_Xclbin.git
  ```

* **Step 2**: Test the build:

  ```bash
  cd <FLM_XCLBIN>
  cd Dequant_Q4_0_w128x512
  make bitstream
  ```

  You should be able to find the final xclbin at:

  ```text
  build/xclbins/dequant.xclbin
  ```
