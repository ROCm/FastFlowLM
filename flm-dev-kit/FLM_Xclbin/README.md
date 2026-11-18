<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# FastFlowLM xclbin projects

Every folder here with a `Makefile` builds NPU kernels (`.xclbin`): IRON Python writes the design as
MLIR, Peano compiles the per-tile C++, and `aiecc.py` turns both into an xclbin. Peano is the default
compiler of every folder; Chess is an option (see the end of this file).

## What you need

- x86-64 Linux, tested on Ubuntu 24.04, with Python 3.12.
- About 15 GB of free disk space; the mlir-aie build alone is about 8 GB.
- To run what you build: an AMD Ryzen AI NPU (Strix, Strix Halo or Krackan) with the `amdxdna` driver
  and XRT 2.19 or newer. Building does not need the NPU.

```bash
sudo apt install build-essential g++-13 lld ninja-build git curl zstd \
    python3.12 python3.12-venv python3.12-dev \
    libboost-program-options-dev libboost-filesystem-dev
```

**Use exactly these versions.** A newer mlir-aie or Peano can build without any error and still produce
kernels that compute wrong numbers.

| Tool | Version | Source |
|---|---|---|
| mlir-aie | commit `4143812c93bf205250a9b762fb09147410e87558` | https://github.com/ngdxzy/mlir-aie (branch `flm_build`) |
| Peano (`llvm-aie`) | `22.0.0.2026090201+a36c62b9` | https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly |

## One-time setup

Pick a folder for the tools; this file uses `$HOME/flm-toolchain`.

**1. Build mlir-aie** (about 10 minutes on 32 cores):

```bash
export TOOLS=$HOME/flm-toolchain
mkdir -p $TOOLS && cd $TOOLS
git clone https://github.com/ngdxzy/mlir-aie.git
cd mlir-aie
git checkout 4143812c93bf205250a9b762fb09147410e87558
git submodule update --init --recursive

python3.12 -m venv ironenv
source ironenv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r python/requirements.txt
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
python3 -m pip install "mlir_aie==0.0.1.2025101504+ae29ec5" \
    -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-2
bash ./utils/build-mlir-aie-from-wheels.sh
deactivate
ls $TOOLS/mlir-aie/install/bin/aiecc.py     # must exist: the build script can report success after a failure
```

**2. Install Peano** in its own folder, not in the venv (about 1 minute):

```bash
$TOOLS/mlir-aie/ironenv/bin/python3 -m pip install --no-deps --target $TOOLS/peano \
    "llvm-aie==22.0.0.2026090201+a36c62b9" \
    -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

**3. The NPU driver** (only to run kernels): follow AMD's [xdna-driver](https://github.com/amd/xdna-driver)
instructions or use your distribution's packages. `build_drivers.sh` in this folder is the script
mlir-aie used to ship for the same job.

## In every new shell

```bash
export MLIR_AIE_DIR=$HOME/flm-toolchain/mlir-aie
export PEANO_INSTALL_DIR=$HOME/flm-toolchain/peano/llvm-aie
source flm-dev-kit/FLM_Xclbin/setup.sh
$PEANO_INSTALL_DIR/bin/clang++ --version | head -1    # clang version 22.0.0 (... a36c62b9 ...)
```

`setup.sh` activates the mlir-aie venv and runs mlir-aie's `utils/env_setup.sh` with **both** folders.
Set `PEANO_INSTALL_DIR` before sourcing it: without a Peano folder, `env_setup.sh` falls back to the
Peano in the venv, an older version (21) that builds wrong kernels, and if the venv has none it
pip-installs the newest nightly. `env_setup.sh` may print
`/opt/xilinx/xrt/bin/xrt-smi: No such file or directory` when XRT lives in `/usr`; that is harmless.

## Build a kernel

```bash
cd flm-dev-kit/FLM_Xclbin/Dequant_Q4_0_w128x512
make bitstream          # -> build/xclbins/dequant.xclbin
```

Folders that build one kernel per model take the model on the command line, for example
`make bitstream MODEL_TYPE=LLAMA_3_2_1B` in `generic_decoding_layer` (the `Makefile`'s `MODEL_TYPE ?=` is
only the default; `kernel_assignment.md` names the type each model needs). The xclbin lands in a `xclbins/` folder under `build/`
(`find build -path '*xclbins*' -name '*.xclbin'`), often with an `rtp_addresses.txt` next to it.

Useful targets: `make kernel` (only the per-tile C++, a fast way to find compile errors), `make link`
(only the IRON Python), `make clean_all` or `make clean` (what the build made). Not every folder defines
`link` and `clean_all`; check its `Makefile`, or delete its `build/` folder.

**Which folders a model needs:** `kernel_assignment.md` lists every model's xclbins and the build that
makes each one. A build name like `xclbins-Hy_MT2_1.8B_decoding` is the folder path with `/` written as
`_` (here `Hy_MT2_1.8B/decoding`); a trailing model type, as in
`xclbins-generic_decoding_layer_LLAMA_3_2_1B`, is the `MODEL_TYPE` to pass.

## Optional: Chess

`make ... ENABLE_CHESSCC=1` builds a folder with Chess (`xchesscc`) instead of Peano. It needs Vitis AIE
Essentials and an AI Engine license: set `AIETOOLS_ROOT=<Vitis AIE Essentials folder>` and
`LM_LICENSE_FILE=<license file>` before sourcing `setup.sh`, or put `xchesscc` on `PATH`. Clean the
folder (`make clean_all`, or delete `build/`) when switching between the two compilers.

How the makefiles find Vitis (`makefiles/common.mk`): `AIETOOLS_DIR` defaults to `$AIETOOLS_ROOT`, else
to the folder above the `bin/` that holds `xchesscc`. A Peano build never reads it. The Chess flags
(`CHESSCCWRAP*_FLAGS`) start with `$(AIETOOLS_CHECK)`, which expands to nothing when `AIETOOLS_DIR` is
set and stops make with a clear message when it is not. Because make expands those flags only inside
the Chess recipes, the check fires only for `ENABLE_CHESSCC=1`. That works even in folders whose
`Makefile` sets `ENABLE_CHESSCC` after including `common.mk`, where a plain `ifeq` on it would not.
