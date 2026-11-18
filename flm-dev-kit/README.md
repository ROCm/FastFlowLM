<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# FastFlowLM dev kit

The sources behind FastFlowLM's NPU support, for building and changing the parts `flm` loads:

| Folder | What it builds |
|---|---|
| [`FLM_DLL/`](FLM_DLL/README.md) | the **engine libraries**, C++20, one shared library per model family |
| [`FLM_Xclbin/`](FLM_Xclbin/README.md) | the **NPU kernels** (`.xclbin`): IRON Python → MLIR → Peano → `aiecc.py` |
| [`FLM_Q4NX_Converter/`](FLM_Q4NX_Converter/README.md) | the **weight converter**: GGUF (or safetensors) → `model.q4nx` |
| `FLM_Pybind/SmolVLA/` | a Python module for the SmolVLA robot policy |
| `aie2p-launchlib/` | an experimental library for switching apps inside one NPU context |

A model needs three matching parts: its `model.q4nx` weights, its `.xclbin` kernels, and its engine
library. Nothing checks that they belong together. A mismatch gives wrong output or a hang, not an
error, so rebuild all the parts a change touches.

## What you need

- x86-64 Linux, tested on Ubuntu 24.04, with `g++-13`, Python 3.12, Boost (`program_options`,
  `filesystem`) and the XRT development files.
- A CPU with **AVX-512**: the engines use it, and on a CPU without it they build but crash at run time.
- To run what you build: an AMD Ryzen AI NPU (Strix, Strix Halo or Krackan) with the `amdxdna` driver
  and XRT 2.19 or newer.
- For kernels: mlir-aie and Peano (`llvm-aie`) at the exact versions in
  [`FLM_Xclbin/README.md`](FLM_Xclbin/README.md). Chess (Vitis AIE Essentials, with a license) is
  optional; Peano builds every kernel.
- For HRX engine builds: the HRX package pinned in `../hrx-integration/hrx-release.env`; for the
  shippable HRX build, Docker.

## Build

Run these from the FastFlowLM repository root.

**Engines (XRT)**, all of them, into `flm-dev-kit/FLM_DLL/build/lib/` (about 6 minutes on 32 cores):

```bash
bash flm-dev-kit/FLM_DLL/build.sh
bash flm-dev-kit/FLM_DLL/build_hunyuan.sh      # or one engine and what it needs (build_<model>.sh)
```

**Engines (HRX):**

```bash
bash hrx-integration/fetch-hrx-release.sh      # once: the pinned HRX package
FLM_USE_HRX=1 bash flm-dev-kit/FLM_DLL/build.sh
bash flm-dev-kit/FLM_DLL/build_manylinux.sh    # the shippable build: glibc <= 2.28 container + ABI check
```

**A kernel** (first do the one-time setup in [`FLM_Xclbin/README.md`](FLM_Xclbin/README.md)):

```bash
export MLIR_AIE_DIR=$HOME/flm-toolchain/mlir-aie
export PEANO_INSTALL_DIR=$HOME/flm-toolchain/peano/llvm-aie
source flm-dev-kit/FLM_Xclbin/setup.sh
cd flm-dev-kit/FLM_Xclbin/Dequant_Q4_0_w128x512 && make bitstream   # -> build/xclbins/dequant.xclbin
```

[`FLM_Xclbin/kernel_assignment.md`](FLM_Xclbin/kernel_assignment.md) lists which kernel folders (and
`MODEL_TYPE`s) each model uses.

**Weights:**

```bash
cd flm-dev-kit/FLM_Q4NX_Converter
./setup_venv.sh && source activate.sh            # once: venv from the pinned requirements.txt
python convert.py -i model.gguf -o out/          # -> out/model.q4nx; python convert.py --help for the rest
```

## Use what you built

`flm` loads a model from its folder under `$FLM_MODEL_PATH/models/` (else `~/.config/flm/models/`):
`model.q4nx`, `config.json` and the tokenizer. The engine libraries and xclbins ship with `flm` itself.
To try yours, build `flm` from source (see the root [`README.md`](../README.md)) after you:

- copy the engine library to `../src/lib/xrt/` (XRT) or `../src/lib/hrx/` (HRX);
- copy each xclbin, under the name the model expects (`kernel_assignment.md`), to
  `../src/xclbins/<MODEL>-NPU2/`.

For a kernel-only change you can also swap the `.xclbin` in an installed `flm`; see
[`FLM_Xclbin/kernel_assignment.md`](FLM_Xclbin/kernel_assignment.md). Adding a whole model has more
steps: [`FLM_DLL/ADDING_A_NEW_MODEL.md`](FLM_DLL/ADDING_A_NEW_MODEL.md) and "Adding a model?" in
[`../CONTRIBUTING.md`](../CONTRIBUTING.md).

Most kernel folders have a test program (`make host`, then `./run.exe`). Many check the NPU against a
CPU reference on random inputs and print PASSED or FAILED; others compare with reference data you
point them at through `$FLM_REFERENCE_DIR`. Reference data is not shipped.

## Windows

Not tested in this release. [`FLM_DLL/README.md`](FLM_DLL/README.md) keeps the earlier Windows notes.

## License

The dev kit is MIT, like FastFlowLM's source: see [`../LICENSE_RUNTIME.txt`](../LICENSE_RUNTIME.txt).
Files that carry their own license line (for example code adapted from mlir-aie, under
`Apache-2.0 WITH LLVM-exception`) keep it. Third-party components are listed in
[`../THIRD_PARTY_NOTICES.txt`](../THIRD_PARTY_NOTICES.txt).

## More

- [`AGENTS.md`](AGENTS.md): the map, conventions and the constraints that break silently, for coding
  agents and contributors.
- [`FLM_DLL/ADDING_A_NEW_MODEL.md`](FLM_DLL/ADDING_A_NEW_MODEL.md),
  [`FLM_Xclbin/README.md`](FLM_Xclbin/README.md),
  [`FLM_Xclbin/kernel_assignment.md`](FLM_Xclbin/kernel_assignment.md),
  [`FLM_Q4NX_Converter/README.md`](FLM_Q4NX_Converter/README.md).
