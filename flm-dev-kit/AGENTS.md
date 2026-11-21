<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# AGENTS.md — FastFlowLM dev kit

Guide for coding agents and contributors working in `flm-dev-kit/`, the sources of FastFlowLM's NPU
engines, kernels and weight converter. Paths below are relative to `flm-dev-kit/`; `../` is the
FastFlowLM repository root. For building the `flm` application itself, pull requests and licensing,
read [`../CONTRIBUTING.md`](../CONTRIBUTING.md) first; this file covers only the dev kit. Outside users
start at [`README.md`](README.md).

## 1. What the dev kit produces

| Artifact | Produced by | Consumed by |
|---|---|---|
| `model.q4nx` (quantized weights) | `FLM_Q4NX_Converter/convert.py` | an engine at load time |
| `*.xclbin` (NPU kernels) | `FLM_Xclbin/<kernel>/Makefile` → IRON → `aiecc.py` | an engine at load time (`../src/xclbins/<MODEL>-NPU2/`) |
| engine library (`.so` / `.dll`) | `FLM_DLL/build.sh` (g++-13), `FLM_DLL/build_manylinux.sh` (HRX), or the Windows/CMake path | `flm`: `../src/lib/xrt/` for XRT and `../src/lib/hrx/` for HRX. Both are needed; see "Adding a model?" in `../CONTRIBUTING.md`. Windows XRT libraries are named `<family>_npu.dll`, without the `lib` prefix. |

A model runs only when all three agree. Nothing validates that a `.q4nx` matches the xclbins it is
paired with, or that an engine matches either: a mismatch gives garbage output or a hang, not an error.

## 2. Map

```
flm-dev-kit/
├── FLM_DLL/                 C++20 engines, one .so/.dll per model family
│   ├── build.sh             builds every unit: priority units first, then detail/*/
│   ├── build.ps1            the same on Windows (XRT, CMake + MSVC)
│   ├── build_<model>.sh     one engine + its deps (iteration)
│   ├── build_manylinux.sh   HRX engines inside the pinned manylinux_2_28 container, then check_abi.sh
│   ├── hrx.mk, cmake/flm_hrx.cmake   find the HRX package in ../hrx-integration/
│   ├── cmake/flm_xrt_win.cmake   Windows XRT/aiebu/Boost-header paths (FLM_WIN_DEPS_ROOT, default C:/dev)
│   ├── ADDING_A_NEW_MODEL.md  how to add an engine; read before creating one
│   ├── model_list.json, model_list_linux.json   model registries (see §9)
│   ├── include/             public headers: causal_lm.hpp, buffer.hpp, weight_desc.hpp,
│   │                        lm_config.hpp, models/<m>/, modules/, npu_utils/, tensor_utils/ …
│   └── detail/              implementations; each unit owns a Makefile + CMakeLists.txt
│       ├── q4nx/ mha/ gemm/ dequant/ lm_head/   priority units, built before any engine
│       ├── <family>_npu/    one engine (most dirs; see §5 Naming)
│       └── vision_common/   source pool, no Makefile (not a build unit)
├── FLM_Xclbin/              NPU kernel sources; a build unit is any dir with a Makefile
│   ├── setup.sh             kernel build environment (MLIR_AIE_DIR, PEANO_INSTALL_DIR; Chess vars optional)
│   ├── README.md            toolchain versions and one-time setup
│   ├── kernel_assignment.md which kernel build feeds which model, and with which compiler
│   └── host_common/         shared host-side C++ (buffer.hpp, npu_utils, aiebu headers)
├── FLM_Q4NX_Converter/      GGUF/safetensors → Q4NX (Python, pinned requirements.txt)
├── FLM_Pybind/SmolVLA/      pybind11 module for the SmolVLA policy
└── aie2p-launchlib/         EXPERIMENTAL app switching inside one AIE2P hw_context; nothing
                             else in the dev kit links it. Needs MLIR_AIE_DIR.
```

The aiebu libraries are not in the dev kit: XRT builds link FastFlowLM's `../src/lib/xrt/libaiebu.a`
(`aiebu_static.lib` on Windows), the same files `flm` ships.

## 3. Where to change what

| Task | Location |
|---|---|
| Add a model family | Follow `FLM_DLL/ADDING_A_NEW_MODEL.md` (template: `detail/gemma4_12b_npu/`). `build.sh` picks up any `detail/*/` with a Makefile. Then follow "Adding a model?" in `../CONTRIBUTING.md` (both backends, `../src/model_list.json`, `../src/xclbins/<MODEL>-NPU2/`). |
| Inference logic of one engine | `FLM_DLL/detail/<family>_npu/<family>_npu.cpp` |
| NPU instruction sequence | `FLM_DLL/detail/<family>_npu/<family>_npu_sequence.cpp` |
| Engine state | `FLM_DLL/detail/<family>_npu/<family>_npu_detail.hpp` (the `Impl` struct) |
| Public model interface | `FLM_DLL/include/causal_lm.hpp`. Every text engine implements it. |
| Shared layer (gemm/mha/dequant/lm_head) | `FLM_DLL/detail/<layer>/` + `FLM_DLL/include/modules/`. Rebuild everything. |
| Q4NX parsing | `FLM_DLL/detail/q4nx/`, `FLM_DLL/include/weight_desc.hpp` |
| Kernel dataflow (tiles, DMA, RTPs) | `FLM_Xclbin/<kernel>/iron/*_aie.py` |
| AIE compute code | `FLM_Xclbin/<kernel>/kernel*/*.cc` |
| Kernel model parameters | `FLM_Xclbin/<kernel>/iron/models/*.py`, `iron/model_config.py` (where present) |
| Chess vs Peano | `ENABLE_CHESSCC` (§8) |
| Converter support for an architecture | `FLM_Q4NX_Converter/q4nx/model_converter.py` + `FLM_Q4NX_Converter/configs/` |
| HRX package version | `../hrx-integration/hrx-release.env` (also used by the `flm` build) |
| manylinux image for HRX engines | `MANYLINUX_IMAGE` default in `FLM_DLL/build_manylinux.sh` |

## 4. Commands

Run from `flm-dev-kit/` unless noted. The build scripts change to their own folder, so they also work
from anywhere else.

| Command | Result | Time |
|---|---|---|
| `bash FLM_DLL/build.sh` | `rm -rf FLM_DLL/build`, then every unit with `make clean && make -j4`; lists failed dirs, exits 1 on failure | ~6 min on 32 cores |
| `bash FLM_DLL/build_<model>.sh` | one engine + its deps | minutes |
| `FLM_DLL\build.ps1 [-Clean] [-DepsRoot <dir>]` | Windows: every engine → `FLM_DLL/build/lib/*.dll`; stops at the first priority failure. Developer PowerShell; deps from `../.github/script/ci/setup_windows_deps.ps1` | minutes |
| `FLM_USE_HRX=1 bash FLM_DLL/build.sh` | the same against the HRX package (fetch it first: `bash ../hrx-integration/fetch-hrx-release.sh`) | ~6 min |
| `bash FLM_DLL/build_manylinux.sh` | HRX engines in Docker + `check_abi.sh`; the only way to get HRX libs FastFlowLM can ship | tens of minutes |
| `cd FLM_DLL/detail/<dir> && make clean && make -j4` | one unit → `FLM_DLL/build/lib/<TARGET>` (see the Makefile's `TARGET =`); needs its deps built first | 1-3 min |
| `source FLM_Xclbin/setup.sh` | kernel toolchain env; set `MLIR_AIE_DIR` and `PEANO_INSTALL_DIR` first (`FLM_Xclbin/README.md`) | seconds |
| `cd FLM_Xclbin/<kernel> && make bitstream [MODEL_TYPE=<T>]` | IRON → MLIR → `aiecc.py` → `build/[<T>/]xclbins/*.xclbin` + `rtp_addresses.txt` | seconds to 30 min |
| `make kernel` / `make link` / `make host` | single stages (`.cc`→`.o` / IRON→`.mlir` / host exe); not every unit has `link` | seconds-minutes |
| `make clean_all` (or `make clean`) | wipe the unit's build tree; not in every unit | instant |
| `cd FLM_Q4NX_Converter && ./setup_venv.sh && source activate.sh` | venv from the pinned `requirements.txt` | minutes |
| `python convert.py -i model.gguf -o out/` | `model.q4nx`; `-t {language,vision,audio}`, `-f <arch>`, `-e <safetensors>`; `--help` for the rest | minutes |
| `cd FLM_Pybind/SmolVLA && make` / `make python` | `run.exe` / `smolvla_npu*.so`; then `python verify_smolvla.py --steps 8` | minutes |
| `make -C aie2p-launchlib/launchlib test` | launchlib host unit tests (no NPU; needs a precompiled package, see that Makefile) | minutes |

Kernel makefiles add `-j8` (`makefiles/common.mk`) unless `-j` is set. Test hosts find models the way
`flm` does (`$FLM_MODEL_PATH/models/<model>`, else `~/.config/flm/models/<model>`) and reference data in
`$FLM_REFERENCE_DIR` or a `--reference` option; reference data is not shipped.

**Testing.** There is no unit-test suite, lint or formatter for the dev kit as a whole. What exists:
- Kernel test hosts (`make host`, then `./run.exe`): many compare the NPU against a CPU reference on
  random inputs by default (for example the Gemma4_12B_QAT and SMOLVLA expert attention hosts) and print
  PASSED / FAILED; others need reference data from `$FLM_REFERENCE_DIR`.
- `FLM_DLL/detail/minicpm_{v_4_7,ocr}_npu/host_ref/run_gate.sh <model_dir> <xclbin_dir>`: the NPU build
  of the engine scored against its all-host build.
- `aie2p-launchlib/launchlib` (`make test`) and `FLM_Pybind/SmolVLA` (`test_smolvla_*.py`).

Engines and kernels are otherwise validated by running the model in `flm` and comparing outputs and
accuracy. Do not invent a `make test` or claim tests pass.

**FastFlowLM CI** (`../.github/workflows/build-and-test.yml`; which jobs a change selects is decided by
`../.github/script/ci/changes.py`) runs these dev-kit jobs:
- `devkit-static` (`../.github/script/ci/devkit_check.py`, seconds): the data files below agree with
  each other and with the tree, every engine Makefile has `-Wl,-Bsymbolic-functions`, files a pull
  request adds carry the copyright notice, and an RTP lock change comes with the engines that
  hard-code those addresses. Its catalog rules also run when `../src/model_list.json`,
  `../src/xclbins/` or `../src/lib/` change. On a pull request, a binary the catalog needs and
  `../src/` lacks yet (a new model's kernels, a new engine's libraries) is *pending*, not an error,
  as long as the run stages it (below); anywhere else it is an error.
- `devkit-engines`: `FLM_DLL` built for Linux and Windows, XRT and HRX, on hosted runners, when a
  change touches it or a `../src/lib/xrt/` library it links, and on every push to main and nightly.
- `devkit-xclbins`: `FLM_Xclbin` built with Peano in the `aie-build` image (Chess needs a license no
  hosted runner has). A pull request builds the kernels it touches, plus the kernel assigned to each
  `../src/xclbins/` file it replaces; main builds every assigned kernel, nightly every build unit. The result is compared with
  `../src/xclbins/` (a report, never a failure), and a kernel whose RTP addresses differ from
  `FLM_Xclbin/rtp_addresses.lock.txt` fails the job.
- `devkit-stage` (pull requests whose engine or kernel source changed, and the full refresh below;
  `../.github/script/ci/devkit_stage.py`): puts what the two builds above made into every product
  build, before CMake runs, in the build's workspace only (never committed): the libraries of the
  engines serving the families the change touches (`.so`; `.dll` and `.lib` on Windows) into
  `../src/lib/<runtime>/`, and the rebuilt kernels of the models whose kernel source or
  `kernel_assignment.json` entry changed into `../src/xclbins/<folder>/<slot>`. A file `../src/`
  has is replaced; a new one only when `engines.json` or `kernel_assignment.json` declares it. A
  replaced library must keep the exports and imports the other libraries need (the ABI check). It
  uploads a manifest of every staged file (`devkit-stage-manifest-<os>-<runtime>`, kept 30 days),
  which the swap below reads. The product packages are then exactly what main would ship after the
  swap, and the product's NPU test jobs (Linux and Windows, XRT and HRX portable) test them once, on
  the union of the product's selection and the staged files' (the touched families, the models
  whose kernels were rebuilt and the new models, which are required). A change to the dev-kit CI
  alone builds the dev kit but stages nothing. The product builds of a pull request that stages
  wait for the dev-kit builds.
- `devkit-bench` (`../.github/workflows/devkit-bench.yml`, called by Build and Test after its tests on a pull request,
  in the same run): perf (`flm bench`) and GSM8K accuracy, only when engine or kernel source changed:
  on pull requests, of their own staged packages, reporting the pull request's numbers only (no
  comparison with a base package or the nightly baseline for now; a comparison is to be designed
  later), and nightly (job `devkit-bench-nightly`, which does not wait for the tests) when the
  source changed since the last baseline. Results go to one pull request
  comment; any bench failure other than the accuracy hard gate (perf, a pull, a timeout,
  infrastructure) warns; only a model failing that gate (acc_common: no predictions, or too many
  unanswered) fails the required check, which therefore waits for the benches. A manual dispatch of
  that workflow re-benches an earlier Build and Test run: it benches by default (force), and without
  `models` takes the nightly's rotation, so name the models to re-bench a pull request's. The full
  refresh benches every benchable model on the four runtimes and reports only its own numbers.

**Prebuilt binaries: never edit `../src/lib/{xrt,hrx}/` or `../src/xclbins/` by hand.** The engine
libraries (`.so`, `.dll`, `.lib`) and kernels (`<folder>/<slot>.xclbin`) that `flm` ships are
updated only by a pull request that changes dev-kit source, and the bytes committed are exactly the
bytes its Build and Test run built, staged into its product builds and NPU-tested. Kernel builds are
not byte-reproducible, so nothing is rebuilt afterwards: the tested files themselves are committed,
as a *swap commit*.
1. Change the dev-kit source and push. Do not touch `../src/lib/` or `../src/xclbins/`.
2. Build and Test builds the engines and kernels the change touches (`devkit-engines`,
   `devkit-xclbins`), stages them into the product builds (`devkit-stage`, which records every
   staged file in its manifests) and NPU-tests those packages in the product test jobs. Fix and push
   until everything passes. While the branch lacks the staged files, the run's `summary` (the
   required check) fails at its *Dev-kit swap pending* step, which prints the command below: the
   pull request cannot merge without its swap.
3. From a clean checkout of the pull request's branch at exactly the tested commit (nothing
   committed after it, nothing unpushed): `python .github/script/ci/devkit_swap.py fetch --run <id>`
   (`--dry-run` for the plan only; without `--run` it takes the newest run of `HEAD`). It uses your
   own GitHub access (the `gh` login, or `GH_TOKEN`; read access is enough), checks the run (this pull
   request's Build and Test at your `HEAD`, completed, every job passed one by one, the only failure
   allowed being the *Dev-kit swap pending* step; a failed bench job does not count unless a model
   failed the accuracy hard gate, which fails `summary`'s *Dev-kit checks*; the product test job of every system and runtime it staged files into passed;
   and no commit on main since the run's base touches `FLM_DLL/`, `FLM_Xclbin/`, `../src/lib/` or
   `../src/xclbins/`: otherwise merge main and let CI run again), downloads the staged files, checks
   every SHA-256 against the manifests, writes them and makes one commit with the trailers
   `Devkit-Swap-Run: <run id>` and `Devkit-Swap-Tested: <tested sha>`. It does not push. The dev-kit
   artifacts are kept 7 days: once they expire, re-run the workflow.
4. `git push`. Build and Test's changes job verifies the swap commit (`devkit_swap.py verify`: its
   parent is the tested commit, it changes only files the run's manifests list, each to exactly the
   staged bytes, it carries every file the run staged, and the run passed as fetch requires) and then
   builds and tests nothing again; `devkit-static` still checks the catalog. Then the pull request is
   reviewed and merged.
5. Pushes after the swap (a review fix, a merge of main): when they change no dev-kit engine or
   kernel source or build input (`changes.py` `after_swap` decides from the files changed since the
   newest swap commit), nothing is rebuilt or staged; the product test jobs test the committed
   binaries and `devkit_swap.py check` re-verifies the swap commit. When they do change dev-kit
   source, the run builds and stages again and asks for a new swap: run `fetch` again on that run.
   So make the swap after the last dev-kit source change.

A swap pull request must not change the CI that stages the files, writes and checks the manifests
or builds the packages they are tested in (`devkit_stage.py`, `devkit_overlay.py`, `devkit_swap.py`,
`devkit_check.py`, the `devkit-stage-apply`, `detect-changes`, `lin-build` and `win-build` actions,
`build-and-test.yml`, `lin-build.yml`, `win-build.yml`, `devkit-engines.yml`, `devkit-xclbins.yml`,
and the NPU test harness that decides whether they pass: `../.github/script/npu_test/`, the
`npu-test` and `install-package` actions; `devkit_swap.py SWAP_CI_PATHS`): `verify` then refuses
the swap, the *Dev-kit swap pending* step says so, and the pull request has to be split (the CI change first). `changes.py` is not on that list: a
new model family needs its engine table. `verify` and `check` run from the pull request's own merge
commit, so they keep the flow honest but are no enforcement boundary: reviewers must read any
`.github/` change in a pull request that touches `../src/lib/` or `../src/xclbins/`, and the real
guarantee is that review plus branch protection.

`devkit_swap.py check` runs on every pull request and fails it (through `summary`) on any change
under `../src/lib/` or `../src/xclbins/` that a verified swap commit did not bring in, other than the
deletion of a file the catalog no longer needs and the third-party files (`libaiebu.a`,
`aiebu_static.lib`, the runtime libraries directly under `../src/lib/`).

**Adding a model** (engine, kernels, catalog entry) goes through the same flow; its binaries are
committed only by the swap:
1. Add the dev-kit source: the engine under `FLM_DLL/detail/<engine>/` (or the model in an existing
   engine), its kernels under `FLM_Xclbin/`, and their entries in `FLM_DLL/engines.json`,
   `FLM_Xclbin/kernel_assignment.json` and `.md`, `build_units.json` and the RTP lock as needed.
2. Add the product side: the catalog entry in `../src/model_list.json` (and `model_info.json`), the
   frontend, the engine's line in `FLM_ENGINE_LINK_LIBS` (`../src/CMakeLists.txt`), and the engine
   table in `../.github/script/ci/changes.py`. Do not add anything under `../src/lib/` or
   `../src/xclbins/`, and do not change the NPU test harness (`../.github/script/npu_test/`) in this
   pull request: it is swap CI (above). A new family needs no `plan.py` entry (the pr tier then
   takes its smallest model); a pr-tier entry or a new suite goes in a pull request of its own.
3. Push. `devkit-static` reports the engine's libraries and the model's kernel slots as pending
   (not in `../src/` yet); `devkit-engines` and `devkit-xclbins` build them; `devkit-stage` stages
   them as new files (declared by `engines.json` and `kernel_assignment.json`), so the product builds
   link the new engine and package the new kernels; the product test jobs test the new model as a
   required model, beside the rest of their selection.
4. `summary` says *Dev-kit swap pending*: run `python .github/script/ci/devkit_swap.py fetch --run
   <id>` and push. CI verifies the swap commit and tests nothing again; the pull request is reviewed
   and merged, and main has the engine, its kernels and the source they were built from. A later
   push that changes dev-kit source needs a new run and a new fetch (step 5 above).

**Full refresh** (to align main with the dev kit once, or after a toolchain change): dispatch Build
and Test on the pull request's branch with `gh workflow run build-and-test.yml --ref <branch> -f
devkit-full-refresh=true`. It builds every engine and every shipped kernel, stages all of them into
the product builds, runs the product test jobs on the full tier with every NPU catalog model
required, and benches every benchable model. When everything but the *Dev-kit swap pending* step
has passed (no commit pushed to the branch meanwhile), run `python .github/script/ci/devkit_swap.py
fetch --run <its run id>` and push: the swap commit takes every file its manifests list, and
`verify` accepts it as above.

Keep in sync, in the same pull request: `FLM_Xclbin/build_units.json` (every directory with a
Makefile and its `MODEL_TYPE`s), `FLM_Xclbin/kernel_assignment.json` and `.md` (model → folder → slot
→ artifact), `FLM_DLL/engines.json` (engine → families, models, uses; shared units;
`served_elsewhere`), `FLM_Xclbin/rtp_addresses.lock.txt`, and the engine table in
`../.github/script/ci/changes.py`. **The dev kit is the source of what `flm` ships:** every NPU
(`aie2p`) model in `../src/model_list.json` needs a `kernel_assignment.json` entry whose folder is
its xclbin folder (a tag that only reuses another model's kernels names that folder in
`xclbin_folder` instead) and an `engines.json` engine listing it; every assigned slot ships as
`../src/xclbins/<folder>/<slot>`, and every shipped `.xclbin` is assigned; every library in
`../src/lib/{xrt,hrx}/` is built by an `FLM_DLL/detail/` directory in `engines.json`, and each such
library ships as `lib<name>.so`, `<name>.dll` and `<name>.lib` for both runtimes. A model added
only under `../src/` fails `devkit-static`. The few deliberate exceptions are listed, with reasons,
in `CATALOG_ALLOW` in `devkit_check.py`.

## 5. Conventions

- **Copyright header on every new source file**, after any shebang or coding line:
  `// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.` for
  `.cpp/.cc/.h/.hpp`; the same with `#` for `.py/.sh/.mk/Makefile/CMakeLists.txt/.cmake/.yml`;
  inside `<!-- -->` for `.md/.html`. No year, no SPDX line: AMD code is MIT, see
  [`../LICENSE_RUNTIME.txt`](../LICENSE_RUNTIME.txt). A copied file keeps its header. Files written
  with an AI assistant add `Portions of this file consist of AI-generated content.` on the next line.
  - Code copied or adapted from mlir-aie/IRON: the AMD notice plus the upstream license line
    `SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception` (`MIT` for `aie_kernel_utils.h`).
    Never a Xilinx notice.
  - Other third-party code (Hugging Face transformers, llama.cpp, model cards …): keep the
    original notice and license; add
    `Modifications Copyright(C) <year> Advanced Micro Devices, Inc. All rights reserved.` below it,
    and record the component in `../THIRD_PARTY_NOTICES.txt`.
  - If a copied file lost its notice, restore the upstream copyright line and license from its
    source (file header or `LICENSE`) and name the source in the PR.
  - Data and generated files (`.json`, `.csv`) get no header.
- **Scalar 1/sqrt in a kernel:** use `flm_rsqrt()` from the unit's `flm_rsqrt.h` (copy it into a
  new kernel folder). Do not hand-roll or copy bit-trick approximations.
- **pimpl pairing.** `include/models/<m>/<m>_npu.hpp` declares the class with `struct Impl; Impl* _impl;`;
  `detail/<m>_npu/<m>_npu_detail.hpp` defines `Impl`. State goes in `Impl`. Engines are non-copyable.
- **Naming.** Most engine dirs are `detail/<family>_npu/` → `lib<family>_npu.so`; each Makefile's
  `TARGET =` line is authoritative (exceptions include `gemma4e_flash`, `gemma4e_dflash`,
  `gemma4e_dflash_flash`, `gemma_embedding`, `qwen3vl_flash`, `qwen3_5_omni`). Use the `_npu` form
  for new engines. Kernel dirs are descriptive (`Dequant_Q4_1_w128x512`) or `<Family>/<stage>`
  (`Qwen3_6/attention_decoding`).
- **Kernel artifact names** in `FLM_Xclbin/kernel_assignment.md` are `xclbins-<folder>[_<MODEL_TYPE>]`:
  the kernel's folder path under `FLM_Xclbin/` with `/` replaced by `_`, plus the `MODEL_TYPE`
  if the unit fans out (e.g. `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_4`).
- **Kernel unit anatomy.** `iron/` (dataflow), `kernel/` or `kernels/` (AIE C++), `host/` (driver),
  `makefiles/` (`common.mk`, `kernel.mk`, `bitstream.mk`, `host.mk`). The top `Makefile` sets
  `MODEL_TYPE`, `KERNEL_ARGS`, `IRON_ARGS`.
- **`FLM_Xclbin/kernel_assignment.md`** is the model → kernel map. Update it when you change which
  kernel build a model uses.
- **Engine flags:** `-std=c++20 -fPIC -Wall -O3 -fopenmp`, then `$(FLM_MARCH) -ffast-math`, and
  `-fmax-errors=1` (q4nx: `-std=c++20 -fPIC -Wall` only). `FLM_DLL/flags.mk` sets `FLM_MARCH` to the
  product's AVX-512 set (`-mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma`, no `-march=native`) and
  `XRT_ROOT` (default `/opt/xilinx/xrt`, `/usr` for a distro libxrt-dev); gemma, gpt_oss, whisper and
  qwen3_asr add the BF16/VNNI/VBMI flags their unguarded intrinsics need. Threads via `OMP_NUM_THREADS`.
- **Vendored, do not edit:** `FLM_DLL/include/nlohmann/`, `FLM_DLL/include/aiebu/`,
  `FLM_Xclbin/host_common/aiebu/`, `FLM_DLL/include/tokenizer/tokenizers_c.h` and `tokenizers_cpp.h`,
  `FLM_DLL/include/base64.hpp`, every `biovault_bfloat16.h` and every `npu_utils/amdxdna_accel.h`.
  Full list: `../THIRD_PARTY_NOTICES.txt`; FastFlowLM's own vendored trees are in `../CONTRIBUTING.md`.

## 6. Data and control flow

Engine: `LM_Config` parses `config.json` → `npu_xclbin_manager` opens the device and registers the
model's xclbins → the constructor allocates KV caches and per-kernel `npu_app`s →
`load_weights(Q4NX&)` stages weights → `prefill(ids)` runs the prompt, `forward(id)` decodes one
token; `checkpoint()` / `restore()` save and restore KV state.

Kernel build: `iron/*_aie.py` → `build/[<MT>/]mlir/*.mlir` → `aiecc.py` → `.xclbin` →
`build/[<MT>/]xclbins/`. The bitstream rule also writes `%RTP` buffer addresses to
`xclbins/rtp_addresses.txt`. That file is part of the host contract (§7).

## 7. Constraints: these break silently

1. **`-Wl,-Bsymbolic-functions` in every `FLM_DLL/detail/*/Makefile`.** Each engine `.so` compiles its
   own copy of the shared runtime (`npu_sequence`, `npu_app_manager`, `weight_desc_t`, `buffer`, …)
   with default visibility, and `flm` links every engine at once. Without the flag, ELF
   interposition binds every reference, including calls inside one `.so`, to whichever engine is
   first in `DT_NEEDED`: you silently run another model's code. Functions only, so vtables and
   typeinfo stay interposable and `dynamic_cast` across `.so`s works. Copy the flag and the comment
   above `LIB_FLAGS += -Wl,-Bsymbolic-functions` into every new engine Makefile. Do not share
   headers with inline functions across `detail/` dirs; copy them under a new namespace (see
   `detail/minicpm_*/image/vision_common.hpp`).
2. **`buffer.hpp` ownership.** Rules for `bytes` / `buffer<T>`:
   - `FLM_Xclbin/host_common/buffer.hpp` and the HRX branch (`FLM_USE_HRX`) of
     `FLM_DLL/include/buffer.hpp` share the allocation on copy (`shared_ptr`) and zero-fill on
     `resize()`. Do not regress either.
   - The XRT branch of `FLM_DLL/include/buffer.hpp` (the default build) still copies **without**
     ownership: `unique_ptr`, and both the copy constructor and copy assignment set
     `is_owner_ = false`. Its constructor zero-fills but its `resize()` does not. A copy of an
     owning buffer dangles once the original dies: heap-use-after-free, garbage that appears far
     from the cause. Move owning buffers (`std::move`), never copy or copy-assign them, and never
     rely on contents after `resize()`.
3. **Build order.** `q4nx mha gemm dequant lm_head` (`FLM_DLL/build.sh`, `priority_dirs`) build
   first because engines depend on them (some link their libraries from `FLM_DLL/build/lib`,
   others compile their sources directly). Do not reorder or parallelize across that boundary.
4. **RTP addresses.** The host writes runtime parameters at fixed addresses. If a kernel rebuild
   or an `iron/` change moves an RTP, the engine writes to a stale address: wrong output or a
   decode hang, no error. After rebuilding, diff `build/[<MT>/]xclbins/rtp_addresses.txt` against
   the addresses the engine hardcodes (for example `detail/hy_mem_npu/hy_mem_npu_sequence.hpp`) and
   the `rtp_expected/` files where a unit has them. Never invent RTP values in a config.
5. **`include/causal_lm.hpp`** is implemented by every text engine (11 pure virtuals). Adding or
   changing a virtual is a lockstep, dev-kit-wide edit and a change to `flm`.
6. **Weights/kernels/engine agree only by convention** (§1). Rebuilding one without the others is
   a silent mismatch.
7. **Toolchain versions.** Build kernels only with the mlir-aie commit and Peano version in
   `FLM_Xclbin/README.md`. A newer or older one can build without error and compute wrong numbers;
   the Peano that mlir-aie's build puts in its venv (21.x) is one of them.

## 8. Gotchas

- **AVX-512 is mandatory** for engines: built on a host without it they compile, then die with SIGILL.
  `make FLM_MARCH=-march=native` tunes a local build to its host; such a library may fault elsewhere.
- **Chess vs Peano.** Every kernel defaults to Peano (`ENABLE_CHESSCC ?= 0`: `clang++
  --target=aie2p-none-unknown-elf`, `aiecc --no-xchesscc --no-xbridge`). `make … ENABLE_CHESSCC=1`
  builds with Chess (`xchesscc_wrapper`, `--xchesscc --xbridge`); it needs Vitis AIE Essentials
  (`AIETOOLS_ROOT`, or `xchesscc` on `PATH`) and a license (`LM_LICENSE_FILE`), and stops with a clear
  message when Vitis is missing. There is no automatic fallback. Clean the unit when switching.
- **`setup.sh` and `PEANO_INSTALL_DIR`.** Set it before sourcing `FLM_Xclbin/setup.sh`: without it
  mlir-aie's `env_setup.sh` uses the venv's Peano, or pip-installs the newest nightly.
- **Kernel builds take seconds to 30 min** per `MODEL_TYPE`. Fail fast with `make kernel`.
- **BFP16 emulation.** `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` is set in most kernels'
  `makefiles/common.mk` (`PEANOWRAP2P_FLAGS` / `CHESSCCWRAP2P_FLAGS`), not all; check the kernel's
  `common.mk`. It changes numerics.
- `FLM_DLL/detail/vision_common/` has no Makefile; it is compiled into the vision engines.
- **Windows** engines take every dependency path from `FLM_DLL/cmake/flm_xrt_win.cmake`; do not add
  `C:/...` paths to an engine CMakeLists. They need Boost *headers* (XRT's `xrt/detail/any.h` uses
  `boost::any` under MSVC without `/Zc:__cplusplus`) but link no Boost library. An engine CMakeLists
  only warns when a priority `.lib` is missing; use `build.ps1`, which stops instead.
- Two registries, `FLM_DLL/model_list.json` and `model_list_linux.json`; FastFlowLM's
  `../src/model_list.json` is what `flm` reads.

## 9. Not documented anywhere

- No exported-symbol contract for engine libraries; the ABI is the C++ headers.
- Which of the two `FLM_DLL` model registries is authoritative per platform.
