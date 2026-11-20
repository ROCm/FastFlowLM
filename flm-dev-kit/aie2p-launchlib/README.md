<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# aie2p-launchlib

Switch the NPU from one app to another **inside one hw_context**, with control packets, instead of switching
hw_contexts. For AIE2P (npu2: Strix, Strix Halo, Krackan). The same design concept as `aie4-launchlib` (branch
`flm_aie4_double_pm`), without the PM ping-pong.

## Why

FastFlowLM runs each xclbin in its own hw_context, so every time an engine moves from one xclbin to another (matmul
→ attention, ...) the NPU switches hw_contexts: ~2.2 ms on 32 cores, ~2.5 ms on average in FLM's traced models
(`flm_survey/`). The launch library keeps all apps in one hw_context and builds, offline-from-aiecc and in
memory, a small "warm" kernel per switch that rewrites only what the next app needs. Control packets carry most of
that rewrite down every column at once.

## What it gains (measured)

**Switch cost**, µs (beyond a rerun of the same app; `vadd_vmul_bf16`, two apps on the same cores; Linux = npu5
Strix Halo, Windows = Krackan):

| array | two hw_contexts (FLM today) | warm switch, full | warm switch, program only | gain |
|---|---|---|---|---|
| 32 cores (8 × 4), Linux | 2,196 | 77.5 | **24.5** | 28–90× |
| 32 cores (8 × 4), Windows | 2,321 | 89.7 | **26.2** | 26–89× |
| 16 cores (8 × 2), Linux | 1,584 | 64.3 | **16.3** | 25–97× |
| 8 cores (4 × 2), Linux | 815 | 30.8 | **8.9** | 26–92× |
| 1 core (1 × 1), Linux | 143 | 7.3 | **3.4** | 20–42× |

Linux: median of 3 runs, every array from 1 to 32 cores in `vadd_vmul_bf16/results/sweep.md`; Windows: one run
(`vadd_vmul_bf16/WINDOWS.md`).

A *full* switch works between any two apps; a *program-only* switch needs the two apps to share a floorplan (same
BDs, locks, routes) and is then just the new program.

**Whole launch on real kernels**, µs per launch including the switch:

| what | two hw_contexts (FLM today) | one hw_context, warm | gain |
|---|---|---|---|
| Whisper-V3-Turbo matmul ↔ attention, alternating, Linux | 2,706 | **190** | 14× |
| the same, Windows | 2,773 | **186** | 15× |
| `pm_runtime_load_singletile`: vadd → vmul → vminus → ..., three floorplans, 32 cores, Linux (ELF / xclbin flow) | > 2,200 (the switch alone) | **111 / 136** (3 warm switches), 109 / 135 (ping-pong) | 16–20× |
| the same, Windows | > 2,300 | 121 / 133, 117 / 131 | 17–20× |

**Stable**: about 1.8 million checked launches on Linux and 0.95 million on Windows with control packets (random
walks over every kernel kind, runlists of 1–1000, buffers up to 16.8 MB, 50 processes coming and going): **0 wrong
outputs, 0 hangs** (`vadd_vmul_bf16/results/`, `pm_runtime_load_singletile/results/`).

## How it works

`launch::build(list, sequences)` reads each app's package (made from aiecc's output) and builds one instruction
stream per kernel of the list:

| kernel | what it does | when |
|---|---|---|
| `cold(app)` | resets the array and loads the app | first launch of a runlist, or after anything else used the NPU |
| `warm(prev, app)` | switches the array from `prev` to `app`: the whole switch, or only the program (`{.program_only = true}`) | between apps |
| `rerun(app)` | the app's runtime sequence again | the same app again |

The host launches them as `xrt::run`s of one xclbin's kernel in an `xrt::runlist`, all in one hw_context, or as
ELFs (FLM's flow; the library writes them, no aiebu). `Transport::ctrlpkt` (default) sends the program, locks,
BDs and chain starts as control packets on each column's shim DMA, all columns in parallel, and checks every kernel
on a switch model before it runs; `Transport::txn` (command-processor writes only) is the fallback. Details:
`launchlib/README.md`.

Two rules:

- **One context.** The array's state lives only while this hw_context is the only one running on the NPU; when
  another runs, this context comes back as its xclbin. Start each runlist with `cold` after anything else may
  have used the NPU (`launch::Session` checks it for you; `launchlib/README.md` §3.2).
- **No empty-context churn.** A process that keeps opening and closing hw_contexts that run nothing can corrupt
  our runlists without the Session seeing it (Linux) or slow them down a lot (Windows). Do not share the NPU with
  one.

## Folders

| Folder | What is in it |
|---|---|
| `launchlib/` | The header-only C++20 library, its tools and host unit tests. |
| `vadd_vmul_bf16/` | `y = (A + B) * C + D`: two programs on the same cores, 1 to 32 cores; switch costs, stress and pressure tests; the Windows build (`WINDOWS.md`). |
| `pm_runtime_load_singletile/` | Three apps (vadd, vmul, vminus fp32) in three floorplans on 32 cores: the AIE4 host test, stress, pressure. |
| `whisper_encoder/` | Whisper-V3-Turbo's matmul and attention kernels switched in one hw_context; the hang it found (fixed in the library). |
| `flm_survey/` | Where FastFlowLM switches hw_contexts (18 models, traced) and what warm switches would save. |

The folders must stay side by side (the examples use `../launchlib`).

## Environment

`vadd_vmul_bf16/env.sh` (and its copy in `pm_runtime_load_singletile/`) puts the toolchain on PATH: an mlir-aie
build with Peano, given by `MLIR_AIE_DIR` (required), `PEANO_DIR` and `IRON_PYPATH` (both default to that build's
ironenv). Set them in the environment or in `env.local.sh` next to this README (one machine's `VAR=value` lines,
ignored by version control). XRT is in the system paths. Measured with XRT 2.21.75 and amdxdna 7.0.0-rc1 (Linux),
XRT 2.21.0 (Windows).

The packages (`precompiled/` of each example: xclbin, configure sequence, instructions) are built locally, not
checked in: the first `make` of an example builds them from source (aiecc, Peano, about a minute), or
`make precompile` does it explicitly. The library tests and the host then need only a C++20 compiler and XRT.

## Use

| Command | What it does |
|---|---|
| `make -C launchlib test` | Host unit tests of the library on the example's package (no device). |
| `cd vadd_vmul_bf16 && ./env.sh make run COLS=8 CORES=4` | The switch-cost table's row for 32 cores. |
| `cd vadd_vmul_bf16 && ./env.sh tools/sweep.sh` | The switch costs of every array (`vadd_vmul_bf16/results/sweep.md`). |
| `cd vadd_vmul_bf16 && ./env.sh make stress COLS=8 CORES=4` | Stress and corner-case tests (32 cores). |
| `cd pm_runtime_load_singletile && ./env.sh make test COLS=8` | The three-app host test (32 cores). |
| `cd whisper_encoder && ../vadd_vmul_bf16/env.sh make switch` | Whisper's matmul ↔ attention, one context vs two. |
