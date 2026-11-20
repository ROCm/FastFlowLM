<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# pm_runtime_load_singletile (AIE2P): three apps, three floorplans, one hw_context

The AIE2P counterpart of `aie4-launchlib/pm_runtime_load_singletile` (branch `flm_aie4_double_pm`). Three fp32
apps that use the same cores in **different floorplans** (other routes, other DMA channels, other locks), switched
by the launch kernels of `../launchlib` in one hw_context, and the AIE4 host test with its rows and JSON lists.

`../vadd_vmul_bf16` switches between two apps of **one** floorplan (only the program differs). Here a switch
changes everything, as on AIE4.

## The three apps

COLS columns (1..8) of 4 cores; one core works on N = 256 fp32 per launch; column c owns slice c of B and C.

| App | C = | How A gets to the cores | Shim channels of column c |
|---|---|---|---|
| vadd | A + B (A: 256 values, the same for every core) | column 0's shim, broadcast to every core of every column | col 0: A on MM2S 0, B on MM2S 1; others: B on MM2S 0; C on S2MM 0 |
| vmul | A x B | each column's A and B joined in the mem tile, the block broadcast to the column's cores; each core takes its slice | A on MM2S 0, B on MM2S 1, C on S2MM 0 |
| vminus | A - B | **no DMA**: the shim streams the column's A straight to the stream port of each core (`get_ss_int`); each core keeps its slice. Each core also has an RTP buffer (`ct_rtp_<c>_<r>`, word 0 = 256) and an RTP lock (`ct_rtplock_<c>_<r>`, lock 15) that the runtime sequence writes, as AIE4's vminus | A on MM2S 1, B on MM2S 0, C on S2MM 0 |

The cores loop forever (as on AIE4). Sources: `<app>_fp32/kernel.cc` (Peano) and `<app>_fp32/<app>_fp32_placed.py`
(IRON, `--cols`).

## AIE4 examples and what became of them

| AIE4 | Here | Status |
|---|---|---|
| `vadd_fp32`, `vmul_fp32`, `vminus_fp32` | the same three apps, other floorplans (AIE2P has 6 shim MM2S per mem tile, no CERT) | done, tested 1..8 columns |
| `host_test/test.cpp` rows: reset, cold, cold + rerun, chain, ping-pong | the same rows; the chain is the cycle of warm kernels; ping-pong = two warm kernels A -> B, B -> A (AIE2P has no second PM bank, so each one writes its program) | done |
| `three_apps.json`, `ping_pong.json`, `sequence.json` (`vadd_rev`) | the same three lists (`three_apps.json` builds the same 11 kernels as `--apps`, checked) | done |
| two transports (packets, ucdma) | two transports, `Transport::ctrlpkt` (the default: control packets from the shim DMAs) and `txn` (launchlib README section 4.1); two flows: xclbin (instruction BO) and ELF (as FLM) | done |
| `checks/check_sequence` | each C++ sequence == aiecc's `insts.bin`, word for word | done |
| `vpick_nocp` (the uC polls a lock, copies BD words the core wrote into a shim BD) | **cannot be written**: AIE2P instructions have mask-poll but no register-to-register copy and no branch | not ported |
| `vpick` (the core moves a shim BD with control packets) | needs control packets from the core to the shim | not ported (here only the shim DMAs send control packets) |

## Use

Run from this folder. `env.sh` sets up the toolchain (only needed to build the designs).

| Command | What it does |
|---|---|
| `./env.sh make test` | Builds the host, runs every row on the NPU with `precompiled/c4` (16 cores). `COLS=8` for 32 cores. `APPS=`, `CONFIG=host_test/three_apps.json`, `ITERS=` |
| `./env.sh make check` | No NPU: `test_txn` on each package binary, `check_sequence`, and `test_build` on all six pairs of apps (register model, below) |
| `./env.sh make stress` | `stress.exe --mode all` (runlist lengths, then a random walk) |
| `./env.sh tools/pressure.sh` | P0..P11 on 32 cores, about 25 minutes; see `results/` |
| `./env.sh make precompile COLS=8` | Builds the designs from source (aiecc, Peano) into `precompiled/c8` |

`test.exe --package DIR [--apps a,b,c | --config FILE.json] [--iters N] [--flow xclbin|elf] [--build-only]
[--exclusive]`.
`stress.exe --mode random|lengths|order|recover|cap|ctx|guard|watch|lifetime|latency|churn|all [--flow elf] [--policy P]
[--max-gap US] [--no-guard] [--blind] [--exclusive] [--max-failures N] [--child loop|churn]` (header of
`host_test/stress.cpp`).

Both programs run the kernels through `launch::Session` (`../launchlib/include/launch/xrt/session.hpp`): it builds
the kernels, opens the hw_context, refuses an illegal order before it runs, runs warm and rerun launches as cold
kernels while another hw_context is on the NPU (always when it cannot see them, unless `--exclusive`; on Windows it
sees them only when built with `-DNPU_WATCH=ON`, below),
opens a new hw_context after a failed or disturbed run (on Windows: at most 2 in a row, then it stops), and splits
long ELF runs into runlists that fit (see the library README).

## Results

Control packets (the default), 32 cores (`COLS=8`), every row PASS, mean µs per launch of a runlist of 100
(`make test COLS=8`; Linux: npu5 Strix Halo, Windows: Krackan):

| Row | Linux xclbin | Linux ELF | Windows xclbin | Windows ELF |
|---|---|---|---|---|
| reset | 172.9 | 169.5 | | |
| cold_vadd | 153.6 | 136.8 | 152.6 | 146.4 |
| cold_vmul | 168.7 | 137.9 | 157.1 | 143.0 |
| cold_vminus | 184.0 | 168.2 | 183.4 | 168.2 |
| cold_vadd, rerun_vadd | 35.8 | 20.4 | 28.1 | 20.0 |
| cold_vmul, rerun_vmul | 34.4 | 23.0 | 31.7 | 19.1 |
| cold_vminus, rerun_vminus (RTP + lock writes) | 58.2 | 41.3 | 52.5 | 41.3 |
| cold_vadd, 3 warm (vadd -> vmul -> vminus -> vadd) | 136.4 | **111.3** | 133.1 | **121.1** |
| cold_vadd, 2 ping-pong (vadd <-> vmul) | 135.3 | **108.9** | 131.2 | **116.8** |

A switch between two floorplans (new routes, DMA channels, locks and program) costs about as much as a cold
kernel: ~110–135 µs per launch on 32 cores, against ~2.2 ms for a switch of hw_contexts (`../vadd_vmul_bf16`,
`results/sweep.md`). A rerun is ~20–35 µs. `ping_pong.json` and `sequence.json` also pass.

Pressure, 32 cores (`results/pressure_32core.md`): about 840,000 launches on Linux (every kind in random legal
order, runlists of 1 to 1,000, both flows, 50 processes one after the other) and about 210,000 on Windows: 0 wrong
outputs, 0 words written past an output, 0 hangs.

## Windows (XRT)

As `../vadd_vmul_bf16/WINDOWS.md`: build the package on Linux (`./env.sh make precompile COLS=8`), copy
`precompiled/c8` over, and build `test`, `stress` and `check_sequence` with MSVC (`CMakeLists.txt`; a short build
path, CMake's try-compile tree passes 260 characters under a long work path):

```powershell
cmake -S . -B C:\b\pm -A x64 -DCOLS=8
cmake --build C:\b\pm --config Release
powershell -File tools/pressure.ps1 -Bin C:\b\pm\Release -Package precompiled/c8 -Group must -StopOnFail
```

Ryzen AI 7 350 (Krackan, 8 columns), XRT 2.21.0, 32 cores: the table above and the walks in
`results/pressure_32core.md` pass. Not run on Windows: P5b, P8, P9, P10 (`-Group hang`). They hang the NPU on
purpose, and repeated hangs can make the machine unresponsive (reboot to recover).

The builds above are blind (no context list). With `-DNPU_WATCH=ON` (public XRT headers only), `test` and `stress`
see other contexts through XRT's list, and warm and rerun launches run as asked while the NPU is theirs; they print
`sharing guard on` or why it is off. Details: `../vadd_vmul_bf16/WINDOWS.md`. In `host_test/checks/`: `list_probe`
prints and times that list as `NpuWatch` reads it; `ctx_probe` and `slot_check` (need XRT's source and boost,
`-DXRT_SRC_DIR`) print it through XRT's own headers and check `NpuWatch`'s vtable slots against them.

## What this example found

1. **fp32 on the cores is not always IEEE round to nearest.** 1.2% of sums and differences and 0.3% of products
   differ from the CPU (no rounding mode fits). A sum is within 1 ulp of the largest of its inputs and result (the
   adder drops the bits it shifts out of the smaller input), a product within 1 ulp. The check uses that bound per
   value (`designs/design.hpp`); a wrong program or a stale output is far outside it.
2. **A switch left the old app's shim mux bits and DMA Ctrl bits** (register model, `test_build` between
   different floorplans): aiecc sets the mux with mask writes, and Ctrl bit 0 on the channels it uses. Fixed in
   `launchlib`: the reset part now writes the mux and demux to 0 and each channel Ctrl to 0. The bf16 example
   still passes (`make run`, `make check`).
3. **Breaking the order contract cannot be repaired by any kernel here.** Between different floorplans, a warm
   kernel with the wrong prev hangs (the runlist fails after about a minute, then every runlist of that context);
   a rerun of the wrong app (for example after another hw_context put its xclbin back) leaves shim tasks that
   never finish. The AIE2P shim DMA has **no channel reset** (Ctrl bit 1 is Pause_Mem), so those tasks stay
   through every later kernel, cold ones too. Only a new hw_context clears them. Between apps of one floorplan
   (bf16 example) a wrong prev gave right data.
   **Now handled by `launch::Session`:** its order check refuses such a launch before it runs (stress S3), and a
   failed run gets a new hw_context (S4: a wrong prev with the check off fails at the timeout, then everything
   passes on the new context with the old buffers).
4. **A hang in one context fails the other context's runlist too** (P5b). Another context gives no sign when it
   takes the array, XRT's `access_mode::exclusive` does not keep it out, and it can open, run and close in ~3 ms.
   **The Session's guard** asks the driver for the other contexts before and after each run: while one exists, warm
   and rerun launches run as cold kernels (P10: a full walk next to a looping context and next to contexts that come
   and go, 0 wrong). Run as asked instead (`--policy allow`), runs next to another context fail or are flagged
   `other_context_ran`, and none came back ok with wrong data. Another context's commands do run **inside** our
   runlists, so every warm and rerun launch runs as cold while the NPU is shared, not only the first of a runlist.
   Not covered: a context that lives and dies inside one of our runlists.
5. **ELF flow limits.** Each ELF run keeps its own copy of its instructions in the process's 64 MB device heap (4 KB
   pages), and at most ~2,040 runs exist at once. The Session splits long runs into runlists of at most 24 MB and
   1,024 runs and uses the runs again: walks with runlists of 1..1000 pass in the ELF flow (P11). The xclbin flow
   has neither limit.
6. After another hw_context ran, cold kernels and warm kernels from the base xclbin's app (vadd) are right
   (stress S5).

## Files

| File | Content |
|---|---|
| `<app>_fp32/kernel.cc`, `<app>_fp32/<app>_fp32_placed.py` | the three designs |
| `Makefile`, `env.sh` | build, package, test, check, stress, precompile |
| `precompiled/c4`, `precompiled/c8` | packages for 4 and 8 columns (`make_package.py` output) |
| `host_test/test.cpp` | the AIE4 host test: list (`--apps` or JSON), `launch::Session` per flow, rows |
| `host_test/sequence.hpp` | the runtime sequences in C++ (`Tasks` = IRON's task API on `launch::Rt`), `vadd_rev` |
| `host_test/designs/` | inputs, buffers and golden results with their fp32 bounds |
| `host_test/checks/check_sequence.cpp` | C++ sequence == aiecc's `insts.bin` |
| `host_test/checks/ctx_probe.cpp` | XRT's list of hw_contexts, printed and timed (Windows CMake, needs XRT's source) |
| `host_test/checks/list_probe.cpp` | The same list as `NpuWatch` reads it with public XRT headers (`LAUNCH_NPU_WATCH_XRT`), printed and timed |
| `host_test/checks/slot_check.cpp` | `NpuWatch`'s vtable slots and entry offsets against XRT's headers (Windows CMake, needs XRT's source) |
| `host_test/*.json` | the three lists of AIE4 |
| `host_test/stress.cpp`, `tools/pressure.sh` | stress modes and the pressure run |
| `results/` | measured results |
