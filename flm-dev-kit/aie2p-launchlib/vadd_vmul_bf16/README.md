<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# vadd_vmul_bf16 — two programs, one hw_context, the launch library

The AIE2P (npu2) counterpart of `aie4-launchlib/vadd_vmul_bf16`: the same two kernels on the same cores compute

```
T1 = A + B     vadd
T2 = T1 * C    vmul
Y  = T2 + D    vadd

y = (A + B) * C + D, three kernels, one runlist, one hw_context
```

`lhs` and `rhs` are the two inputs of each kernel, `out = lhs (op) rhs` (A and B for the first vadd, T1 and C
for vmul, T2 and D for the second vadd); each has its own shim DMA channel per column.

The array is a build parameter: `COLS` columns (1–8) of `CORES` cores (1–4, up to 32 cores). Each column
streams its slice of every buffer through its mem tile, in one of two layouts (`LAYOUT`, `iron/`):

| layout | cores per column | the mem tile sends each core | mem-tile MM2S per column |
|---|---|---|---|
| `split` (default for 1–2) | 1–2 | its own lhs slice and rhs slice, two streams | 2·CORES + 1 |
| `pair` (default for 3–4) | 1–4 | `[lhs block | rhs block]` of the whole column, one broadcast stream; the core computes its own slice | 2 |

An AIE2P mem tile has 6 MM2S channels, so `split` stops at 2 cores per column; `pair` fits 4 (32 on the array) at
the price of CORES times the data on each core's stream. IRON cannot join and split in one link, so `pair`
broadcasts. The cores loop forever, as on AIE4, so the length `N` (a multiple of 256·COLS·CORES) is chosen at run
time.

There is no PM ping-pong: a kernel that changes program writes the program itself.

## The kernels (`host/host.cpp`)

| # | config | runs | the kernel writes |
|---|---|---|---|
| 0 | `cold("vadd")` | A+B | resets the whole array, loads vadd (program, locks, BDs, routes, chain starts), starts the cores |
| 1 | `warm("vadd", "vmul")` | T1·C | resets the cores and channels of vadd ∪ vmul, loads vmul with its program |
| 2 | `warm("vmul", "vadd")` | T2+D | the same, back |
| 3 | `warm("vadd", "vmul", {.program_only = true})` | T1·C | per core: reset, vmul's program, unreset, enable |
| 4 | `warm("vmul", "vadd", {.program_only = true})` | T2+D | the same, back |
| 5 | `rerun("vadd")` | | the runtime sequence only |
| 6 | `rerun("vmul")` | | (for the two-context baseline) |

```cpp
launch::ConfigList list{package, launch::Transport::ctrlpkt, {}};
list.configs = {launch::cold("vadd"), launch::warm("vadd", "vmul"), /* ... */};
auto stage = [n](launch::Rt &rt) { ew::stage(rt, n); };       // include/elementwise_sequence.hpp
launch::Result r = launch::build(list, {{"vadd", stage}, {"vmul", stage}});
// XRT: hw_context on r.base_xclbin, kernel "MLIR_AIE", run(3, instruction BO of r.kernels[i].words, size, A, B, C)
```

A warm kernel must follow a kernel that left its `prev` in the array (the runlist order). The program-only
kernels also need an even number of blocks per core per launch (ping/pong chains back at ping); `build()`
checks that vadd and vmul share a floorplan, the host checks the block count.

The host runs, in one hw_context: the expression (full and program-only switches), 1,600-launch loops of each
switch, reruns, the expression through FLM's ELF flow (`--elf`: an ELF per kernel, `txn::to_elf`), and the FLM way
(vadd.xclbin and vmul.xclbin in two hw_contexts). Every case is checked (one bf16 ulp; +0 = −0).

## Program-only switch vs full switch

Both kernels end the same way: the runtime sequence of the new app (3 shim BDs, their address patches, the pushes
and one wait per column). They differ in what they write before it.

```
full switch, warm(vadd, vmul)                       program-only switch, warm(vadd, vmul, {.program_only})

reset   stop the cores (Core_Control reset on)     program  per core:
        reset every started DMA channel                     stop it (reset on, enable off)
        set changed routes (none here)                      write vmul's program (PM)
        un-reset the channels                               unreset (PC 0), enable
load    vmul's program into PM
        every lock value of vmul   <- needed after the DMA reset
        every BD of vmul (core, mem tile)
        shim mux
        every chain start (Start_Queue + Ctrl)
        start the cores (unreset, enable)
sequence  shim BDs, patches, pushes, wait           sequence  the same
```

What each sends (`host.exe --build-only`, which prints `launch::breakdown`). With control packets the program,
locks, BDs and chain starts are one packet buffer the columns' shim DMAs send in parallel (rows with the same
writes get one broadcast); the command processor writes the rest (channel resets, the routes the packets take,
there and back, and the runtime sequence):

| kernel | | 1×1 | 8×2 (16 cores) | 8×4 (32 cores) |
|---|---|---|---|---|
| full switch | command-processor stream | 75 ops, 2,232 B | 777 ops, 22,280 B | 906 ops, 26,944 B |
| | packets | 2,232 B | 2,916 B | 6,784 B |
| | **measured switch cost** | **7.3 µs** | **64.3 µs** | **77.5 µs** |
| program-only switch | command-processor stream | 29 ops, 920 B | 257 ops, 7,912 B | 346 ops, 11,520 B |
| | packets | 1,140 B | 1,140 B | 4,560 B |
| | **measured switch cost** | **3.4 µs** | **16.3 µs** | **24.5 µs** |

Why the full switch writes all of that, and when the program-only one is enough:

| full switch writes | because | the program-only switch skips it, because |
|---|---|---|
| DMA channel reset | the old app's BD chains must stop before new BDs go in | vadd and vmul have the same BDs and chains: the running chains are already right |
| every lock | an idle S2MM channel holds the lock of its next BD; the reset drops it (no lock writes: the next launch hangs) | no channel is reset, so every lock still has its right value |
| BDs, chain starts, mux | another app has other BDs, channels and mux bits | equal in both apps (`build()` checks it: `floorplan_difference`) |
| routes (only those that differ) | another app has other routes | equal (here the full switch writes none either) |
| program | the cores must run the other code | the same |

The price of the program-only switch is two conditions: the two apps must share a floorplan (build time,
checked), and every launch before it must leave the ping/pong DMA chains at ping, i.e. an even number of blocks
per core per launch (run time, the host's job: a restarted core starts at ping). The full switch has neither
condition: it works between any two apps and after any length.

Both grow with the number of columns: what is left on the command processor (the channel resets and the
routes) is per column, while the packets of all columns go at once.

## Files

```
kernels/      vadd_bf16.cc, vmul_bf16.cc (16 bf16 lanes; the *_pair_bf16 functions for the pair layout),
              elementwise_config.h
iron/         vadd_vmul_bf16_aie.py: the design (npu2, --op add|mul --cols --cores --n --layout split|pair)
include/      elementwise_sequence.hpp: the C++ runtime sequence (any length); bf16.hpp: conversions, the check;
              spawn.hpp: the stress test's second process (fork/exec, CreateProcess)
host/         host.cpp: the kernels, checks, timing; stress.cpp: stress and corner cases
precompiled/  n256-c1x1, n256-c4x2, n256-c8x2, n256-c8x4-pair: the packages (make precompile)
tools/        sweep.sh, summarize.py: the results table; stress_all.sh: the stress runs;
              pressure.sh: the long runs on one array (default 32 cores)
results/      sweep.md, stress.md, pressure_32core.md
env.sh        the toolchain on PATH
CMakeLists.txt, WINDOWS.md   the Windows (MSVC, XRT) build of host and stress, and its results
```

## Building and running

```bash
cd aie2p-launchlib/vadd_vmul_bf16
./env.sh make                       # host; the package from precompiled/ (or aiecc + Peano)
./env.sh make run                   # 1 x 1, N = 512
./env.sh make run COLS=8 CORES=4 N=262144 RUN_ARGS=--elf    # 32 cores (pair layout)
./env.sh make check                 # the library's tests on this package (no NPU)
./env.sh make stress COLS=8 CORES=2 # stress and corner cases on the NPU
./env.sh make precompile COLS=4 CORES=1
./env.sh tools/sweep.sh             # the results table
./env.sh tools/stress_all.sh        # the stress table
LAUNCHES=1000000 ./env.sh tools/pressure.sh   # 32-core pressure runs (~22 min)
```

## Results

RyzenAI-npu5 (Strix Halo), XRT 2.21.75, amdxdna 7.0.0-rc1, `Transport::ctrlpkt`. µs per launch, median of 3 runs
of 50 runlists of 32 launches; all 57 runs passed every check. Switch cost = alternate − rerun (the same
buffers); 2 hw_contexts = the same two apps in two contexts, alternating, minus one context alone. Full table:
`results/sweep.md` (`./env.sh tools/sweep.sh`).

| array | cores | layout | blocks / core | N | program (all cores) | rerun | **switch, program only** | switch, full | **2 hw_contexts: switch** |
|---|---|---|---|---|---|---|---|---|---|
| 1×1 | 1 | split | 2 | 512 | 736 B | 19.8 | **3.4** | 7.3 | **142.5** |
| 1×2 | 2 | split | 2 | 1,024 | 1,472 B | 20.0 | **4.7** | 8.6 | **239.5** |
| 2×1 | 2 | split | 2 | 1,024 | 1,472 B | 21.1 | **3.3** | 13.4 | **257.6** |
| 4×1 | 4 | split | 2 | 2,048 | 2,944 B | 23.2 | **7.7** | 25.1 | **464.3** |
| 4×2 | 8 | split | 2 | 4,096 | 5,888 B | 23.8 | **8.9** | 30.8 | **814.5** |
| 8×1 | 8 | split | 2 | 4,096 | 5,888 B | 29.7 | **14.2** | 52.2 | **903.0** |
| 8×2 | 16 | split | 2 | 8,192 | 11,776 B | 29.8 | **16.3** | 64.3 | **1,584** |
| 1×1 | 1 | split | 32 | 8,192 | 736 B | 24.5 | **4.8** | 7.3 | **152.6** |
| 8×2 | 16 | split | 32 | 131,072 | 11,776 B | 38.4 | **17.4** | 65.6 | **1,583** |
| 8×2 | 16 | split | 128 | 524,288 | 11,776 B | 72.3 | **16.6** | 67.1 | **1,583** |
| 1×1 | 1 | split | 512 | 131,072 | 736 B | 90.4 | **8.6** | 11.8 | **162.3** |
| 1×1 | 1 | pair | 2 | 512 | 720 B | 19.7 | **3.6** | 7.2 | **136.5** |
| 1×4 | 4 | pair | 2 | 2,048 | 2,944 B | 20.0 | **6.1** | 10.8 | **307.7** |
| 2×4 | 8 | pair | 2 | 4,096 | 5,888 B | 22.2 | **7.6** | 21.1 | **579.6** |
| 4×4 | 16 | pair | 2 | 8,192 | 11,776 B | 23.6 | **14.7** | 38.8 | **1,124** |
| 8×2 | 16 | pair | 2 | 8,192 | 11,776 B | 30.6 | **16.4** | 57.3 | **1,262** |
| **8×4** | **32** | pair | 2 | 16,384 | 23,552 B | 29.9 | **24.5** | **77.5** | **2,196** |
| 8×4 | 32 | pair | 32 | 262,144 | 23,552 B | 48.6 | **25.5** | 78.6 | **2,200** |
| 8×4 | 32 | pair | 128 | 1,048,576 | 23,552 B | 120.8 | **23.7** | 75.8 | **2,202** |

- **The program-only switch** costs 3–5 µs on 1 core, ~16 µs on 16 cores and **24.5 µs on 32 cores (23 KB of
  program)**: 40× (1 core) to 90× (32 cores) cheaper than switching hw_contexts.
- **The full switch** (any two apps) costs 7.3 µs on 1 core and **77.5 µs on 32 cores**: 20× to 28× cheaper than
  switching hw_contexts.
- **Two hw_contexts** cost 2.2 ms per switch at 32 cores, as FLM measured with its 32-core designs (2.1–3.2 ms).
- **The layout changes the full switch, not the program-only one**: on 16 cores, pair has fewer mem-tile
  channels and BDs than split, so its full switch costs 57.3 µs instead of 64.3; the program-only switch costs
  the same (16.4 vs 16.3).
- **A rerun is the sequence only** (the cores loop): ~20 µs per launch at small N, the floor of a run in a
  runlist on this NPU.
- **The cold kernel** (whole-array reset and load) is the expensive one (one launch on its own, p50 on 32 cores:
  cold 202 µs, full switch 155, program-only 102, rerun 70; `results/pressure_32core.md`): start a runlist with
  it once, then stay warm.
- The switches do not depend on N; the large-N rows only add data movement.
- Windows (Krackan, `WINDOWS.md`): 8 × 4 program-only switch 26.2 µs, full 89.7, two hw_contexts 2,321.
- For comparison, AIE4 with the PM ping-pong (6 uCs, control packets): 7.6 µs (1×1), 9.0 µs (3×4).

## Stress and corner cases (`host/stress.cpp`)

`MODES=all ./env.sh tools/stress_all.sh`, control packets: 1×1, 8×2, 8×4 (32 cores), seeds 1–3. **192,051
launches, 177,330 outputs checked: 0 unexpected wrong, 0 words written past the end of a launch.** Details:
`results/stress.md`.

| case | what it does | result (all three arrays, control packets unless marked txn) |
|---|---|---|
| S1 random walk | 20,000 launches per run in runlists of 1–64: every kind (cold, full, program-only, self-switches, rerun, reset), five lengths, random inputs; it only picks legal kernels (it knows the app in the array and the ping/pong position) | PASS |
| S2 runlist length | 1, 23, 24, 25, 48, 200, 1000 launches (XRT cuts runlists into chains of 24) | PASS |
| S3 odd length | odd, then a full switch | PASS |
| | odd, then a program-only switch | wrong data, **no hang** (expected: the chains are at pong); cold recovers |
| S4 wrong `prev` | a warm kernel whose `prev` is not in the array (same floorplan) | right data |
| S5 another context (`--mode ctx`, txn) | a launch in another hw_context of this process, or another process, then rerun | **wrong: the array is the xclbin again** (see below); `warm(vadd, x)` and cold recover |
| S5b concurrent process (txn) | another process loops on the NPU meanwhile | full switches right; reruns and program-only switches partly wrong |
| S6 largest length | 1024 blocks per core (16.8 MB per buffer on 32 cores), every kind | PASS |
| S8 special values | ±0, ±inf, NaN, denormals, extremes | only denormals differ (flushed to 0 by the NPU) |
| S9 library errors | `make -C ../launchlib test` (`test_errors`): bad Rt calls, bad lists and JSON, a sequence that throws, 11 kinds of broken package | each one throws and names the configuration, file or op |

**Pressure on 32 cores**, control packets (`./env.sh tools/pressure.sh`, `results/pressure_32core.md`):
3 × 200,000 random launches, runlists up to 1000, lengths up to 16.8 MB per buffer, 50 processes coming and
going — about 740,000 launches, **0 wrong, 0 hangs**. Latency of one launch on its own (p50 / p99 / max, µs):
rerun 70 / 133 / 299, program-only switch 102 / 292 / 331, full switch 155 / 347 / 368, cold 202 / 357 / 394.
Windows (Krackan): the same parts, 0 wrong (`WINDOWS.md`). The two-process parts (P5, P7) run
txn (the rule below).

**The rule S5 adds:** the state a kernel leaves in the array lives only as long as this hw_context is the only
one using the NPU. When the NPU runs another hw_context (in this process or another one) and comes back, this
context's array is its xclbin again (here: vadd, the base xclbin) — and that can happen between any two
launches, also inside one runlist. Reruns and program-only switches then compute the wrong program. Full
switches between apps of one floorplan are not affected. So: start every runlist with `cold` (or a warm kernel
from the base xclbin's app) after anything else may have used the NPU, and do not rely on reruns or program-only
switches while another context is active.

## Findings

1. **vadd and vmul share a floorplan**: aiecc gives both the same BDs, locks, chain starts and routes; only the
   program differs (checked by `build()` for every program-only kernel).
2. **A DMA channel reset needs the lock writes after it** (an idle S2MM channel holds a lock; without the
   writes the next launch hangs). The full switch writes every lock of the app.
3. **A reset mask write on a Start_Queue pushes a BD.** A first version picked the queue register 4 bytes
   before the first MM2S Ctrl as a Ctrl (wrong data, then a hang); the library's `channel_ctrl_of` is tested.
4. **Stopping cores that wait on a lock works, but their acquire stays pending**: every warm kernel stops
   cores that wait (cores loop forever), more than 100,000 times in the stress runs here without a wrong
   result. The Whisper encoder showed what this example did not: the pending acquire is granted by the next
   app's lock writes and takes one of its tokens (`../whisper_encoder/README.md`, step 2). The reset part now
   drains the core-tile locks after the DMA reset (63, then 0).
5. **Another hw_context resets this context's array to its xclbin**, also between launches of one runlist (S5).
6. **The NPU flushes denormals to 0** and gives +0 where the CPU gives −0 (0 × a negative number).

## Limits

- `split`: 1–2 cores per column; `pair`: up to 4 (32 cores), with CORES times the data on each core's stream.
- The kernels' state is lost when another hw_context runs (S5).
- Packages are built on Linux; Windows builds and runs the host programs (WINDOWS.md: same bits, same
  switch costs). The Session's sharing guard sees other contexts on Linux, and on Windows only when built
  with `LAUNCH_NPU_WATCH_XRT=1` (WINDOWS.md).
- `Transport::ctrlpkt` (the default) is for this hw_context alone on the NPU; the stress modes that share the
  NPU (`ctx`, `churn`, `guard`) run `Transport::txn` unless given `--hang-risk` (launchlib README section 4.1).
