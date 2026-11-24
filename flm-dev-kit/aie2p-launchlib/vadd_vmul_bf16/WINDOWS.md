<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# vadd_vmul_bf16 on Windows (XRT)

The packages (xclbin, instructions, manifest) are built on Linux with
`make precompile` and copied over; Windows builds only the two host programs
(`CMakeLists.txt`, MSVC, static CRT like FLM_DLL) and runs them.

```powershell
cd aie2p-launchlib\vadd_vmul_bf16
# precompiled\n256-c8x4-pair copied from a Linux build (./env.sh make precompile COLS=8 CORES=4)
cmake -S . -B build\c8x4 -A x64 -DCOLS=8 -DCORES=4
cmake --build build\c8x4 --config Release
build\c8x4\Release\host.exe   --package precompiled\n256-c8x4-pair --n 16384 --iters 50 --pairs 16
build\c8x4\Release\stress.exe --package precompiled\n256-c8x4-pair --mode all
```

CMake defaults come from `FLM_DLL/cmake/flm_xrt_win.cmake` (as FLM_DLL): XRT headers in
`C:/dev/XRT/src/runtime_src/core/include`, `xrt_coreutil.lib` in `C:/dev/xrtNPUfromDLL`;
override with `-DFLM_WIN_DEPS_ROOT=`, or `-DXRT_INCLUDE_DIR=` / `-DXRT_LIB_DIR=`. `--preempt` (host and stress) puts a
preemption point at the start of each sequence, as FLM does on Windows.

Machine: Ryzen AI 7 350 (Krackan, 8 columns), Windows 11 26200, NPU driver
32.0.20102.3930, firmware 1.1.2.64, XRT 2.21.0 (the `dll` CI runner; run
44178702152 of `launchlib-win.yml`). Linux: npu5 (Strix Halo), the same
packages and commands. Everything below is `Transport::ctrlpkt`.

## Bits

`host` prints an FNV hash of the result after each check; inputs come straight
from mt19937, so both platforms compute the same numbers. 8 × 4 (pair):
`7bec9d6398ec8050` on Linux and Windows, in every check (full switch,
program-only switch, after rerun, the ELF flow, two contexts).

## Time, µs (8 × 4, 32 cores)

| what | Linux | Windows |
|---|---|---|
| rerun, per launch in a runlist | 29.9 | 28.8 |
| program-only switch (cost beyond a rerun) | 24.5 | 26.2 |
| full switch (cost) | 77.5 | 89.7 |
| two hw_contexts (cost) | 2,196 | 2,321 |
| one-launch runlist, p50: rerun / program-only / full / cold | 70 / 102 / 155 / 202 | 94 / 128 / 181 / 231 |

Linux: median of 3 runs (`results/sweep.md`); Windows: one run (a second run
of the same job: 26.0, 88.0, 2,478). On Windows a program-only switch costs
1 % of a hw_context switch, a full switch 4 %.

## Stress and pressure (8 × 4, 32 cores)

| run | launches | result |
|---|---|---|
| `stress --mode lengths`, `special`, `big` | | PASS |
| `pressure.ps1`, P1: 3 random walks | 3 × 200,000 | 0 wrong |
| P2: runlists up to 1,000 | 100,000 | 0 wrong |
| P3: lengths up to 16.8 MB per buffer | 4,000 | 0 wrong |
| P4: one-launch latency | 2,000 per kind | 0 wrong |
| P6: 50 processes one after the other | 50 × 500 | 0 failed |

The parts with two processes on the NPU at once (P5, P7) run `Transport::txn`
(`tools/pressure.ps1`), see the launchlib README section 4.1.

## Sharing guard (context list from XRT)

The driver has no Linux-style ioctl here, but XRT keeps a list of every
hw_context on the NPU (`xrt_core::device_query<query::aie_partition_info>`:
pid, process name, context id, columns, command submissions). It is internal
to XRT (no public API, no stable layout), so `NpuWatch` reads it only when
built with `LAUNCH_NPU_WATCH_XRT=1`, with XRT's public headers only (no XRT
source, no boost):

```powershell
cmake -S ..\pm_runtime_load_singletile -B C:\b\watch -A x64 -DCOLS=8 -DNPU_WATCH=ON
cmake --build C:\b\watch --config Release --target stress test
C:\b\watch\Release\stress.exe --package precompiled/c8 --mode watch --launches 15000
```

It calls XRT's query through a copy of the two vtable classes it needs, finds
the query by the RTTI name of its request type (the key number moves between
XRT builds), and reads each entry by byte offset, the entry size (200, 248 or
264 bytes) taken from the length of the list. At run time it checks XRT 2.19
or later, an ABI it knows (MSVC without iterator debugging, or libstdc++) and
that every entry is plausible; else the Session is blind as before: every
warm and rerun launch runs as cold unless `exclusive`. `Session::watch_why()`
says why. A query that throws is tried up to 3 times (about 0.6% throw while
other processes open and close contexts); `NpuWatch::last_error()` keeps the
reason. `slot_check` (needs XRT's source, dev check) compares the vtable slots
with XRT's headers; `list_probe` prints the list as `NpuWatch` reads it.

`max_gap` is 1 ms on Windows (2 ms elsewhere): a context that opens, runs and
closes faster than that can pass unseen; more than `max_gap` since the last
query, the next warm or rerun launch runs as cold. A query costs 35 µs on
Windows with one other context. pm `stress --mode watch` (S7) checks the
guard: next to a process that runs cold(vmul), every warm and rerun launch of
the walk runs as cold, 0 wrong, on Linux and on Windows. Not tested on
Windows: a context that runs a launch inside one of our runlists (on a shared
machine a hang there can make the machine unresponsive; reboot to recover).

## Port notes

- `include/spawn.hpp`: the stress test's second process (CreateProcess on
  Windows, fork/exec on Linux).
- Without the XRT list above, a Windows Session cannot see other contexts. It
  then takes the NPU as shared: with the default `SharedPolicy::cold_only`
  every warm and rerun launch runs as its cold kernel.
  `SessionOptions::exclusive = true` (the caller vouches that no other process
  uses the NPU) runs them as asked. A warm or rerun launch after another app's
  context hangs the NPU, and on Windows repeated hangs can make the machine
  unresponsive (reboot to recover). So a
  Windows Session also stops after `max_failures` (2) failed runlists in a row
  instead of opening a new hw_context each time (status `stopped`, `reopen()`
  to go on). Pressure results of the Session on Windows:
  `../pm_runtime_load_singletile/README.md`.
- aiebu: FastFlowLM's `src/lib/xrt/aiebu_static.lib` is built with /MT; link with
  the static CRT.
- `xrt-smi examine --report preemption` does not exist in XRT 2.21 on Windows.
