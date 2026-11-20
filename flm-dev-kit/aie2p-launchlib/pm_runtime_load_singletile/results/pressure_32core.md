<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Three apps, 32 cores: pressure results (control packets)

npu5 (Strix Halo dev box), XRT 2.21.75, amdxdna 7.0.0-rc1, `COLS=8` (8 columns × 4 cores), `Transport::ctrlpkt`
(the default). Everything runs through `launch::Session` (`../launchlib/include/launch/xrt/session.hpp`), default
options: order check on, sharing guard on, ELF runlists of at most 24 MB / 1,024 runs. Parts P0-P4, P6 and P11 of
`./env.sh tools/pressure.sh`; about 2 minutes.

| Part | What | Launches | Result |
|---|---|---|---|
| P0 | `test.exe`, every row, xclbin and ELF flow, 1,000 launches per row | 18,012 | 18 rows PASS |
| P1 | random walk over every kind, 3 seeds × 200,000 launches, runlists of 1..64 | 600,000 | PASS, 0 wrong (147 µs per launch, checks included) |
| P2 | random walk, runlists of 1..1000 | 100,000 | PASS, 0 wrong |
| P3 | runlists of 1, 23, 24, 25, 48, 200, 1000, the order check, then a 20,000-launch walk | 21,324 | PASS, 0 wrong |
| P4 | latency, 2,000 one-launch runs per kind, guard on and off | 28,000 | PASS (below) |
| P6 | 50 processes: start, build, 500 launches, exit | 25,000 | 0 failed |
| P11 | ELF flow walk, runlists of 1..1000 (111 runs) | 50,000 | PASS, 0 wrong |

**About 840,000 launches, every output checked against the CPU: 0 wrong, 0 words written past an output, 0
hangs; no launch ran as cold, no new hw_context.** The walks run all nine warm switches (each app to the two
others and to itself), reruns (vminus's with its RTP and lock writes), colds and resets.

Windows (Krackan, XRT 2.21.0, CI run 44178702152): every `test` row in both flows, `stress --mode all` 20,000
launches per flow, random walks 3 × 30,000, 50,000 with runlists up to 1000, 30,000 in the ELF flow, the order
check (`warm_first`): about 210,000 launches, all PASS, 0 wrong.

## P4 latency (µs per launch, execute + wait of a one-launch run)

| Kind | guard on: mean / p50 / p99 | guard off: mean / p50 / p99 |
|---|---|---|
| rerun vadd | 155.1 / 153.6 / 336.3 | 109.9 / 97.3 / 277.4 |
| vminus: warm, rerun, warm back (RTP + lock in the rerun) | 233.1 / 228.3 / 444.8 | 209.8 / 217.3 / 397.5 |
| warm vadd <-> vmul | 259.5 / 239.8 / 477.2 | 218.1 / 199.4 / 355.4 |
| warm cycle vadd -> vmul -> vminus -> vadd | 263.8 / 249.6 / 512.3 | 215.9 / 204.7 / 362.8 |
| warm vadd -> vadd (to itself) | 252.7 / 227.8 / 473.3 | 220.1 / 209.8 / 385.9 |
| cold vadd | 294.9 / 260.5 / 495.0 | 240.5 / 217.3 / 401.2 |
| reset + cold vadd (2 launches) | 321.0 / 306.0 / 497.6 | 252.3 / 235.3 / 409.8 |

The guard asks the driver for the other hw_contexts around each run (35–55 µs per query once any hw_context
exists; the driver asks the firmware): +40–55 µs per one-launch run here. In a runlist of N launches it is paid
once. In runlists (P0, 1,000 launches) the same kinds cost far less per launch: rerun 16–52 µs, a warm switch
between floorplans 129–143 µs, cold 146–183 µs.

## Not run with control packets

P5 (two processes at once), P7 (50 Sessions in one process), P8 (other contexts, S5), P9 (recovery from a hang,
S4) and P10 (the guard next to other contexts) share the NPU or hang it on purpose: their stress modes run
`Transport::txn` (launchlib README section 4.1), and the hang parts are not run on the shared dev boxes. What they
show is in the README ("What this example found").

## Driver facts the Session relies on

| Fact | Detail |
|---|---|
| Other hw_contexts are visible | `DRM_IOCTL_AMDXDNA_GET_ARRAY` (`HW_CONTEXT_ALL`) lists every context of every process, with pid and submitted commands. Context ids are per process and used again. |
| No counter shows the takeover | `preemptions`, `suspensions`, `migrations` of our context stay 0 when another context runs on its columns. |
| `access_mode::exclusive` is ignored | With our context exclusive, other processes still open contexts (shared or exclusive) and run. |
| A context can be short | open + one launch + close: ~3 ms. |
| ELF runs | each holds its instructions in the 64 MB device heap (4 KB pages) and at most ~2,040 runs exist at once. Freed with the run; a run can be in the next runlist again (not twice in one). |
| Buffers survive a new hw_context | made before `reopen`, they work after (S4). |
