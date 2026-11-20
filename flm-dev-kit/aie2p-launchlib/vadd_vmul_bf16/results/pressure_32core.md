<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Pressure tests on 32 cores (8 × 4, pair layout), control packets

`./env.sh tools/pressure.sh` (P1-P4, P6) on the dev box (RyzenAI-npu5 Strix Halo, XRT 2.21.75, amdxdna
7.0.0-rc1), package `precompiled/n256-c8x4-pair`, `Transport::ctrlpkt` (the default). About 5 minutes.

**About 740,000 launches, every output checked against the CPU: 0 wrong, 0 words written past the end of a
launch, 0 hangs or timeouts.**

| part | what | launches | outputs checked | time | result |
|---|---|---|---|---|---|
| P1 soak, seed 11 | random walk over every kind (cold, full switch, program-only switch, self-switches, rerun, reset), lengths 1–8 blocks per core, runlists of 1–64 | 200,000 | 184,517 | 49 s | PASS |
| P1 soak, seed 12 | the same | 200,000 | 184,522 | 49 s | PASS |
| P1 soak, seed 13 | the same | 200,000 | 184,619 | 49 s | PASS |
| P2 long runlists | runlists of 1–1000 launches | 100,000 | 92,240 | 26 s | PASS |
| P3 big lengths | lengths 2, 64, 256, 1024 blocks per core (up to 16.8 MB per buffer), runlists of 1–16 | 4,000 | 3,704 | 63 s | PASS |
| P4 latency | 2,000 one-launch runlists per kind | 10,000 | per kind | 2 s | PASS (below) |
| P6 process churn | 50 processes: start, build the kernels, 500 launches, exit | 25,000 | | 13 s | PASS, 50 of 50 |

P1 runs at 246 µs per launch with the CPU checks (the NPU part is in `sweep.md`).

## P4: latency of one launch (execute + wait, one at a time), µs

| kind | mean | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|
| rerun | 76.8 | 70.1 | 99.7 | 132.8 | 294.8 | 298.7 |
| program-only switch | 109.9 | 101.5 | 124.4 | 291.7 | 319.9 | 330.5 |
| full switch | 173.0 | 154.6 | 208.9 | 347.4 | 365.2 | 368.3 |
| cold | 215.3 | 201.7 | 254.8 | 356.9 | 389.9 | 393.7 |
| reset | 231.1 | 213.3 | 269.3 | 402.4 | 407.3 | 407.4 |

A one-at-a-time launch pays the host round trip (~50 µs more than inside a runlist). No launch took more than
0.41 ms.

## Not run with control packets

P5 (two processes at once) and P7 (50 hw_contexts opened and closed in one process) share the NPU with other
hw_contexts, so `pressure.sh` runs them with `Transport::txn` (launchlib README section 4.1). What they show
(the one-context contract) is in the README, S5.
