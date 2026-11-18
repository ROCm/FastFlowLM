<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Stress tests (control packets)

`host/stress.cpp` on the dev box (RyzenAI-npu5 Strix Halo, XRT 2.21.75, amdxdna 7.0.0-rc1), `Transport::ctrlpkt`
(the default). Command: `MODES=all ./env.sh tools/stress_all.sh` (arrays 1×1, 8×2, 8×4-pair; seeds 1, 2, 3; mode
`all` with 20,000 random launches).

## Mode `all` (S2, S3, S4, S8, S6, S1): 9 runs, all PASS

**192,051 launches in 5,628 runlists, 177,330 outputs checked: 0 unexpected wrong, 0 words written past the
end of a launch.** Launches by kind (all 9 runs):

| kind | launches |
|---|---|
| rerun | 59,783 |
| warm (full switch to the other app) | 44,373 |
| cold | 29,461 |
| pm (program-only switch to the other app) | 21,721 |
| warm self (full switch to the same app) | 14,984 |
| reset | 14,721 |
| pm self | 6,981 |
| corner cases (pm at pong, wrong prev) | 27 |

| case | 1×1 | 8×2 (16 cores) | 8×4 (32 cores) |
|---|---|---|---|
| kernels built (73), control packets | 29 ms, 111 KB | 88 ms, 136 KB | 113 ms, 369 KB |
| S1 random walk, 20,000 launches, runlists of 1–64 | PASS (3.0 s) | PASS (28 s) | PASS (53 s) |
| S2 runlists of 1, 23, 24, 25, 48, 200, 1000 launches | PASS | PASS | PASS |
| S3 odd length, then full switch | PASS | PASS | PASS |
| S3 odd length, then program-only switch | wrong data, no hang (expected) | same | same |
| S3 cold afterwards | PASS | PASS | PASS |
| S4 warm kernel with a wrong `prev` (same floorplan) | right data | right data | right data |
| S6 largest length, 1024 blocks per core | PASS (0.5 MB) | PASS (8.4 MB) | PASS (16.8 MB per buffer) |
| S8 special values: elements that differ from the CPU (of 4 launches) | 97, 91, 137 of 2,048: all denormals flushed to 0 | 2,343, 1,562, 1,574 of 32,768: all denormals | 4,663, 3,175, 4,669 of 65,536: all denormals |

S1's time is mostly the CPU checking every output; the launches themselves are in `sweep.md`.

Mode `ctx` (S5, another hw_context or process) shares the NPU, so it runs `Transport::txn` (control packets are
for this hw_context alone); what it shows is in the README ("The rule S5 adds").
