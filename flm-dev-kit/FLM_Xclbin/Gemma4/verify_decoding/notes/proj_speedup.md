<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Stream P: short-context speed of the 8-row verify layer (2026-10-29)

All measurements: hardware, real Bolt weights. Private build tree `FLM_Xclbin/Gemma4/vd_pw` (git-excluded copy of verify_decoding;
committed kernels are copied back to verify_decoding). Scripts: work/scripts/pw_*.sh. "min" = min over >= 100 dispatches (the NPU is
shared, medians are often contention-inflated; quiet-box min == median).

## Committed
| # | change | before | after | accuracy |
|---|---|---|---|---|
| 1 | q4_l8: mmul pair scoped to one row half (all four 8x8 accumulators live -> compiler spilled them every k-tile) | toy 1.92 us/block (Peano slope), projonly L5 min 1724-1732 us | toy 1.62 us/block, projonly L5 min 1510-1549 us (L0 1346-1388) | bit-identical (proj8 relL2 0.0143-0.0202 unchanged; fused8 oracle unchanged) |
| 2 | PLE MVM: two tokens per 64-lane accumulator, all 8 tokens one pass, column loop unrolled | toy 1.10 us per [128k x 32] x 8 tok | 0.91 us | bit-identical (ple8 stage relL2 0.0063-0.0091 unchanged, 4 layers) |

Reference points (same session): projonly L=1 decode core L5 min 1271-1289 us; projonly 8-token with compute removed (stream floor) min 1427-1453 us.
-> the 8-token projection is now within ~5-8 % of its weight-stream floor (was ~20 %).

## Whole layer (fused8, hammer8 min latency, cursor 64)
| build | L0 swa | L5 glob | L35 glob-skip | L40 swa-skip |
|---|---|---|---|---|
| base (verify_decoding fused8 before P), full | 3252-3346 | 3691-3702 | 3523-3528 | 3252-3256 |
| p1 (both commits), nople | 2662-2712 (base nople 2823-2899) | 3069 (3291) | 2906 (3110) | 2662 (2823) |
| p1, full | 3098 (quiet, 2-layer run) | 3408 (quiet) | contended | contended |
| p1 oracle (verify8 vs 8 sequential decodes) | PASS 11 cursors; cursors >=505 "DIFF" = the known first-dispatch-after-load 1-ulp SWA effect, present in the base build too | PASS | PASS | PASS |
hammer8 p1 full: 5000 dispatches, 8 (layer,cursor) pairs, 0 diffs, no hang.

## Experiments (not adopted)
- int8 x uint4 native mmul<4,16,16> (v6): mmul itself fast, but x quantisation (int8 per token / 32-col group) costs 0.6-1.7 % rel L2 (worse than bf16 path on some tensors) and the epilogue (int32->float, per-group scale) made it 4344 cycles/block (> v4 3384). Dropped.
- int16 x uint8 integer core (v9, exact weights): 7113 cycles/block. Dropped.
- native bfp16 mmul (v10, v13): no gain (3271 vs 3384 cycles); A conversion costs what it saves.
- nibble -> bf16 via 0x4300 bit trick (v12): no gain.
- Peano in-kernel timer: read_tm(0x340F8) crashed, read_tm(0x140F8) hung the core (30 s timeout). Use Chess get_cycles for cycle counts.
- fused8 'attn' debug stage hangs on dispatch #1 via hammer8 (known flaky debug build; do not run).

## Final fused8 with both committed kernels (build `vd_pw/build/fused8_full_L8_pf`, copy: work/pstudy/xclbins/verify_P.xclbin, RTP map identical to the engine's verify.xclbin)
Quiet box, hammer8 min / median per dispatch, same session back to back:
| layer | cursor | base fused8 (min) | P fused8 min / median |
|---|---|---|---|
| L0 swa | 64 / 505 | 3310 / 4016 | 3105 / 3800 (3147 / 3842) |
| L5 glob | 64 / 505 | 3694 / 4726 | 3430 / 4490 (3496 / 4539) |
| L35 glob-skip | 64 / 505 | 3530 / 4571 | 3292 / 4330 (3336 / 4378) |
| L40 swa-skip | 64 / 505 | 3236 / 3947 | 3020 / 3730 (3088 / 3770) |
-> 210-265 us (6-7 %) less per layer.  42-layer estimate at short ctx (20 swa + 15 swa-skip + 4 glob + 3 glob-skip, hammer mins):
base 140.1 ms -> P 131.0 ms (the measured base chain8 was 147 ms incl. host overhead, so ~138 ms expected).
- verify8 vs oracle, layers 0/5/40/35, 11 cursors 0..1000, repeat 3: every cosine / ulp number identical to the base build (bit-identical
  outputs); the 4 "FAIL"s are the repeat DIFF of SWA L0 at cursors >= 505, the first-dispatch-after-load effect of the base build (same in base).
- hammer8 P: 5000 dispatches, 8 (layer, cursor) pairs incl. cursor 505, 0 diffs, no hang (token-balanced stream of stream H).
- chain8 (42 layers, 6 seqs x 8 rows): argmax 42/48, min cos 0.96677 -- identical to the base build; timing NOT valid (another fork's chain8
  and the D fork's engine test ran at the same time: base 382 ms, P 516 ms under contention).
- Engine tok/s not measured: the D fork held the NPU with engine tests (only one engine test at a time).

## PLE split (gate and up each over two tiles): NOT adopted
Design (patches in work/pstudy/ple_split/): G (4,2) + G2 (4,3) writing its half of u into G's memory (locks 6/7 of G), U (5,2) + U2 (5,4),
second weight halves on IT4.MM2S1 (packets 29/30), MT5 ple stage with two landings.  Findings:
- raw gate outputs of the split are bit-exact vs non-split; u = g * pli (no activation) is deterministic and NaN-free;
- the LUT activation (getActivationBf16) in G returns zeros / NaN whenever G2 exists, even with the activation in its own noinline function
  and the tables verified intact in G's memory (dumped: ab and cd 512/512).  Not root-caused (suspect: the LUT gather 4x64 loads vs the
  north neighbour's memory interface).  Also: a call (not inlined) between the gate rows and the activation broke it differently (Peano).
- expected gain was only ~0.15-0.2 ms per layer; stopped here.
