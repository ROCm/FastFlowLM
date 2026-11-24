<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Stream A: long-context 8-token verify attention (attnx)

Goal: the 8 verify tokens read K/V ONCE per verify (attn8 replayed K/V once per token: 8x the single-token traffic).
Target: global and SWA 8-token attention <= ~2x the single-token (decode) attention at 4k.  Status: **met** (stage design and fused8).

## Design
Files: `kernels/attnx_common.h`, `kernels/attnx_score.cc`, `kernels/attnx_pv.cc`, `iron/attnx.py` (tiles, q park, o assembly),
`iron/attnx_aie.py` (stand-alone stage design, DESIGN=attnx), `host_attn8/attnx_host.hpp` (stage host, `attn8.exe --stage attnx`),
`iron/fused8_aie.py` (`FUSED8_ATTN=x` selects attnx, default `8` = old attention), `host/test_fused8.hpp` (attnx RTPs + K/V replay).

- K/V streamed from DDR once per verify, head-major, in 4 KB sub-chunks (global 8 keys x 512 dims; SWA 16 keys x 256), linear landing.
  K is multicast to 2 score tiles, V to 4 PV tiles.
- **Score tiles** (fused8: CT(1,4), CT(3,4)): each takes 4 of the 8 query heads x 8 tokens (32 rows) from the q park (MT4).
  - S = Q.K^T with bfp16 mmul.
  - Causal/window masks come from RTP row tables.
  - Online softmax with a **lazy running max**: the max moves only if a score exceeds it by > 8, so the rescale factor c = 1 almost always and the
    per-message flag lets PV skip the fp32 rescale (fp32 x fp32 is emulated on AIE2P: this alone was 3x).
  - P (bf16) + c go to the PV tiles.
- **PV tiles** (column 2, the decode attention tiles): y += P.V^T (bfp16 mmul), y f32 32 KB in L1, l row sums.
  - End: o = y / l to bf16 in place.
  - o goes over circuits to the o assembly memtile MT2, which emits 8 token rows to the layer's o landing.
  - SWA: PV 0/2 do the work, 1/3 drain V.
- Routing lessons: no multi-source packet merges (mlir-aie multicasts duplicate ids / mask-merged rules capture other ids -> hangs); circuits for o.
- Stack: the score tile needs 8 KB stack (a 4 KB stack silently overflowed into Q block 0: head-0 rows wrong).

Engine contract:
- K/V cache rows past the context, rounded up to 32 (global) / 16 (SWA) rows, must be finite (engine keeps them zero).
- V must not hold big values in an 8-row block that contains a valid key.
- RTP: score [n_sub, hi0, u0, is_swa, tb]; PV [n_sub, is_swa, j].

## Accuracy (stage, vs fp32 reference, poison on, repeat 3, all cursors in ONE hardware context: work/attnx/accept_v1.log)
- Global L5 / L35, c in {0,1,7,8,9,14,15,16,30,40,56,500,505,511,512,520,1000,2000,4000}: PASS (min cos 0.979 at 4000, uniform inputs).
  - c=8000/16000 with near-uniform logits: cos 0.963 / 0.929, below the 0.975 tolerance; **attn8 on the identical inputs: 0.9626 / 0.9291** (bf16 P over
    many near-equal keys; not an attnx defect).
  - attnx vs attn8 output, same inputs: cos >= 0.99949 (L5 8000, L5 16000 peaked, L0 4000 peaked).
- SWA L0 / L40, c in {0..56, 500,505,510..513,520,527,528,1000,1040,4000}: PASS.
- Peaked attention (q_norm x6):
  - L5 1000/4000/16000: 0.9984/0.9963/0.9957 (attn8 16000: 0.9962).
  - SWA L0 4000: 0.9903 (attn8 0.9880).
- Sequence c=0 then c=9 with poison: PASS.

## Latency, stage design (min of 30, quiet NPU)
| ctx | attnx global | attn8 global | attnx SWA | attn8 SWA |
|---|---|---|---|---|
| 0 | 0.26 ms | 0.31 | 0.20 | 0.21 |
| 1000 | 0.57 | 2.72 | 0.34 | 2.35 |
| 4000 | 1.53 | 9.83 | 0.32 | 2.35 |
| 16000 | 5.24 | 38.1 | (flat) | |
- Global slope: 0.31 us/key for all 8 tokens. The decode single-token attention slope is 0.23 us/key, so attnx costs **1.34x one token** (attn8 cost 2.3 us/key = 10x).
- At 4k, above the ctx-0 base: attnx 1.28 ms vs decode ~0.92 ms = 1.39x.
- SWA is flat at 0.32 ms (attn8 2.35 ms).

## Soak / determinism
Stage design, one hardware context, poison:
- Global L5 c=9/4000: 2 x 1000 repeats + 5000 soak = 7000 dispatches.
- SWA L0 c=9/505/1000: 8000 dispatches.
- L35 c=4000: 4000 dispatches.
- **0 diffs** (bit-identical to the first dispatch).

fused8x (`build/fused8x_full_L8`):
- hammer8, layers 0/5/40/35 x cursors 0/505/1000/4000: **6000/6000 dispatches, 0 diffs, 0 first-after-load diffs** (work/attnx/hammer_fused8x.log).
- The token ledger in f8::run is balanced (one wait per token-issuing BD; o lands with the layer's existing o S2MM tokens).

## Hangs caused by this stream (all before the fixes above; none since)
- 10:04, 10:10: `attn8.exe --stage attnx`, packet routing: same id from 4 PV tiles multicast + mask-merged rules misrouting.
  Fixed by the o assembly memtile with circuits.
- ~11:08: an ATTNX_DBG=1 debug build (debug branches broken; DBG builds must not be used again).

## Read-before-write bug (coordinator item 3): ROOT CAUSE = test harness, fixed
Repro: `-t verify8 --mode verify --layers 5 --cursors 0,9 --poison` (cos 0.83 at c=9 when c=0 runs first).

Cause: `kv_cache` is `buffer<bf16>`, and `size()` returns ELEMENTS.
- The snapshot/restore (`snap.assign(data, data + size())`, `memcpy(.., snap.size())`) covered only the first half of the bytes, i.e. the K plane.
- The V plane was never restored, so the c=0 V poison (1e4 from row 8) stayed in rows 8..15, which are valid keys at c=9.
- The same bug made later repeats/dispatches see a different V than the first one: that was the "first dispatch after load differs by 1 ulp on SWA
  non-skip layers with > 32 chunks" (also H's 3 first-after-load diffs in hammer8).

Fixed in test_verify8.hpp (snapshot, skip-layer copy) and test_fused8.hpp (chain + hammer8 snapshots).
After the fix:
- 0,9 poison PASSES on BOTH the old fused8 (0.99935 / 0.99951) and fused8x (0.99934 / 0.99956).
- SWA L0 505/1000 x4 repeats are `same` on both.
- hammer8 shows 0 first-after-load diffs.

Not fixed (not my file): the copy in `Gemma4/vd_pw/host/test_{verify8,fused8}.hpp` has the same bug.
No on-chip attention state was involved.

## fused8 integration (FUSED8_ATTN=x, build: `[ATT=x|8] scripts/build_fused8x.sh full` -> build/fused8x_full_L8 / build/fused88_full_L8)
- Oracle (`run_fused8.sh --maxl 4608`, layers 0/5/40/35 x {0,1,7,8,9,14,15,16,30,40,56,500,505,511,512,520,527,528,1000,2000,4000}, poison, repeat 3):
  **84/84 PASS, all repeats same** (work/attnx/fused8x_accept.log).
- Global layers at 8000 / 16000 (`--maxl 16400`, poison, repeat 3): L5 0.99976 / 0.99975, L35 0.99884 / 0.99879, all same: PASS.
- hammer8 soak: 6000/6000 dispatches, 0 diffs (above).
- hammer8 latency, min ms of one layer dispatch incl. launch, quiet NPU (11:36).
  - "x" = fused8x.
  - "8" = fused8 with the attn8 attention built from the SAME tree (`ATT=8 build_fused8x.sh full`, i.e. same proj/PLE kernels).
  | layer | ctx 0 | 1000 | 4000 | 16000 |
  |---|---|---|---|---|
  | L5 global, x | 3.36 | 3.65 | **4.54** | **8.17** |
  | L5 global, 8 | 3.42 | 5.73 | 12.78 | 41.10 |
  | L35 global-skip, x | 3.20 | 3.50 | 4.43 | 8.05 |
  | L35 global-skip, 8 | 3.21 | 5.55 | 12.63 | 40.93 |
  | L0 SWA, x | 3.05 | 3.19 | 3.19 | 3.17 |
  | L0 SWA, 8 | 3.22 | 3.82 | 3.83 | 3.83 |
  | L40 SWA-skip, x | 2.99 | 3.12 | 3.10 | 3.06 |
  | L40 SWA-skip, 8 | 3.21 | 3.75 | 3.73 | 3.73 |
  - A first measurement at 11:34 showed fused8x at 5.7-5.9 ms at ctx 0: NPU contention from another fork (the medians of both builds were bimodal then).
  - Global layer at 4k: 1.49x a decode step (3.05 ms @1k decode); at 16k it is 2.7x (the decode step itself also grows: 0.23 us/key).
  - 42-layer estimate (35 SWA + 7 global) at 4k: x ~ 35 x 3.15 + 7 x 4.5 = 142 ms vs 8: 35 x 3.8 + 7 x 12.7 = 222 ms; at 16k: 167 vs 421 ms.
- Default: `FUSED8_ATTN` stays `8`. The engine's verify sequence (FLM_DLL gemma4e_dflash_verify.cpp, not stream A's file) must first adopt the
  attnx host contract, which is the same as `f8::run` with `attnx == true` in host/test_fused8.hpp:
  - RTPs (names as in fused8_rtp_addresses.txt): `RTP_attnx_score_3_4` / `_5_4` (s = 0 / 1) = [n_sub, hi0, u0, is_swa, 4*s].
  - `RTP_attnx_pv_{2..5}_2` (j = 0..3) = [n_sub, is_swa, j].
  - n_sub = rows / (8 global, 16 SWA keys per sub-chunk).
  - K (MM2S ch0) / V (ch1) replay: one BD each, sizes {2, rows/G, G, DH}, strides {DH, G*DKv, DKv, 1}, no packet header.
  - G = 32 global / 16 SWA; rows = the replayed row count rounded up to G.
  Then build with `ATT=x` (build/fused8x_full_L8) and deploy that xclbin.

## 42-layer chain (`-t chain8`, random-token sequences; `CHAIN_SEED_FUSED=1` seeds the context with fused passes, the oracle and the
fused pass then start from the same snapshot; logs work/attnx/chain_*.log)
| xclbin | c0 | fused 42 layers (dispatch only) | oracle 8 x 42 decode | argmax | min hidden cos |
|---|---|---|---|---|---|
| fused8x | 1000 | 142.2 ms | 898.7 ms | 2/8 | 0.976 |
| fused8x | 4000 | 152.2 ms | 952.4 ms | 3/8 | 0.980 |
| fused8x | 16000 | 172.8 ms | 1162.2 ms | 3/8 | 0.983 |
- fused8x 1000 with decode seeding (first run): 148.0 ms, argmax 4/8, min cos 0.972.
- Per-layer at 16000: SWA 3.20 / global 8.22 ms (min).
- Noise floor after the snapshot fix: decode vs a second decode pass from the same snapshot = 8/8, cos 1.00000.
  The 92% / 0.967 "floor" in fused8_results.md was produced by the V-half restore bug.
- A/B with the old attention, same tree (`fused88`, i.e. ATT=8):
  - c0 1000: 181.0 ms, argmax 4/8, min cos 0.973.
  - c0 4000: 229.6 ms, argmax 3/8, min cos 0.982.
  - c0 16000: not run (seeding exceeds the 90-min rig timeout); estimate from hammer8 per-layer: 35 x 3.8 + 7 x 41 ~ 420 ms vs 173 ms measured for fused8x.
- The chain's accuracy is the same class for both attentions: on random-token contexts the fused design differs from 8 decode passes by
  hidden cos 0.97-0.98 and argmax 2-4/8 at these contexts.
  - This is NOT an attnx property: same numbers with attn8.
  - Per layer the fused design matches the decode oracle at cos >= 0.9988.
  - Stream acceptance on real text (AL in the engine) is the right judge.
  - Open question for the coordinator: whether the rows with large fused margins (e.g. 102 vs oracle 1.9) are random-token
    artefacts (huge hidden norms) or an accumulated bf16 bias of the fused projections.
- Rig speed: seeding costs ~1.8 s per 8-position fused pass on the host (pli_row since made ~10x faster; the rest is per-call host
  BO setup), so a 16k chain takes ~75 min.

## Summary
- Chosen design: attnx (K/V once per verify, score/PV split, lazy-max online softmax). It is integrated as `FUSED8_ATTN=x`
  (build/fused8x_full_L8); the default is still `8` until the engine adopts the host contract above.
- Whole-layer oracle: 88/88 PASS. hammer8: 6000 dispatches, 0 diffs. Chain: 42 layers 142 / 152 / 173 ms at 1k / 4k / 16k
  vs 181 / 230 / ~420 ms with attn8.
