<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# fused8: the whole 8-row verify layer on hardware (plan S6.11 / S6.12), 2026-10-29

IRON `FLM_Xclbin/Gemma4/verify_decoding`, branch feat/bolt-dflash: commit 3d1a586a3 (design + host driver) and the chain8 test commit after it.
Design `DESIGN=fused8` (`iron/fused8_aie.py`, `iron/fused8_mt.py`), build `work/scripts/build_fused8.sh <attn|nople|full>` (kernels: proj_main rms_residual_l8 glu_l8 rope swa_rope
attn_qk_e4b attn_kv_e4b swa_attn_qk swa_attn_kv ple8_gate ple8_up ple8_norm; Peano; the three xclbins build and route in 2-6 s, `full` = 211,980 B).
Allocation table: `work/notes/fused8_allocation.md`.  Test: `work/scripts/run_fused8.sh` (whole layer vs the oracle) and `run.exe -t chain8` (42 layers).

## What is integrated
shim x[8][D] -> norm tile (3,2) phase 1 (h on the lender (3,3)) -> qkv projection (MT3 replay) -> y stages (MT4) -> RoPE (3,4)/(3,5) -> q park (MT4) -> attention (col 2, KV through IT2/MT2)
-> o rows (8 packets, ids 17/18) land in MT3 (the landing buffers are the o park) -> o projection -> rows stage (MT5) -> norm phase 2 -> up|gate -> GLU (4,4) -> MT3 down buffer -> down
-> rows stage -> norm phase 3 -> PLE feed (MT5) -> G (4,2) / U (5,2) / ple stage / N (5,3) -> layer output rows on IT5.S2MM0.  All four layer variants (SWA/global x skip/non-skip) in ONE xclbin
(runtime RTPs + host sequences as in decode).

## Plan changes found while integrating (all forced by the tools / measured)
1. **A memtile has one DMA body**: the stages that shared MT4 / MT5 became chains of one `memtile_hub` program (iron/fused8_mt.py).
2. **Memtile BD ids are split by channel parity** (AIEAssignBufferDescriptorIDs: even channels use ids 0-23, odd 24-47, S2MM and MM2S share): MT3 even = main 2 + down 20 + replay MM2S0/2, odd = o_global 8 + o_swa 8 + replay MM2S1/3 -> phase order [qkv, o_global, down, o_swa, up_gate].
3. A memtile BD cannot repeat with stride 0: the PLE feed to G repeats the h slices as 8 chained BDs.
4. The attention's o leaves as 8 packets (one per token); the o landing buffers of the replay tile are the o park (landing generalised to `land_rounds`).
5. The PLE inputs (pli, constants, h2 rows) are ordered through MT5 (one source per tile input), not the shim.
6. Poison contract: V padding rows must be finite (NaN or 1e4 right after the valid keys destroys even the shipping decode xclbin: the decode oracle fails with it), so the harness poison is K = NaN, V = 1e4 from the next 8-row boundary, applied to the verify path only.
7. `decoding_layer` loaded the 3.5 GB embedding tables per layer (OOM with 32 layers): now one shared copy.

## Acceptance
**Whole layer vs the oracle (8 sequential decode dispatches), real Bolt weights, layers 0/5/40/35 x cursors 0,1,7,8,9,16,40,505,511,512,520,1000 (48 cases)**: PASS 48/48.
Output rows: min cosine 0.99916 (L35 c=520), typical 0.9995-0.9998; K/V rows appended: cosine 0.9993-0.9997 (non-skip layers; skip layers append nothing).  Max 15 bf16 ulp (global L5).
Larger contexts (perf run, repeat 5): cursors 0..4000 all four layer types cosine >= 0.99934.
Poison (K NaN, V 1e4 from the next 8-row boundary) with repeat 3 passes on L5 c=9/505, L0 c=9, L35 c=40; repeat 30 bit-identical on L5 c=505, L35 c=505, L40 c=505, L0 c=9 (no hangs).
**Regression, D4**: the L_TOKENS=1 build of the same folder (constants.py change only touches L_TOKENS>1) passes the oracle harness on all four layer types (cos 0.99996-1.0, skip layers exact).

**Known issue (rounding level, deterministic)**: SWA non-skip layers with 33+ chunks (cursor >= 505): the FIRST dispatch after a context switch (here: right after the decode oracle ran) differs from the following dispatches
by 1 bf16 ulp in ~3% of the o elements (rel L2 3.6e-4); dispatches 2..N are bit-identical to each other; the first dispatch is itself reproducible across processes (md5 equal).  The appended K/V rows are bit-identical (so the difference arises in
the attention path, SWA qk/kv kernels), the attn-only debug design additionally hangs on the second dispatch of this case at times (not used for anything else).  It is noise-level and does not affect acceptance; open item: find which kernel state is read before it is initialised.

**42-layer chain (`-t chain8`), 20 random real-token sequences, c0 in {0,1,7,24,60,100,200,400}**: final hidden rows vs 8 sequential decode passes: cosine >= 0.9988 on 157/160 rows (3 rows < 0.99,
min 0.96568); logits argmax equal on 130/160 rows (81%).  In a 12-sequence run with the noise floor measured: **a second decode pass from the same state vs the first: argmax 88/96 (92%), min hidden cos 0.96714**;
fused vs decode in the same run: 81/96 (84%), min cos 0.96677.  So the low-cosine rows and most mismatches are the model's own sensitivity on random-token contexts (logit margins of the oracle's own top token are 20-70 units
while a 0.1% hidden change moves logits by tens): the decode path is not reproducible against itself at this level either.  The plan's 7/8 target is not met on this synthetic set (84% vs a 92% self-agreement floor): to be judged
on real text in Phase 8/11 (acceptance length and stream agreement).  Each mismatch is printed by the test with both margins (work/chain42.log, work/chain42b.log).
The verify-vs-fp32 error ratio (<= 2.5x decode) was not measured at layer level; the stage tests gave 1.4x the decode core's error for the projections.

## Performance (dispatch only: create_run + wait of ONE layer, host overhead of one launch included; decode column = the same kind of measurement for one token)
| layer | ctx 0 | 500 | 1000 | 2000 | 4000 | decode 1 token (ctx 1000) |
|---|---|---|---|---|---|---|
| SWA non-skip (L0) | 3.41 ms | 4.06 | 4.14 | 4.12 | 4.13 | 2.60 ms |
| global non-skip (L5) | 3.64 | 4.85 | 6.00 | 8.39 | 13.06 | 3.05 ms |
| SWA skip (L40) | 3.26 | 4.01 | 4.01 | 4.00 | 4.04 | 2.53 ms |
| global skip (L35) | 3.44 | 4.68 | 5.81 | 8.15 | 13.10 | 2.92 ms |
SWA layers: 1.6x a decode step and flat in the context; global layers: 1.2x at ctx 0 growing to 2x at 1k and 3.3x at 4k (the K/V replay runs once per token: 8x the single-token traffic).
42-layer chain (small contexts, c0 <= 400): fused 139-170 ms for 8 tokens (mean 147 ms) vs 826-980 ms for 8 sequential decode passes of the same host rig (the rig adds ~1 ms per decode dispatch of host overhead).
Estimate at ctx 1000: 20 x 4.14 + 15 x 4.01 + 4 x 6.0 + 3 x 5.8 = 184 ms (launch overhead of 42 separate runs included; in the engine's runlist the launches amortise).
Stage split of one SWA layer (stage-design measurements, not instrumented inside fused8): projection weight stream at 8 tokens ~1.5-1.6 ms (compute-bound 1.36x the weight stream), norm/GLU round trips +0.15-0.2 ms,
PLE ~0.55 ms (1.8x its stream floor), SWA attention ~1.0 ms (flat), RoPE/qkv y stage ~0.2 ms; the sum (~3.6 ms) plus launch ~0.4 ms matches 4.0 ms.
Speedup outlook: 8-token verify ~= 2.0-2.6x one decode token (engine decode 70 ms/token at 1k): with AL 3.8 that is ~1.4-1.6x end to end before draft and lm_head costs: the >=1.6x gate needs the perf work below.

## Interface for the engine (Phase 8)
- xclbin: `fused8.xclbin` from `DESIGN=fused8 L_TOKENS=8 MODEL_TYPE=GEMMA4_E4B` (deploy as `verify.xclbin`), RTP addresses in `fused8_rtp_addresses.txt` next to it (read them at load: do not hardcode).
- Kernel args (same count as decode): 0 `x` BO, 1 `proj` (the layer's weights BO, unchanged layout), 2 `rms` (4 x D norm vectors, unchanged), 3 an output BO (`y`: layer output rows [8][D] at offset 0 from IT5.S2MM0; the engine can point it at a slot of its x BO),
  4 `kv_cache` (global: the decode cache, K plane then V plane, row = absolute position; SWA: see below).
- x BO per dispatch: `[8][D] layer input rows | [8*512 + 2*512] global RoPE table | [8*256 + 2*256] SWA RoPE table | [8][256] pli | [D + 32] w_post_pli, layer_scalar, pad`; only the table of the layer's kind is sent.
  RoPE table = `[(cos(DH/2) | sin(DH/2)) x 8 | q_norm(DH) | k_norm(DH)]` (cos/sin of positions cursor..cursor+7 with the rope_scaling factor, as decode writes them for one position).
- pli rows computed on the host (tools/ple8_ref.py pli_row; `f8::pli_row` in host/test_fused8.hpp): `t = bf16(W_plmp . x0); t = bf16(t * 0.01976423537605237); t = bf16(t * rsqrt(mean(t^2) + 1e-6) * w_pn); t = bf16(t + x0_pl); pli = bf16(t * 0.7071067811865476)`;
  W_plmp = the layer's PLE weight at `w_per_layer_input_proj_offset` ([row block][k][32] bf16), x0 = embedding row of the token, x0_pl = the layer's slice [layer*256, +256) of the per-layer embedding row, w_pn = pli norm weights (rope_rms BO at 3*DH + 256).
- Shim sequence per layer (host/test_fused8.hpp `run`): RTP writes; x rows (IT3 MM2S0, pkt 26), rms weights (IT3 MM2S1, pkt 11), the RoPE table (global IT3 MM2S0 pkt 9 / SWA IT3 MM2S1 pkt 10); output S2MM (IT5 S2MM0); pli (IT4 MM2S0 pkt 27), constants (IT4 MM2S1 pkt 28),
  W_gate (IT5 MM2S0), W_up (IT5 MM2S1); qkv weights; KV append of the 8 rows (K plane then V plane, wait each) BEFORE the replay; KV replay (MM2S IT2, repeat count L carried in size[0]); o, up|gate, down weights; waits.
- RTPs written per dispatch: proj tiles x16: IS_SWA (53248), SKIP_KV (15360), lock 6 = 1; global RoPE skip (40960), SWA RoPE skip (20480); global qk L_max (57344), global kv L_max (14976); SWA qk (53248): L[0] = L_max, L[1] = ctx - S, L[2] = ctx + 1 - 512 - S, SWA kv L_max (57664).
- L_max rules: global L_max = ctx + 8, chunks = ceil(L_max/16), replay rows chunks*16 starting at row 0; SWA: S = floor16(max(0, ctx - 511)), L_max = ctx + 8 - S.  Rows past the valid range in the last chunk must be FINITE (zero): the engine's zeroing of rows past the context guarantees it.
- SWA cache contract: a LINEAR window buffer, row = position - base, 8 contiguous rows appended at `ctx - base`, replay range `[S - base, ctx + 8 - base)`; the host compacts (moves the last ~544 rows to the front, advances `base`) when `ctx + 8 - base + 16` exceeds the buffer.
  The harness builds this buffer from the engine ring for each dispatch and copies the 8 appended rows back to the ring slots; the engine should adopt the linear layout for both decode and verify (the decode kernels need no change, its replay becomes one BD).
- What the host must precompute per cycle: the 8 RoPE tables (global + SWA), 8 x 42 pli rows (one 42x256 matvec per token), the per-layer constants (w_post_pli, layer_scalar are static per layer), the SWA linear windows.
- Output per layer: 8 rows [8][D] in place of the next layer's input; taps for the draft are the outputs of layers 5, 11, 17, 23, 35 (just read those 5 outputs: no per-layer x slots needed because every layer writes its own output BO region).

## Not done / next
- The SWA first-dispatch rounding difference (find the uninitialised state); the global-attention cost at long context (K/V replay once per token): ideas: token-pair interchange with 2 x 16 KB y accumulators for the SWA kv tile, larger replay chunks, skipping the MT2 staging.
- Perf to reach the 1.6x gate: remove launch overhead in the engine runlist, speed up the compute-bound 8-token core (mmul<8,8,16>, epilogue scale/min), overlap PLE with the next layer's qkv weights.
- fp32 layer-level error comparison (verify <= 2.5x decode); a Chess build; CI wiring (S6.15).
