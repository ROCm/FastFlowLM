<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# attention path of the 8-token verify layer: RoPE, q park, global + SWA attention (plan S6.5 second half, S6.9, S6.10), 2026-10-29

IRON `FLM_Xclbin/Gemma4/verify_decoding`, branch feat/bolt-dflash, commits 022d7c637 (RoPE), c84ebd8ce (q park + global), aca5666cb (SWA).
Stage designs (stand-alone, shim-fed): `DESIGN=attn8rope` (RoPE tiles only), `DESIGN=attn8attn` (RoPE + q park + global and SWA attention columns).

## Build / run
```
work/scripts/build_attn8.sh attn8attn 8 rope.o swa_rope.o attn_qk_e4b.o attn_kv_e4b.o swa_attn_qk.o swa_attn_kv.o   # explicit kernel list (other forks keep WIP kernels in kernels/)
work/scripts/build_attn8_host.sh 8                                                                                 # attn8.exe (host_attn8/attn8_main.cpp + host/decoding_layer.cpp)
work/scripts/run_attn8.sh <rope|attn> <layer> <ctx> [repeat]      # POISON=3 for the padding-row poison; gen + dispatch + score (tools/attn8_ref.py)
```
Layers: 0 swa, 5 global, 40 swa-skip, 35 global-skip. Reference = fp32 numpy on the same bf16 inputs (real q_norm/k_norm weights and real cos/sin tables of layer N).

## Results (hardware)
RoPE stage (all four variants, ctx 0/7/9/500/1000): cos >= 0.99997 for q, k, v (relL2 <= 0.0065), 20 repeats bit-identical, 150-214 us per dispatch.

Attention stage, cos of o per token vs fp32, min over 8 tokens x 8 heads (poison on, repeats bit-identical, no hangs):
| layer | ctx 0..56 | ctx ~500 | ctx 1000 | ctx 4000 |
|---|---|---|---|---|
| global non-skip (L5) | 0.9989-0.9994 | 0.9961-0.9965 | 0.9939 | 0.9787 |
| global skip (L35) | 0.9989-0.9994 | 0.9965 | 0.9936 | - |
| swa non-skip (L0) | 0.9991-0.9998 | 0.9985-0.9991 | 0.9978 | 0.9983 |
| swa skip (L40) | 0.9995 | 0.9982-0.9984 | 0.9979 | 0.9981 |
Peaked attention (q_norm weights x6, the more realistic regime): global ctx 1000 0.9980, ctx 4000 0.9953.
K/V rows appended before the replay land in the right rows (cos >= 0.99998 vs the fp32 rope reference). Cursor sweep covered
0,1,7,8,9,14,15,16,30,40,56,500,505,510,511,512,513,520,527,528,1000,1040,4000 (swa) and 0..4000 (global).

### What limits o accuracy (not the 8-token structure)
The same bfp16-emulated mmul (`AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`) and the reciprocal LUT of the shipping decode kernels: token 0 at ctx 0 (one key,
o must equal v) is already 1.3 % off; near-uniform attention over thousands of keys (random test inputs) averages unit-size V rows down to a small
output, so the relative error grows with the key count (global relL2 0.03 -> 0.19 at ctx 4000); peaked attention is much better (above). The per-token
error is flat across tokens (no pattern with the token index or the chunk tail). The decode kernels are the same code per token, so the formal
equivalence check is the whole-layer 8 x decode oracle (plan S6.12), not this stage test.

### Kernel contract found with the poison test (important for the engine)
The V mmul runs through bfp16 blocks of 8 keys.  Padding rows of the replayed range (rows past the valid keys inside the last chunk) must be
**finite**, and large values must not share an 8-row block with a valid key (1e4 at the same block as a valid row destroys it); **NaN in V is never
tolerated** (0 x NaN), NaN in K is. The engine already guarantees it: rows past the committed context are zero (kv_rows_zero, rejected-row zeroing on commit).
The stage test poisons K rows >= ctx+8 with NaN and V rows from the next 8-row boundary with 1e4 (output unchanged); `--poison 10` (V NaN) is the expected-FAIL probe.

## Design (what was built)
- `kernels/rope.cc`, `swa_rope.cc` (`#if L_TOKENS > 1`): input one head per fill as `[L][DH]`, per-token cos/sin table
  `rope_w = [(cos(DH/2)|sin(DH/2)) x L | q_norm(DH) | k_norm(DH)]`, q leaves one head per fill, k/v through one buffer: k0,k1,v0,v1 (none on skip layers).
  `iron/rope8.py` (global CT[2][3], swa CT[3][3]); L1 global: qkv 2x8 KB + q 8 KB + kv 8 KB + table 10 KB = 43 KB.
- `iron/qpark.py`: MT4 holds q of all tokens head-major `[head][token][DH]` (global 64 KB, swa 32 KB, two independent parks on channels 0/1); the
  write BD acquires/releases L_TOKENS units, each of the L_TOKENS read BDs 1 unit (every BD needs an acquire and a release), BD t reads
  token t's rows of every head (offset t*DH, dims ((8, L*DH), (DH,1)), 1 KB runs).  The qk tile's own q BD reorders as in decode.
- `kernels/attn_qk_e4b.cc`: per-invocation token counter (static, wraps at L_TOKENS), `keys_t = L_max - 7 + t`, uniform chunk count `ceil(L_max/16)`
  (KV_CHUNK_BUG.md), mask `idx <= clamp(keys_t - 16 j, 0, 16)`; `attn_kv_e4b.cc` only changes the o repeat count to 1 (RTP address 14976 and tile size unchanged: 64,192 B).
- `kernels/swa_attn_qk.cc`: three RTP words `L[0]=rows=ctx+8-S, L[1]=hi0=ctx-S, L[2]=u0=ctx+1-512-S`, token t valid rows `max(0,u0+t) <= n <= hi0+t`
  (|2i - (lo+hi)| <= hi-lo mask, handles chunks fully masked at either end); `swa_attn_kv.cc`: o repeat 1.
- `iron/attn.py`: kv tiles' `O_REPEATS = 1` when L_TOKENS > 1.  o leaves once per token as packet id 0 (the o park / x replay is the integration's job; the stage sends it to the shim).

## Interfaces for the integration
| item | value |
|---|---|
| q park | MT4 (col 4 memtile): S2MM0/MM2S0 global, S2MM1/MM2S1 swa; fed by rope Q_out (flow), feeds the qk tile Q_in (flow) |
| rope inputs | S2MM0 qkv chunks (packet id any: stage uses 1), S2MM1 table (packet 2); MM2S1 kv -> IT2.S2MM0 (global) / IT2.S2MM1 (swa); skip layers: nothing on MM2S1 |
| attention | qk (2,2)/(4 swa), kv: unchanged tiles/routes (IT2 MM2S0/1 -> MT2 -> CU, packet 12 global / 13 swa); o: kv tile MM2S0, packet id `to_proj_x_in_pkt_id.from_attn/from_swa_attn` (0) |
| RTPs | rope skip: `RTP_RoPE_SKIP_KV_4_3` / `RTP_SWA_RoPE_SKIP_KV_5_3` (addresses move: read build/.../attn8attn_rtp_addresses.txt, stage values 40960 / 20480); global qk 57344 (L_max), global kv 14976 (L_max), swa qk 53248 (+4, +8 = hi0, u0), swa kv 57664 (rows) |
| shim BDs (global) | append: S2MM IT2 ch0, one BD per plane, sizes {1, 2 heads, 8 tokens, DH}, strides {0, DH, DKv, 1}, offset row0*DKv (+ v plane offset), wait both BEFORE the replay; replay: MM2S IT2 ch0 (K) / ch1 (V), sizes {8, chunks, 16, DKv}, strides {0, 16*DKv, DKv, 1}, packet 12 (13 swa), `no_cache`; the queue repeat count (size[0]) carries the token repeat |
| KV layout (engine contract) | global: two planes `[MAXL][2 x DH]`, row = absolute position, V plane right after K. **SWA: linear window buffer with host compaction** (deviates from plan D6, see below) |
| chunk rule | host chunk count must equal the kernels': `ceil(L[0]/16)` with L[0] = valid rows of the pass; the replay streams `chunks*16` rows from the range start |

### SWA cache layout: deviation from plan D6
The mirrored ring (every row written at p%R and p%R+R) fails when an 8-token batch straddles the wrap (rows no longer contiguous, and one shim S2MM
stream cannot be split into runs); it also needed R = 544, not 528 (union window of the 8 tokens with the start rounded down to 16 is up to 534 rows).
Used instead and recommended for the engine: a linear buffer of B rows per plane (e.g. 2048), row = position - base, host keeps `base`; when
`ctx + 8 - base + 16 > B` the host moves the last ~544 rows to the front (DDR memmove, ~1 MB per layer once per ~1500 tokens) and advances `base`.
Append = 8 contiguous rows at `ctx - base`; replay = one contiguous range `[S - base, ctx + 8 - base)` with `S = floor16(max(0, ctx - 511))`.  The decode xclbin
needs no kernel change and its host replay shrinks to a single BD (corner and wrapped two-phase cases vanish), so decode and verify can share this layout.

## L1 / L2 occupancy (from input_with_addresses.mlir of attn8attn)
| tile | role | buffers total | highest address | limit |
|---|---|---|---|---|
| (3,4) | global rope | 43,072 B | 57,344 | 65,536 |
| (3,5) | swa rope | 21,568 B | 53,248 | 65,536 |
| (2,2) | global qk | 49,216 | 57,408 | 65,536 |
| (2,3) | global kv | 58,048 | **64,192** | 65,536 (unchanged from decode) |
| (2,4) | swa qk | 41,024 | 53,312 | 65,536 |
| (2,5) | swa kv | 45,760 | 57,728 | 65,536 |
| MT4 | q park global 64 KB + swa 32 KB | 98,304 | | 524,288 |
| MT2 | KV staging (decode design, unchanged) | 196,608 | 475,136 | 524,288 |

## Perf (dispatch latency of the stage, shim-fed, min of 20-100)
| | ctx ~0 | ctx 505 | ctx 1000 | ctx 4000 |
|---|---|---|---|---|
| global (L5) | 0.33-0.44 ms | 1.5-2.1 ms | 2.6-2.7 ms | 10.1 ms |
| swa (L0 / L40) | 0.20-0.25 ms | 1.0 ms | 1.0 ms | 1.0 ms |
Per extra key row a global layer costs ~2.3 us for the 8 tokens (replay-bound: K+V rows are streamed once PER TOKEN, loop interchange is impossible because
8 x 16 KB y accumulators do not fit L1).  For comparison the decode harness (ledger): global layer 2.92 ms @505 -> 3.04 ms @1000 = 0.23 us per key for ONE token, SWA flat 2.5 ms:
the 8-token attention costs about 8x the single-token attention (global 1.27x of 8 sequential), i.e. **no attention saving, all savings come from the projections**.
Budget at ctx 1000: 7 global x ~2.7 ms + 35 swa x ~1.0 ms ~ 54 ms of attention per verify pass; at 4k: 7 x 10 + 35 = ~105 ms. This is the main perf risk of the
verify design at long context (plan section 5 / risks); remedies: larger replay chunks, a K/V replay that skips the memtile staging, token-pair interchange with 2 x 16 KB y accumulators for SWA.

## Regression: L_TOKENS=1 fused build
`build/GEMMA4_E4B_L1/xclbins/layer.xclbin` (explicit decode kernel list, kernels edited as above) passes the decode oracle harness
(`run_verify8.sh --mode oracle --layers 0,5,40,35 --cursors 0,9,505,1000 --poison --repeat 2`): all 16 cursor/layer cases PASS (exact or cos 1.0).
Size equals the decode xclbin (173,548 B); 116 bytes differ (build nondeterminism, as for the plain copy).

## Not done / next
- The o park (global 64 KB, swa 32 KB) and the o x-replay feed into the o projection phase (integration with the proj engine, rms_residual).
- q/k/v rows currently come from the shim; in the layer they come from the projstage y stage (MT4 token-major per head: exactly this stage's rope input format).
- Whole-layer equivalence against 8 sequential decode dispatches (S6.12).
