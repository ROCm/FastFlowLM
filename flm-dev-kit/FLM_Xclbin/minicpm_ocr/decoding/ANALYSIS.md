<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Qwen3.5-2B decode layer: analysis (2026-10-30)

This folder is a private copy of `FLM_Xclbin/Qwen3_5/qwen3_5_decoding` (identical at copy time, 62 files),
made so 2B-only changes do not affect the 0.8B/4B/9B builds. Build target: `MODEL_TYPE=QWEN3_5_2B`.
Same folder depth as the original, so the relative Makefile paths (`../../host_common`, ...) still work.
`host/models.h:24` still points at `/path/to/models/` (standalone host test data only).

Status of the numbers below: **[M]** measured, **[V]** verified in code, **[E]** estimated (no build
output or trace exists yet; cycle counts assume a ~1.8 GHz AIE clock).

## 1. Where the time goes (roofline)

Machine: Krackan test machine, Krackan NPU (8 columns x 4 rows), DDR5-5600 x2 = **89.6 GB/s** peak. Numbers below were taken in power mode Default; in Performance mode (now the baseline) decode is 2-3 % slower: 10K focus 26.44 tok/s, TTFT 7.18 s.

Bytes per decoded token [V]:

| Item | Format | MB / token |
|---|---|---|
| lm_head (separate `lm_head.xclbin`) | Q8_0 | 540.3 |
| 18 GDN layers (qkv, z, out, up/gate/down, a/b) | Q4_K (+Q8_0 a/b) | 613.8 |
| 6 attention layers (q+gate, k, v, o, up/gate/down) | Q4_K | 181.8 |
| GDN state, 1 MB fp32 per layer, read + write | fp32 | 36 |
| KV cache, 12 KB per context token | bf16 | 0.012 x L (123 at 10K) |

Decode time per token [M]: **30.02 ms + 0.654 us x L** (fit over L = 136..10248, residuals < 0.01 ms).
At the 10K focus: 36.72 ms/token (27.2 tok/s), 1,498 MB/token, 40.8 GB/s = 46 % of peak; the floor at peak
bandwidth is 16.7 ms. Math is ~2.7 FLOP/byte, far below the compute ridge: decode is memory/DMA bound.

## 2. Array map [V] (`iron/layer_aie.py`)

| Column | Row 0 | Row 1 | Row 2 | Row 3 | Mem tile / shim |
|---|---|---|---|---|---|
| 0, 1, 6, 7 | proj | proj | proj | proj | weights: shims 0, 1, 6, 7, both MM2S channels each (8 channels) |
| 2 | rms + residual | RoPE | attn output gate | delta-net norm | MT2 aux; IT2 x / rms weights |
| 3 | attn QK | attn KV | **idle** | **idle** | MT3; IT3 K (ch0), V (ch1), new KV in |
| 4 | delta net | delta net | delta net | delta net | MT4; IT4 ch0 state in and out |
| 5 | conv | alpha/beta proj | **idle** | GLU | MT5 linear aux; IT5 AB / conv weights, conv cache |

One xclbin run per layer, 25 runs per token in one runlist (`set_layer_rtp` + 24 layers), then `lm_head.xclbin`
(another full-array context) every token.

## 3. Projection engine (`iron/proj_tile.py`, `kernels/proj_main.cc`, `kernels/q4_core.h`)

- [V] Weights: 8 shim channels -> mem tiles 0/1/6/7 (ping/pong of 2 chunks per S2MM) -> one chunk per core.
  The two cores fed by one channel are chained by locks (`proj_tile.py:398-421`). Each core: 2 weight + 2 x buffers.
- [V] x is one stream broadcast to all 16 cores and re-sent once per 512-row round; all cores move in lockstep.
- [V] Each core makes 32 output rows per round; a round = 512 rows. Host: one flat BD per channel per round,
  ping/pong BD sets, and a **wait on all 8 channels at the end of every projection** (`qwen3_5vl_npu_sequence.cpp:876-887`).
- [E] Inner loop per 4736 B chunk: ~320 MAC issues + ~150 other vector ops, 400-800 cycles (up to ~1,300 if the
  Peano schedule is poor). Data arrives at one chunk per core per ~2,368 cycles if a channel moves 4 B/cycle, so
  the cores look DMA-bound, 20-50 % busy.
- Per GDN layer 33.95 MB, per attention layer 30.3 MB, **793 MB per token**: 8.85 ms at DDR peak.
- **Open question:** the per-channel limit. If it is 4 B/cycle the 8 channels cap at 57.6 GB/s (13.8 ms/token).
  But the attention K stream is measured at >= 9.4 GB/s on one channel, more than 4 B/cycle at 1.8 GHz,
  so this cap is not proven. Measure it.

## 4. Attention (`iron/attn.py`, `kernels/attn_qk.cc`, `kernels/attn_kv.cc`) - drives the 10K slope

- [V] 2B uses `num_attn_cus = 1` (`ATTN_IMPL_2x4x1`): QK on CT[0][3], KV on CT[1][3]; CT[2][3]/CT[3][3] idle.
  K on IT3 MM2S ch0, V on ch1, one linear BD each of `L_padded x 512` bf16; MT3 reorders with 16-byte runs.
  Blocks of lk = 16 keys (16 KB of K and of V), double buffered in MT3 and L1.
- [V] Softmax split: QK core keeps the running max m and computes p and the correction c; KV core does
  l = l c + sum p, y = y c + p V, and the final y / l.
- [M] 0.654 us per context token over 6 layers = **1.744 us = ~3,140 cycles per 16-key block per layer**.
- [V] The QK core calls `update()` (`attn_qk.cc:82-107`) **8 times in a row per block, once per head**: scalar
  `reduce_max`, scalar bf16 max/sub, a whole 16-lane `getExpBf16` to get one correction value, scalar float casts
  (software float on AIE2P). [E] ~1,500-2,500 cycles of the block: this is the bottleneck, not DMA.
- [V] The matmuls compute 8 heads against both KV heads and keep half; BFP16 emulation converts operands every call.
- [V] KV core: 8 software float compares `c[h] != 1.0f` and an fp32 rescale of y every block. [E] 1,200-2,000 cycles.
- DDR is not the limit here (18.8 GB/s marginal).

## 5. GatedDeltaNet (`iron/Gated_Delta.py`, `kernels/gate_delta_net_decode_128_128_128_32.cc`, `Conv`, `AB_tile`)

- [V] State 16 x 128 x 128 fp32 = 1 MB per layer, read from and written to DDR every token through
  **IT4 channel 0 only, each way** (`_move_s`, `qwen3_5vl_npu_sequence.cpp:1040`). 36 MB per token.
- [V] 4 cores; each takes a 32-wide V slice of every head, 16 calls per layer, 16 KB in + 16 KB out per call.
- [E] Delta kernel ~2.5k cycles per head per core (fp32 built from bf16 limbs); cores ~85 % idle waiting for state.
- [V] Host order in one GDN layer, with blocking waits (W): conv weights W -> old conv cache W -> AB weights W ->
  qkv weights -> new conv cache W -> state -> z -> out -> up/gate -> down. Conv waits for the whole 6144 qkv vector
  and runs all 48 chunks before any head reaches the delta net; the norm waits for all 2048 values.
- [V] AB projection: Q8_0 64 x 2048 where 32 of the 64 rows are padding (139 KB per layer).
- [E] Not hidden behind projections: ~150 us per GDN layer, **~2.7 ms per token**. The state move alone is
  ~146 us per layer if a channel is 4 B/cycle (same unproven assumption as section 3).
- **[V] Latent race:** the core has one `S_out_chunk` buffer. The output DMA (`Gated_Delta.py:151-160`) sends
  S_out then O and only then releases `CT_S_O_out_prod_lock`, but the kernel writes the next head's S into
  `S_out_chunk` in its main loop (`.cc:189`) and takes that lock only afterwards (`.cc:199`). Today the slow state
  feed hides it. **Fix this before any change that speeds up the state stream** (double-buffer S_out, or take the
  lock before the main loop).

## 6. Change ideas, ranked for the 10K focus

| # | Idea | Where | Est. gain @10K | Risk |
|---|---|---|---|---|
| 1 | Vectorize the softmax bookkeeping: one pass over all 8 heads, vector m/c, one vector exp for corrections, no scalar float | `attn_qk.cc`, `lut_based_ops.h` | -2.5 to -3.5 ms | low-medium |
| 2 | Cheaper KV core: vector "any c != 1" test, lazy rescale | `attn_kv.cc` | ~-1 ms more | medium (re-check long-KV accuracy) |
| 3 | Use the 2 idle cores for attention: split keys (flash-decoding) with the proven 2x4x1 kernel, or 2-CU head split (`ATTN_IMPL_1x8x1`, needs a `RoPE.py:104,111` DQ_PADDED fix) | `attn.py`, `layer_aie.py`, kernels, host sequence | ~-3 ms (halves the slope) | medium-high |
| 4 | GDN: fix the race, issue the state move before qkv, drop host waits on small transfers, state over 2 channels (IT4 ch1 is free) | `Gated_Delta.py`, kernel, host sequence | ~-1 to -2 ms | medium |
| 5 | GDN: pack AB rows (no padding), stream conv per head | `AB_tile.py`, `conv.cc`, host | ~-0.4 ms | low-medium |
| 6 | Projections: remove end-of-projection waits, deeper mem-tile buffers | `proj_tile.py`, host | ~-0.5-1 ms | medium |
| 7 | Projections: feed weights through more shim channels (16) | `proj_tile.py`, `layer_aie.py` | up to -4-5 ms | high (routing, deadlocks) |
| - | lm_head (36 % of bytes) is a separate xclbin: not in this copy | `Qwen3_5/lm_head_npu_bin` | large | separate work |

## 7. Measure before changing

1. Host-side timing per run (layers vs lm_head, attention vs GDN layer, xclbin switch), engine flag only.
2. A plain rebuild of this copy (`make bitstream MODEL_TYPE=QWEN3_5_2B`) to confirm the toolchain works here and
   matches the shipped `layer.xclbin` speed and output.
3. Cycle counters around the lock waits on CT[0][3] / CT[1][3] (do not keep timer values live across calls).
4. Power mode is now Performance (set by the user 2026-10-30). It did not speed decode up on this box (2-3 % slower), so the clock assumption (~1.8 GHz) still needs a real measurement.

## 8. Step A: rebuild of this copy (2026-10-30, power mode Performance)

- Env: `source ~/setup.sh` (Vitis AIE 1.3.0, mlir-aie ironenv; llvm-aie **21.0.0**, CI pins 22.0.0).
  `make bitstream MODEL_TYPE=QWEN3_5_2B`: Peano build **9 s**, Chess build (`ENABLE_CHESSCC=1`) 2 min 20 s.
  RTP check passes (qk 53248, kv 53536). The 32 "error" lines in the log are the `chess_error` macro text inside
  warnings, not errors.
- The shipped `FastFlowLM/src/xclbins/Qwen3.5-2B-NPU2/layer.xclbin` dates from 2026-09-08 (FastFlowLM f81eba71),
  i.e. **older source** than this copy (09-10 changes and the 10-15..10-17 Peano port came later), most likely a
  Chess build (347,964 B; our Chess build 340,460 B; our Peano build 235,004 B).
- Speed, decode tok/s (3 iters):

  | prompt | shipped | **this copy, Peano (new baseline)** | change |
  |---|---|---|---|
  | 136 | 32.44 | 33.44-33.61 | +3 % |
  | 1,032 | 31.86 | 32.96 | +3.5 % |
  | 4,104 | 29.68 | 30.98 | +4.4 % |
  | **10,248** | **26.44** | **28.18-28.26** | **+6.6 %** |

  Prefill/TTFT unchanged (prefill does not use `layer.xclbin`). Gain grows with L: attention got faster
  (per-context-token cost ~0.65 -> ~0.55 us).
- **Greedy text differs** between shipped, our Chess build and our Peano build, each pair within ~20 decoded tokens,
  all fluent. Decode is sensitive to rounding, so identical text is not a usable correctness test; we need a numeric
  check (see below).
- **Builds are not byte-reproducible:** two Peano builds of the same source differ only in packet-switch arbiter
  assignment (`aie.masterset` / `aie.rule`, 28 lines in `input_physical.mlir`) plus UUIDs. Measured speed is the same
  (28.26 vs 28.22 tok/s at 10K). Still: keep the exact measured binary in an xclbin set, and smoke-test every build
  for hangs.
- (2026-11-02 cleanup) The xclbin test sets and the harness `XCLBIN_ROOT` option were removed. Kernel work and
  speed testing now happen in this folder; only a finished `layer.xclbin` goes to
  `FastFlowLM/src/xclbins/Qwen3.5-2B-NPU2/` for the end-to-end test (the shipped one stays in FastFlowLM git, f81eba71).

## 9. Measured breakdown at the 10K focus (2026-11-02, `make test` in this folder, Performance mode)

Kernel-folder decode test (see `host/README.md`): 24 layers in one run list, random 10K context.

| part | 128 cached | 10,240 cached | note |
|---|---|---|---|
| 24 layers | 16.66 ms | **22.17 ms** | slope 0.545 us per cached token = end-to-end slope |
| one attention layer (alone) | 607 us | 1,509 us | ~900 us of attention at 10K; 30.3 MB weights |
| one GDN layer (alone) | 809 us | 823 us | constant; 34 MB weights -> ~42 GB/s effective |
| lm_head + host work (e2e minus layers) | ~13.2 ms | ~13.3 ms | 540 MB Q8_0; outside this folder |

End to end at 10K is 35.49 ms/token (28.18 tok/s). Rough shares: GDN layers ~14.6 ms (41 %), attention layers
~9 ms (25 %, of which ~5.4 ms is the context-dependent attention), lm_head + host ~13.3 ms (37 %).
A GDN layer costs ~130 us more than its weight bytes explain at the attention-layer rate (607 us for 30.3 MB),
x18 = ~2.3 ms: the GDN-specific stalls of section 5.

## 10. Correctness check (2026-11-02)

`make test ARGS="--check [--kv-scale N]"`: every layer vs an fp32 CPU reference with the same Q4_K/Q8_0 weights
(see `host/README.md`). Baseline worst cosines: GDN state 0.99996, conv 0.99946, attention K/V 0.99904,
attention output 0.99887 (kv-scale 1) / 0.99761 (kv-scale 8). Changes must not lower these.

## 11. Probe measurements: where the time really goes (2026-11-02)

Probe switches in the kernels (no effect unless the flag is set; results are wrong, timing only).
Build with `rm -f build/QWEN3_5_2B/kernels/*.o; make bitstream KERNEL_PEANO_EXTRA="-DPROBE_..."`, copy the
xclbin away, run `./run.exe --xclbin <copy> --per-layer`, then rebuild without the flag.

| switch | what it removes | where |
|---|---|---|
| `PROBE_QK_STUB_UPDATE` | softmax bookkeeping (`update()`: max, exp, correction) | `kernels/attn_qk.cc` |
| `PROBE_QK_SKIP` | all QK-core math | `kernels/attn_qk.cc` |
| `PROBE_KV_SKIP` | all KV-core math (`calculate_l`, `attn_fv`) | `kernels/attn_kv.cc` |
| `PROBE_PROJ_SKIP` | the Q4_K matrix-vector math in the 16 projection cores | `kernels/proj_main.cc` |

Results (layer run alone, Performance mode):

| variant | attention layer @128 | attention layer @10K | GDN layer @128 |
|---|---|---|---|
| baseline | 607-622 us | 1,508-1,544 us | 809-824 us |
| `PROBE_QK_STUB_UPDATE` | | 1,485 us (-23) | |
| `PROBE_KV_SKIP` | | 1,378 us (-130) | |
| `PROBE_QK_SKIP` + `PROBE_KV_SKIP` | | **1,350 us** | |
| `PROBE_PROJ_SKIP` | 584 us | 1,479 us | 785 us (-5 %) |

(`PROBE_QK_SKIP` alone made it slower, 2,080 us: the KV core then gets garbage scores and rescales every block.)

Conclusions:
- **Decode is data-movement bound, not compute bound.** Removing the projection math saves ~2 %; removing all
  attention math saves ~10 % of the 10K attention layer. Section 4's "update() is the bottleneck" was wrong.
- **10K attention = streaming K and V.** With no math the layer still takes ~740 us more at 10K than at 128:
  21 MB of K+V per layer over the 2 channels of shim column 3, ~14 GB/s per channel (~8 B/cycle at ~1.8 GHz).
  Faster attention needs more DMA channels for K/V (columns 4/5 are idle in attention layers) or fewer bytes.
- **Projections run at ~44-50 GB/s** (30.3 MB in 607 us; 36 MB in 823 us) with 8 channels that could carry far
  more, against 89.6 GB/s DDR peak. Open question: is ~50 GB/s this box's real NPU read ceiling (old XRT 2.21 /
  FW 1.1.2.64), or design stalls? A pure-DMA bandwidth test would answer it.

## 12. NPU read bandwidth on this box (2026-11-02, `../dma_bandwidth/`)

Pure DMA test, read-only, Performance mode: 1 channel 13.7 GB/s, 2 -> 27.8, 4 -> 49.7, 8 -> ~55 (4x2 or 8x1),
**16 -> 78.7 GB/s**. So the old XRT/firmware does not cap the NPU at ~50 GB/s; it can read ~79 GB/s.

- Projections use 8 channels and run at 44-50 GB/s: they sit at the **8-channel ceiling (~55 GB/s)**.
  Feeding the 793 MB of layer weights through all 16 channels (~79 GB/s) is the biggest lever: roughly
  -4 to -5 ms per token (~12-14 % end to end). Needs IT2..IT5 channels in the weight path (`proj_tile.py`,
  `layer_aie.py`, host `_move_weights`); those shims also carry x, rms/rope weights, K/V, GDN state.
- Attention K/V uses 2 channels: ~28 GB/s ceiling. 4 channels would roughly halve the 10K attention
  streaming time (~-2 ms per token at 10K).

## 13. 8-bit KV cache accuracy, CPU simulation (2026-11-02)

Real 10,240-token context (story text decoded token by token on the NPU), token 10,241, cosine / rel error of the
attention-layer output x: NPU's existing error 0.99937-0.99982 (1.9-4.0 %); **int8 KV (one scale per token per
KV head) 0.999982-0.999998 (0.18-0.61 %)**; fp8 e4m3 KV 0.99993-0.99999 (0.5-1.3 %). int8 adds 5-15x less error
than the NPU already has. One text, one position: a longer quality check is still needed before shipping.
Expected gain if done: K/V bytes halve -> ~-370 us per attention layer at 10K (~-2.2 ms/token).

## 14. Pauses inside a token can corrupt the decode layer

Determinism test (`--repeat`, `--gap-ms`): back-to-back runs are bit-identical; a pause of >= ~50 ms between two
layer runs of the same token corrupted everything after it (10 ms fine) - while another process (user's other
session, dequant_mm) was using the same NPU. Most likely the NPU's time-sharing loads the other design in the
pause and wipes our on-chip state. Not re-tested with the NPU exclusive. Engine (one run list per token) is safe
within a token. See host/README.md.

## 15. Whole-token profile: two context switches cost ~5.5 ms per token (2026-11-02)

Engine copy with timers (`../engine/`, README there) + `run.exe --lm-head <lm_head.xclbin>` (this folder):

| ms per call, 10K | bundled layer.xclbin | our Peano layer.xclbin |
|---|---|---|
| lm_head alone, back to back | 7.37 (540 MB, ~73 GB/s) | 7.42 |
| 24 layers alone | 24.05 | 22.25 |
| 24 layers right after lm_head | 26.64 (switch ~2.6) | 24.41 (switch ~2.2) |
| lm_head right after layers | 10.39 (switch ~3.0) | 10.38 (switch ~3.0) |

Per token at 10K (engine, bundled xclbins, ~38 ms): layers 24.0, **context switches ~5.6**, lm_head 7.4, host ~1.0
(sampling etc.), run-list rebuild 2.6 ms hidden under lm_head. Switch cost does not depend on context length,
so its share is larger for short prompts (~18 % at 128 tokens).
Ways out: one xclbin for layers + lm_head (overlay / runtime op switch, see memory notes; ~0.1 ms switches seen on
another NPU2 design), or lm_head computed by the layer design's projection engine.

## 16. Feasibility: one xclbin for layers + lm_head (overlay), 2026-11-02

`../lm_head_npu_bin/` was an unchanged copy of `Qwen3_5/lm_head_npu_bin`, built for QWEN3_5_2B (Peano, 4 s). It was
removed on 2026-11-03 (lm_head now runs inside layer.xclbin, so nothing used it); it is in git at IRON `2261dffc2`.

- **Program memory (16 KB per core on NPU2, aie-rt `xaie2pgbl_reginit.c`):** fits. lm_head = 4.8 KB on all 32
  cores; decode = 6.96 KB (projection), 4.5-5.4 KB (attention), 2.8 KB (GDN), max 10.5 KB (conv core (5,2)).
  Worst sum 15.3 KB (conv core), ~1 KB to spare.
- **Data memory: does not fit side by side.** lm_head uses 56 KB of every core and 465 KB of every mem tile;
  decode uses 48-64 KB per core and up to 449 KB per mem tile. The two would have to take turns in the same
  memory: shared buffer areas, decode's fixed data (LUTs 4-8 KB on attention / alpha-beta cores) placed outside
  them, and every kernel resetting its ping/pong statics after a switch.
- **DMA:** lm_head uses all 16 shim channels and every mem-tile input; decode uses its own set. Each switch
  rewrites BDs, locks and routes (earlier NPU2 overlay: 118-363 us per switch, ~1.5-5 K register writes).
- The overlay tooling from earlier work (extract_txn, op dispatch) is not on this machine.

Verdict: possible, but large work (both full-array designs merged, all kernels touched, tooling rebuilt);
expected saving ~5.6 ms -> ~0.3-0.8 ms of switching per token, i.e. ~-5 ms/token.

## 17. Packet route capture: a new flow broke the RoPE flow (2026-11-02)

Adding the lm_head output flow (`proj_pkt_id.to_out`, projection cores -> assemble mem tiles -> main mem tile
(1,1) -> shim) made the first attention layer hang (ERT_CMD_STATE_TIMEOUT) in every build.

- Cause, read from `layer.mlir.prj/input_physical.mlir`: at switchbox (1,1), slave port DMA:5 (main mem tile ->
  RoPE core), the router wrote the packet rules as `rule(mask 16, value 0) -> South` **before**
  `rule(mask 31, value 1) -> North`. The switch takes the **first** rule that matches. Id 1 has bit 4 = 0, so it
  matched the South rule and the RoPE core never got Q/K/V.
- The mask-16 rule is the router merging all ids with bit 4 = 0 that go south (to_conv 4, to_aux 8, to_out 12...).
  Changing to_out's id (3, 5, 6, 7, 10, 11, 12, 13) or its shim (IT[0..5]) did not help: same order every time.
- Fix: `proj_pkt_id.to_rope` 1 -> **17** (bit 4 = 1, so the mask-16 rule can no longer catch it). Ids 16, 18, 20
  also trace clean. `it1_pkt_id.to_rope` (shim -> RoPE) stays 1: it is a different flow.
- Gate: `tools/check_packet_routes.py` follows every packet flow through the physical switchboxes (rules in
  order, first match wins) and fails the build if a flow lands somewhere other than its declared destinations.
  `makefiles/bitstream.mk` runs it after every aiecc build. It caught the id-1 case (rc 1) and passes the fix.
- Checks after the fix: state checksum `d7fd26d20f33f7fc` (same as baseline), 0 of 2 repeats differ, 10K speed
  22.22 ms, `--check --kv-scale 8` worst cosine 0.997611. All as step 1.
