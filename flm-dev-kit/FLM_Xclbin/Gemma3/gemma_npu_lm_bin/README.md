<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Gemma3 4B decode layer with lm_head (`layer_lm.xclbin`)

The decode layer of gemma3:4b, medgemma:4b, medgemma1.5:4b and translategemma:4b (engine `gemma_npu`), with
lm_head on the same xclbin, so a decode token needs no hw_context switch. It replaces `Gemma3/gemma_npu_bin`
(`layer.xclbin`) and `Gemma3/gemma_lm_head_npu_bin` (`lm_head.xclbin`), which cost ~2.4 + ~2.7 ms of switches a
token on Strix Halo; both are in git history (removed 2026-11-23).

## Build

Peano only, with the pinned toolchain in `FLM_Xclbin/README.md`.

| Command | Output |
|---|---|
| `make bitstream` | `build/GEMMA3_4B/xclbins/layer_lm.xclbin` + `rtp_addresses.txt` |
| `make bitstream STUB=1` | `build/GEMMA3_4B_STUB/...`: same locks and DMA, projection and attention math skipped (data floor) |
| `make txn` | the same design built with `--aie-generate-txn` (kernel contract check only) |
| `make test_lm_head` | `build/test/test_lm_head` |

`make bitstream` fails when:
- one of the 25 RTPs the engine writes is not at its address (`IS_ATTN_PROJ_BUFFER_*` 33280,
  `IS_ATTN_RMS_BUFFERS` 54272, `L_attn_qk_core_*` 20992, `L_attn_kv_core_*` 50176; see
  `FLM_DLL/detail/gemma_npu/gemma_npu_sequence_detail.hpp`);
- a core's program is over 16 KB (the size is printed for every core; a bigger program links but aiecc's CDO
  step then fails without a reason).

## Modes (RTP `IS_ATTN`, written by the host before lock 6 is released)

| proj cores | rms core | What runs |
|---|---|---|
| 1 | 1 | one decoder layer (unchanged) |
| 0 | 0 | lm_head with the final RMS norm on the rms core (in the shared kernels too, never used by an engine) |
| 0 | **2** | lm_head, x already normalized on the host: the rms core copies x instead of normalizing it |

Mode 2 is new here. It keeps the engine's host RMS norm, so the logits are the same bits as `lm_head.xclbin`
gives today. Mode 0 would move the norm to the NPU (saves the x round trip, ~0.05 ms, but different bits).

lm_head in the layer design (`gen_lm_head_in_layer_seq` in `gemma_npu_sequence.cpp`):
- args: 0 x (D bf16), 1 lm_head weights, 2 any D bf16 (the rms core takes one weight buffer even in mode 2),
  3 unused, 4 logits;
- the kernel is built for vocab 262,208 (`common.h`), padded to 263,680 rows = 103 rounds of D logits = 515
  rounds of 512 rows (16 cores x 32);
- weights are `lm_head.weight` laid out like the layer weights: `reorder_cpy(..., 2)` (pairs of 32-row blocks
  per 256-column block), not the `reorder_cpy(..., 4)` of `lm_head.xclbin`;
- logits leave the rms core on MM2S 1 to IT1 S2MM 0; the per-round ping/pong selectors move one step per
  buffer on both sides, so a layer run after an lm_head run starts in step.

## Test (`test_lm_head`)

```
build/test/test_lm_head --model <Gemma3-4B-NPU2 dir> --old <lm_head.xclbin> \
    --new build/GEMMA3_4B/xclbins/layer_lm.xclbin --x <file of D-bf16 rows> --random 4 --iters 10 --cpu-ref 3
build/test/test_lm_head --model <dir> --new build/GEMMA3_4B_STUB/xclbins/layer_lm.xclbin --stub 1
```

The engine no longer uses `lm_head.xclbin`; the test keeps its 32-core sequence (`gen_old_lm_head_seq`) as the
reference. The same x goes to `lm_head.xclbin` and to `layer_lm.xclbin` in mode 2; every logit must have the same bits
(PASSED / FAILED). Both are also compared with a CPU Q4_1 reference (`w = q * scale + min`, block layout as
`proj.cc` reads it). Then it times each back to back and alternating (alternating pays a context switch).

Results 2026-11-23, Strix Halo, Performance mode, 24 real normalized x from gemma3:4b decode at 1K and 4K
context + 4 random:
- logits bit-identical for all 28 (0 of 262,208 differ), same top-1, cos vs CPU 0.999987 or better (both);
- time, back to back: `layer_lm.xclbin` 9.84 ms, `lm_head.xclbin` 7.56 ms; alternating 12.23 / 10.20 ms;
- stub `layer_lm.xclbin`: 9.59 ms, so this lm_head is limited by the layer design's data flow (~44 GB/s),
  not by the 16 cores' math. `lm_head.xclbin` reads at ~55.8 GB/s.

## Program size (`.text`, bytes)

| Core | old `Gemma3/gemma_npu_bin` | this folder |
|---|---|---|
| proj (16 cores) | 10,864 | 10,864 |
| rms (`core_1_2`) | 6,608 | 6,784 |
| Q.K attention (4 cores) | 2,736 | 3,248 (bfp16 KV) |
| P.V attention (4 cores) | 3,568 | 4,256 (bfp16 KV) |
| others | 1,184-4,192 | same |

## bfp16 KV (2026-11-23)

The decode attention reads every finished 16-key chunk of K and V from a host-made bfp16 copy (8 values share one
exponent: 9 bytes per 8 values, 44% fewer bytes than bf16) and uses the native bfp16 multiply; only the chunk with
the newest row still comes as bf16 and is packed on the core. Same bits as the bf16 decode attention it replaced.

Why the same bits: the old attention's `aie::mmul<4, 8, 8, bf16, bf16>` was emulated (`-DAIE_API_EMULATE_BFLOAT16_
MMUL_WITH_BFP16`): every multiply converts A and B to bfp16 (`to_v64bfp16ebs8`) and calls `mac_8x8_8x8T_conf`. The
kernels now feed that instruction the same blocks:
- K: one block = one key's 8 dh values (host: `tile_group`); Q converted per dh block as the emulation does.
- V: one block = one dh column's 8 keys, V^T (host: `vt_tile_group`); P converted once per chunk (the emulation
  converted it again for each of the 32 dh blocks, to the same blocks).
- y and the scores keep their accumulator lanes; rows 4-7 of the 8x8 result are unused, as before.
- the host packer copies MiniCPM-OCR's (`FLM_DLL/detail/gemma_npu/gemma_bfp16_pack.hpp`); it truncates like the core
  in its default rounding mode, so the kernels set `floor` at entry.

Data path (per layer run, per stream K01 / K23 on IT0 / IT7 ch0, V01 / V23 on ch1):
- one 9,216 B unit per finished chunk, from the copy behind the 4 bf16 regions of the same kv buffer (no new
  buffer argument): `[9 slices][head 0: 512 B | head 1: 512 B]`, each head's 4,608 B packed bytes cut into 512 B
  slices. That is the shape of 9 bf16 cache rows, so one fixed mem tile pattern (9 x 256 of every 512) splits both
  packed units and bf16 rows to the two cores.
- then the newest chunk as bf16: one shim descriptor reads rows 0-8 and 8-16, so each core gets keys 0-7 in one K/V
  buffer and keys 8-15 in the other (rows 8 and 16 unused; row 16 can be the next region, still inside the buffer).
- at most 3 descriptors per shim channel (two packed ranges for the sliding ring past 1,024 keys, one tail), no
  waits; past 1,024 keys the ring is read in whole chunks after the newest one (engine README, "Sliding ring read
  order").
- RTP `L[1]` = packed chunks (`(L - 1) / 16`, or 63 for the ring past 1,024 keys). The attention RTPs moved with
  the smaller buffers: `L_attn_qk` 9216 -> 20992, `L_attn_kv` 33792 -> 50176 (proj 33280 and rms 54272 did not).
- core K/V buffers 4,608 B (were 8 KB), mem tile buffers 9,216 B (were 16 KB), plus one 4,608 B tail buffer per
  attention core (an IRON buffer argument, as in MiniCPM-OCR: kernel statics have little room).

### Layer test (removed; commit 4e676be2)

While both attentions existed (commit 4e676be2, `make bitstream KV_BFP16=1` and `host/test_layer_kv.cpp`), one decode
layer ran on both builds with the same random inputs (Q4_1 weight blocks, norms, rope tables, x, a random
K/V cache and its bfp16 copy from the host packer): the output x and the K/V row the layer writes must be the same
bytes. Results 2026-11-23 (Strix Halo, Performance mode): PASSED twice, 20 global lengths (1 ... 8,192) and 17
sliding lengths (1 ... 5,000, every 16-row and 1,024-key edge); x checksums change with L (the sliding ring repeats
itself at L and L + 1,024, as expected).

One layer run alone, median of 7, ms (bf16 / bfp16 / bfp16 stub without attention math):

| Layer | bf16 | bfp16 | stub |
|---|---|---|---|
| global, L 1,088 | 1.42-1.52 | 1.37-1.49 | 1.33 |
| global, L 4,160 | 1.72-1.83 | 1.62-1.66 | 1.43 |
| global, L 8,192 | 2.19-2.25 | 2.00 | 1.62 |
| sliding, ring full | 1.35 | 1.32 | 1.28 |

At 4K-8K the global layers' attention math is now partly visible (stub vs bfp16: 0.2-0.4 ms).

## Weight streaming probes (2026-11-23)

Why lm_head inside this design (9.8 ms) is slower than the old 32-core `lm_head.xclbin` (7.5 ms) although both
stream the same 421 MB on 8 channels. Timing-only builds, all on top of `STUB=1` (no math), each removing one data
path (`make bitstream STUB=1 PROBE=<p>`, `build/test/test_lm_head --new <that xclbin> --probe <p>`; the test then
uses a probe copy of the lm_head sequence):

| Build | What it removes | lm_head ms (median of 10) |
|---|---|---|
| `STUB=1` | the math | 9.64 |
| `PROBE=NO_X` | the x re-send: proj cores take no x slices, the rms core sends none | 7.67 |
| `PROBE=NO_Y` | the logits: proj cores send no outputs, the rms core collects none | 7.76 |
| `PROBE=NO_W` | the weights: proj cores take none, the sequence sends none | 2.75 |

Removing either the x re-send or the logits collection brings lm_head to the DDR floor (~7.5 ms). Both run through
the rms core, which works one 512-row round at a time: send x to the 16 proj cores, wait for the round's logits,
copy them out, wait until the x send is done, and only then send x for the next round. So between rounds the proj
cores wait for x and the weight stream stalls. The host side is not the limit: one 3-D descriptor per channel and
matrix instead of one per round with a wait each gave 9.77 / 9.57 ms (tried, reverted). The x and logits traffic
alone (no weights) takes 2.75 ms.

Fix (same day): x is the same for every round, so the rms core now queues round i + 1's x sends before it
collects round i's logits (at most 10 sends outstanding; 515 in total as before). Same data, same order per
stream: logits bit-identical to `lm_head.xclbin` (28 / 28 x vectors). lm_head inside the layer design: 9.84 ->
7.81 ms (the separate `lm_head.xclbin`: 7.71 ms in the same run). rms core program 6,736 -> 6,784 B.

`PROBE` only adds `#ifdef`s; the normal build's program sizes are unchanged.

## Layer time split (2026-11-23)

`make test_layer_time`, then `build/test/test_layer_time --model <Gemma3-4B-NPU2 dir> --xclbin <layer_lm.xclbin>`:
one decode layer run at context L = 16 / 1,088 / 4,160, random inputs, median of 20 (ms). Strix Halo, Performance
mode, after the lm_head fix above:

| Layer | L 16 | L 1,088 | L 4,160 |
|---|---|---|---|
| global | 1.24 | 1.34 | 1.63 |
| global, `STUB=1` (no math) | 1.24 | 1.28 | 1.43 |
| sliding | 1.27 | 1.34 | 1.34 |
| sliding, `STUB=1` | 1.24 | 1.28 | 1.29 |

- Weights alone (L 16) already take 1.24 ms for 59 MB (~85% of the ~55.7 GB/s DDR limit, which would be
  ~1.06 ms); the math is hidden there. lm_head, one long matrix, now streams at ~97%. A layer streams 4 matrices
  (q/k/v, o, up/gate, down) and between them the proj cores wait for the previous step (rope + attention, rms norm,
  GELU), while each weight mem tile holds only 2 x 20 KB of read-ahead (41 KB of its 512 KB used), so the weight
  stream stops at each step: ~0.18 ms per layer, ~6 ms a token.
- K/V adds 0.04-0.19 ms; the attention math ~0.05 ms at 1K and ~0.2 ms for a global layer at 4K.

### Weight read-ahead (tried 2026-11-23, not kept)

Idea: more weight buffering in the 4 weight mem tiles (they use 41 of 512 KB), so the shim keeps streaming while the
proj cores wait between the layer's matrices. Variants as N buffers x S k-steps (today 2 x 1), same bytes and order:
- 4 x 1: lm_head logits bit-identical; one layer at L 16 1.242 ms (1.246 today), lm_head 7.91 ms (7.99): no gain.
- 2 x 5 and 4 x 5: deadlock. In each weight half the first core's mem tile channel sends its whole buffer before
  the second core's channel may start (one lock chain), and all 16 proj cores take x from one broadcast stream that
  moves only when every core has taken the slice, with 2 slices of room per core. So the first core can be at most
  2 k-steps ahead of the second: buffers of more than 2 k-steps cannot work in this design.

So read-ahead depth is not what costs ~0.18 ms per layer: the projection path runs in lockstep (one x broadcast,
chained weight readers), and any wait (the steps between the 4 matrices) stops all 16 cores and the weight stream.
A fix would be a redesign of the projection data flow (for example x resident per core for a whole matrix, as in
the old `lm_head.xclbin`, and independent weight readers).

## Attention math (2026-11-23)

Timing probes `make bitstream PROBE=NO_QK_MATH` / `PROBE=NO_KV_MATH` skip the Q.K / P.V math (data still moves);
`test_layer_time`, median of 20, ms:

| Layer | normal | no Q.K math | no P.V math |
|---|---|---|---|
| global, L 1,088 | 1.339 | 1.335 | 1.288 |
| global, L 4,160 | 1.632 | 1.620 | 1.439 |
| global, L 8,192 | 2.039 | 2.018 | 1.639 |
| sliding, ring full | 1.337-1.358 | 1.330-1.336 | 1.289 |

Q.K costs at most 0.02 ms (so converting Q once per call instead of per chunk would not pay); P.V costs ~0.19 ms per
global layer at 4K and ~0.40 ms at 8K, ~0.05 ms per sliding layer.
