<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# gemma_npu engine notes

Serves gemma3:4b, medgemma:4b, medgemma1.5:4b and translategemma:4b (registered as `"gemma3"`). All four have
the same text shape (D 2560, 34 layers, vocab 262,208).

## Decode: lm_head inside the layer xclbin (2026-11-23)

Decode runs on one xclbin, `layer_lm.xclbin` (kernel folder `FLM_Xclbin/Gemma3/gemma_npu_lm_bin`, see its
README for the kernel side). Before, lm_head was its own `lm_head.xclbin` in a second hw_context, so every token
switched contexts twice (~2.4 ms into the layers, ~2.7 ms into lm_head on Strix Halo).

How a token runs now (`forward()`), same order as before:
1. the 34 layer runs as one runlist; wait;
2. host final RMS norm of x (unchanged: `_rms_norm`, so the logits keep their bits);
3. `lm_head_run.start()`: a second `npu_app` on the same xclbin, sequence `gen_lm_head_in_layer_seq()`, made once
   at load (it does not depend on the context length);
4. `set_context_length(L + 1)` rebuilds the next token's 34 runs (~4.8 ms of host work) while lm_head runs;
5. wait, read the logits.

Step 4 must stay between start and wait: lm_head (~9.8 ms) is what hides the rebuild. Appending lm_head to the
layer runlist instead would put the rebuild back on the critical path.

lm_head run arguments (5, the layer kernel's slots): 0 `x`, 1 `lm_head_weights`, 2 `lm_head_norm` (a copy of
`model.norm.weight`; the rms core takes one weight buffer even in pass-through mode), 3 `lm_head_unused`,
4 `logits`.
- `lm_head_weights` = `lm_head.weight` through `reorder_cpy(..., 2)`, the layer weights' layout (the old
  `lm_head.xclbin` used `reorder_cpy(..., 4)`).
- `logits` has `get_lm_head_in_layer_rows()` = 263,680 rows (vocab padded to whole rounds of D = 2560);
  `logits_valid` is the first 262,208. The padding rows of the weights stay zero.
- The proj cores get mode 0 and the rms core mode 2 (copy x, no norm); the layer runs write mode 1. Both are
  RTPs behind lock 6 that every sequence releases.

The old 32-core lm_head sequence lives on only in the kernel test (`host/test_lm_head.cpp`), as its reference.

### Check against the old engine (2026-11-23, Strix Halo, Performance mode)

Engine-level A/B (gemma3:4b): the engine at 995203ba (layer.xclbin + lm_head.xclbin) vs this one
(layer_lm.xclbin), same kernels otherwise, same prompts (stage0 Darwin text, 1,024 and 4,096 tokens), greedy:
- logits byte-identical at every step: 1,001 steps at 1K (prefill's last row + 1,000 decode tokens), 257 at 4K;
  same tokens; the 1K reply ends (end of turn) at token 654.
- `forward()` median over 3 alternating rounds of 128 tokens: 1K 57.38 -> 54.54 ms (-2.84 ms, +5.2%),
  4K 59.17 -> 56.40 ms (-2.77 ms, +4.9%).

The gain is less than the 5.1 ms of the two switches because lm_head in the layer design takes ~9.8 ms instead
of ~7.6 ms (the layer design's weight flow reads ~44 GB/s; see the kernel README).

## Sliding ring read order (2026-11-23)

A sliding layer keeps its last 1,024 K/V rows in a ring (row = position % 1024). Past 1,024 tokens the decode layer
now reads the ring in whole 16-row chunks, starting at the chunk after the one holding the newest row
(`ring_split` in `_move_kv_cache`), instead of oldest-first from `L % 1024`. The attention cores take keys in groups
of 16, so each group is now one ring chunk. That lets a later bfp16 copy of the cache be packed once per chunk.

Same keys, same mask (all 1,024 valid), different grouping, so past 1,024 tokens the bf16 online softmax rounds
differently. Up to 1,024 tokens, and whenever `L % 16 == 0`, the read is exactly as before.

Measured (gemma3:4b, Strix Halo; engine harness, step-2 engine vs this one, same kernels):
- 510-token prompt + 514 tokens (context <= 1,024): logits byte-identical at all 515 steps.
- 1K / 4K prompt, 256 forced tokens (same context at every step), three orders compared pairwise (old
  oldest-first, this one, and "always from row 0"): every pair differs alike, median cos 0.99984-0.99987, 5th
  percentile 0.99945-0.99963, top-1 the same at 95-99% of steps, KL median 5-10e-4. So the old order is not more
  accurate; the decode attention's bf16 running max and corrections are this sensitive to grouping.

## bfp16 KV cache (2026-11-23)

The decode layer (`layer_lm.xclbin`, kernel README "bfp16 KV") reads every finished 16-row chunk of K and V from a
bfp16 copy (44% fewer bytes) and only the chunk with the newest row as bf16. The bf16 cache stays the source of
truth: prefill, `get_k_cache` / `get_v_cache` and the state functions read and write it as before.

Where the copy lives: behind the 4 bf16 regions in the same kv buffer (arg 4 of the layer run; no new argument):
per stream (K01, K23, V01, V23) one 9,216 B unit per chunk (`gemma_npu_sequence::get_kv_bfp16_offset`; stream 4 =
the buffer's total size). Global layers: `MAX_L / 16` units per stream (+56% on top of the bf16 rows), sliding
layers: 64. `kv_bf16_elems(layer)` is the bf16 part; every sync or copy that only concerns the rows uses it, so
states, checkpoints and prefill syncs do not grow.

Who packs, when (`_kv_bfp16_ensure`, called by `set_context_length`, so before every layer run list):
- the chunk that will get the next row is marked stale (the array reads it as bf16 and writes the row);
- every other chunk the next run reads packed and that is stale is packed now: its rows are pulled from the device
  (exact ranged syncs), packed on the host (`gemma_bfp16_pack.hpp`, OpenMP over chunks) and pushed back.
- In decode `set_context_length` runs under the lm_head run, so the packing (one chunk per layer every 16 tokens)
  is hidden. After a prefill, a restore or a state write, the first run packs everything stale at once.

What makes chunks stale (`kv_packed_valid`): prefill rows written on the host (`_kv_bfp16_invalidate_positions`),
`clear_context`, `write_state`, `write_state_kv` (all chunks), `restore` (the sliding rings; global rows below the
checkpoint did not change, and a chunk above it is the newest chunk, so stale, before it is read packed again),
`update_max_length` (new global buffers). `checkpoint`, `read_state`, `new_state` only read.

Checked against the 3a engine (eded76ee: bf16 cache, same ring order, its bf16 kernel), gemma3:4b, Strix Halo,
Performance mode; one process runs six scenarios, each after `clear_context`: logits and tokens byte-identical in
all of them:
- 510-token prompt + 514 tokens (context <= 1,024); 1K and 4K prompts + 256 tokens;
- 1K prompt, 40 tokens, a second 300-token prefill, 60 tokens;
- 600-token prompt, `checkpoint`, 40 tokens, `restore`, 40 tokens;
- 700-token prompt, `read_state`, 30 tokens, another request, `write_state`, 30 tokens (replays exactly).

Decode (`forward()`), median of 3 alternating rounds of 128 tokens: 1K 54.60 -> 53.75 ms (-0.85 ms, +1.6%), 4K 56.39
-> 55.20 ms (-1.19 ms, +2.2%). Prefill time unchanged (1K 1.65 / 1.61 s, 4K 4.90 / 4.86 s, one run each, including
the packing of the whole prompt).

`restore` and a bit-for-bit replay (fixed 2026-11-23): `restore` puts back only the sliding rings, so the global
rows above the checkpoint kept what was decoded after it. The attention takes the softmax max over all 16 lanes of
the newest chunk without a mask, so those rows changed the rounding (cos 0.99983 at the first step, greedy tokens
differed from step 9; the same before this engine). `restore` now zeroes them as `write_state` does
(`kv_rows_zero` up to `dirty_rows`) and marks their chunks stale: the decode after `restore` is byte-identical to
the one after `checkpoint` (40 / 40 steps); the other five scenarios are unchanged.
