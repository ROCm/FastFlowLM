<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# gemma_npu engine notes

Serves gemma3:4b, medgemma:4b, medgemma1.5:4b and translategemma:4b (registered as `"gemma3"`). All four have
the same text shape (D 2560, 34 layers, vocab 262,208).

## Decode: lm_head inside the layer xclbin (2026-11-23)

Decode runs on one xclbin, `layer_lm.xclbin` (kernel folder `FLM_Xclbin/Gemma3/gemma3_4b_npu_bin`, see its
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
