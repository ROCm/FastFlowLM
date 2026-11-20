<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Bidirectional prefill attention (d3)

`gated_attention_prefill` (Qwen3.5's causal prefill attention) with the causal bound removed, for the d3 decision
encoder (`FLM_DLL/detail/d3_npu`), whose full-attention layers let every token see every token of its prompt.
Same shapes (head_dim 256, 8 q heads over 2 kv heads, 128-row q chunks, 4 heads per pass), same dataflow, RTP
addresses and memtile design; only the bounds differ:

| | `gated_attention_prefill` | this kernel |
|---|---|---|
| keys per q chunk | `[max(0, q - window), q chunk end)` | `[0, ceil128(L_valid))` |
| mask | key <= query, key > query - window | key < `L_valid` |
| `window_size` RTP slot | window | `L_valid` (prompt length) |

`kernel/attn.cc` and `kernel/attn_kernel.h` (`apply_key_mask_and_get_max`) carry the change; `host/seq_gen.hpp`
(`gen_seq_bidirectional`) streams all of K and V for every q chunk. A score tile is `[query][key]` (keys along the
lanes); padding keys get the same `-max` score the causal mask uses, so their weight is 0 — but their V rows still
enter `P * V`. Keep padding V finite **and small** (zero): the array's bf16 MMUL is emulated with bfp16, whose blocks
share an exponent along the reduction dimension, so a padding V of 1000 costs the real V rows of its block their
precision (|err| ~8) even at weight 0.

Test: `make host && ./run.exe -x build/xclbins/attn.xclbin` — random inputs at L_valid 1 ... 4096 (chunk edges and
non-multiples of 16 and 128), padding K large (an unmasked padding key would take the softmax), every head of sampled
rows (`--all_rows 1`: every row) against an fp64 reference, and repeat runs bit-identical.
