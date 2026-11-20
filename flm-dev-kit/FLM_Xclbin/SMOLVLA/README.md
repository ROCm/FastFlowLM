<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# SMOLVLA: the expert test hosts

This note covers the test programs of `expert/`, `expert_attn/` and `expert_mm/`. Each one runs its kernel on
the NPU and, by default, checks it against a CPU reference on random inputs. PASSED means cosine > 0.97 and
normalized MSE < 0.06 (the rule of `Hy_MT2_1.8B/attention_prefill`); a FAILED run exits with 1.

```bash
make bitstream host
./run.exe
```

## `expert_mm/` and `expert/`: the projection matmul

Both run the same check: `layer_0.qkv_proj`, 50 rows of `x [50][768]` (zero-padded to 64 for the tile)
times `W [1792][768]`, so `y = x W^T`. `expert/` runs it through the `expert.xclbin` design and `expert_mm/`
through `mm.xclbin`. Random `x` has standard deviation 1 and `W` has `1/sqrt(768)`, which keeps `y` near 1.
Measured on an NPU2 box: cosine 0.99988.

`--Random false --reference <activations> --weights <weights>` compares with a dump instead
(`expert_reference.safetensors` and `expert_weights_pad256.safetensors`, produced outside the dev kit;
the defaults look in `$FLM_REFERENCE_DIR`). `-r` is taken by `--Row`, so `--Random` has no short form.

## `expert_attn/`: the expert's attention over the prefix

15 query heads, 5 KV heads (3 query heads per KV head), head dim 64, scores scaled by `1/8`. The keys come in
four chunks of 64 rows, and `ref_attn_chunked` in `host/host.cpp` applies the kernel's rules
(`kernel/attn0.cc`, `attn1.cc`):

| chunk | holds | visible keys |
|---|---|---|
| 0, 1 | the 128 image rows | all 64 |
| 2 | text rows and the state row | the first `text_rows + 1` (`effective_number`) |
| 3 | the expert's own 50 rows | causal: query `i` sees keys `0..i` |

Random mode fills every input with standard deviation 1 (with the `1/8` scale the scores also come out near
1) and uses `--text_rows` (default 40) valid text rows. Measured: cosine 0.9969. A reference without the
causal rule on chunk 3 scores 0.94 and fails, so the check does see the mask.

`-r false --reference <expert_reference.safetensors> --layer_reference <ref.safetensors>` compares with
dumps instead; then the text length comes from the dump's `prefix_pad_masks`.
