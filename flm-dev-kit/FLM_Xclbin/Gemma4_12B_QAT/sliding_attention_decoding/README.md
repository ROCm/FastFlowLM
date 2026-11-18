<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Gemma4-12B-QAT sliding-window attention (decode)

One token, 16 query heads, 8 KV heads (2 query heads per KV head), head dim 256. The kernel has no local
mask: the host applies the window by handing it only the last `--Window` rows (default 1024 of 3094).

```bash
make bitstream host
./run.exe                      # random q/k/v, checked against the CPU reference; prints PASSED or FAILED
./run.exe -r false --reference <dump.safetensors>   # q/k/v and the expected output from a dump
```

## The default check: random inputs and a CPU reference

`host/attn_ref.hpp` recomputes the attention in fp32 on the CPU from exactly the q and KV buffers the NPU
gets (zero-padded rows included), so the two must agree up to the array's bf16 arithmetic. Like the kernel
it applies no `1/sqrt(dh)` scale (Gemma4 uses scale 1.0) and maps query head `h` to KV head
`h / (query heads per KV head)`.

- q and k are drawn with standard deviation 0.3 and v with 1.0. With unit-variance q and k the scores would
  have a standard deviation of `sqrt(dh)` (16-23), and the softmax would pick one key; at 0.3 they sit
  around 1.4-2, where many keys count and an error in any of them shows.
- PASSED means cosine > 0.97 and normalized MSE < 0.06, the same rule as `Hy_MT2_1.8B/attention_prefill`.
  Measured on an NPU2 box: cosine about 0.994.
- The CPU reference is plain scalar code (the host build has no `-O`), so it takes most of the run time.

## Checking against a dump

`-r false` reads `layer_0.q_embed`, the K input, `layer_0.v_norm` and `layer_0.attention_out` from a safetensors
file of the model's activations, produced with Hugging Face transformers outside the dev kit. Pass it with
`--reference`, or put it in `$FLM_REFERENCE_DIR/gemma4_12b_qat_reference.safetensors`.
