<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# End-to-end results: DFlash speculative decoding with the attnx verify xclbin (2026-10-29)

Engine IRON `1a6fc4cd7` (attnx host contract in gemma4e_dflash_verify.cpp; FLM_DFLASH_ATTN=8 selects verify_attn8.xclbin for A/B).
Verify xclbins rebuilt from IRON HEAD (all stream P kernels): `verify.xclbin` = build/fused8x_full_L8 (attnx), `verify_attn8.xclbin` = build/fused88_full_L8;
both passed the whole-layer oracle (layers 0/5/40/35 x cursors 0/9/505/1000/2000, poison, repeat 2: 20/20 each) before deployment
into work/local-xclbins/xclbins/Nemotron-3.5-Bolt-E4B-DFlash-NPU2/. DFlash package: fc BF16 (IRON 3e458215b), 270 MB.
Box: quiet (load < 0.5), only one NPU job at a time. All runs greedy unless stated.

## 1. Accuracy on real text

### 1a. Stream equality (engine rig test_spec_engine, plain greedy vs speculative greedy)
| set | variant | identical | ties (<= 1 bf16 ulp) | flips (margin) | AL |
|---|---|---|---|---|---|
| 12 chat prompts x 128 | attnx | 5 | 4 | 3 (0.25, 0.25, 0.25 = 2 ulps) | 3.30 |
| 12 chat prompts x 128 | attn8 | 5 | 5 | 2 | 3.26 |
| 3 long prompts (495/1117/2317) x 256 | attnx | 0 | 1 | 2 (0.375, 0.375 = 3 ulps) | 2.37 |
| 3 long prompts x 256 | attn8 | 0 | 1 | 2 (0.375, 1.5) | 2.46 |
Every flip is at a top-1/top-2 margin of 2-3 bf16 ulps of the logit (logits ~16-32, ulp 0.125); none larger with attnx.

### 1b. Teacher-forced verify vs 8 sequential decodes (tf_replay, FLM_DFLASH_VERIFY=check, real text: SPEED-Bench throughput prompts
cut to 976 / 3976 / 15976 tokens + 256-token llama.cpp greedy continuations, thinking on; ref/seqs_long.json; 2 sequences per context)
| ctx | variant | rows | argmax differs | beyond 2 bf16 ulps | mean logits cos | min cos |
|---|---|---|---|---|---|---|
| 1k | attnx | 1136 | 24 (2.1%) | 8 (0.70%) | 0.99954 | 0.99496 |
| 4k | attnx | 1512 | 36 (2.4%) | 8 (0.53%) | 0.99945 | 0.99735 |
| 16k | attnx | 1192 | 42 (3.5%) | 4 (0.34%) | 0.99948 | 0.99646 |
| all | attnx | 3840 | 102 (2.66%) | 20 (0.52%) | | |
| all | attn8 | 3816 | 111 (2.91%) | 31 (0.81%) | ~0.99947 | 0.99573 |
TF acceptance length (drafts vs the forced continuation; taps from the verify path): attnx fused 3.142, attn8 fused 3.161, **sequential verify 3.049**;
target agreement with llama.cpp greedy: 0.915 / 0.919 / 0.912. Runs are deterministic (attnx check run twice: identical).
**Conclusion:** the 42-layer accumulated verify error does not matter on real text: 97.3% argmax agreement with sequential decode, 99.5% outside
near-ties, TF AL not lower than with sequential verify. The 2-4/8 agreement of the random-token chain (attn_longctx.md) is a property of flat
logits on random tokens. No stage investigation needed.

## 2. Speed (flm serve, local build, --prefix-slots 0, 512 max tokens, 3 prompts per context: ref/e2e_prompts.json; decode tok/s from the server)
Plain base tag `nemotron3.5-bolt:e4b` (greedy): 13.26 (~30 tokens of prompt) / 12.67 (1k) / 11.76 (4k) / 9.10 (16k) tok/s.
DFlash tag `nemotron3.5-bolt-dflash:e4b` (guard on, default):
| ctx | sampler | think | tok/s | speedup | AL |
|---|---|---|---|---|---|
| ~30 | greedy | off | 23.01 | 1.74x | 2.87 |
| ~30 | greedy | on | 25.12 | 1.89x | 2.95 |
| ~30 | top_k 64, T 1 | off | 21.12 | 1.59x | 2.59 |
| ~30 | top_k 64, T 1 | on | 22.65 | 1.71x | 2.66 |
| 1k | greedy | off | 19.56 | 1.54x | 2.49 |
| 1k | greedy | on | 22.51 | 1.78x | 2.86 |
| 1k | sample | off | 19.09 | 1.51x | 2.44 |
| 1k | sample | on | 22.29 | 1.76x | 2.74 |
| 4k | greedy | off | 15.63 | 1.33x | 2.10 |
| 4k | greedy | on | 18.25 | 1.55x | 2.52 |
| 4k | sample | off | 14.96 | 1.27x | 2.01 |
| 4k | sample | on | 16.24 | 1.38x | 2.35 |
| 16k | greedy | off | 13.01 | 1.43x | 2.03 |
| 16k | greedy | on | 14.94 | 1.64x | 2.31 |
| 16k | sample | off | 12.73 | 1.40x | 1.97 |
| 16k | sample | on | 14.65 | 1.61x | 2.27 |
Per prompt the speedup ranges 1.1x - 3.1x (coding prompt: 40.6 tok/s at AL 4.78). The long-context prompts are summarization of SPEED-Bench
throughput texts, which have lower AL than chat.
Engine rig (chat prompts, greedy, no thinking, 128 tokens): plain 13.37 vs spec 28.24 tok/s (2.11x, AL 3.30) with attnx; long prompts 12.42 vs
19.02 tok/s (1.53x) attnx vs 14.32 (1.15x) attn8.

### Per-cycle split (ms, tf_replay free mode / engine rig)
| ctx | host draft | draft lm_head_8 | verify total | of which 42 layers | lm_head_8 (verify) | accept | cycle | plain decode step |
|---|---|---|---|---|---|---|---|---|
| short | 6.2 | 10.1 | 98.2 | 84.5 | 11.9 | 0.5 | ~115 | 75 |
| 1k | 7.3 | 9.7 | 104.0 | 90.6 | 11.8 | 0.5 | ~121 | 79 |
| 4k | 7.8 | 9.7 | 110.4 | 97.0 | 11.8 | 0.5 | ~128 | 85 |
| 16k | 7.6 | 9.7 | 135.8 | 122.4 | 11.8 | 0.5 | ~153 | 110 |
Prime (draft context from the prompt taps): 100-116 ms once per request at 1k-16k (window 1024).

## 3. What limits speed now
1. **AL**: the cycle costs 1.4-1.5 plain steps, so tok/s = AL / 1.45 x plain; AL on these workloads is 2.0-2.9 (chat ~3.3, thinking on higher).
2. **Verify 42 layers = 1.1-1.15x a plain decode step at every context** (attnx made attention cheap): projections are near the weight-stream floor;
   the rest is the in-layer stage chain (rms -> qkv -> RoPE/attention -> o -> rms -> FFN -> PLE) and the per-layer launch.
3. **Fixed per-cycle overhead ~30 ms (~25% of a cycle)**: the draft head (lm_head_8 for the 7 draft rows, 9.7 ms), the verify lm_head_8 (11.8 ms),
   the host draft (6-8 ms). Both lm_head_8 dispatches are weight-stream bound (vocab 262144 x 2560 Q4 = ~350 MB each).
4. Long context: global attention still grows (42 layers 84.5 -> 122.4 ms from short to 16k) but slower than plain decode does, so the speedup holds at 16k.

## 4. `flm bench` (for the record; iterations 1, ~30 decoded tokens per stage, so AL depends on one bench prompt)
| ctx | base tok/s | DFlash tok/s | ratio |
|---|---|---|---|
| 1k | 13.15 | 17.86 | 1.36x |
| 2k | 12.78 | 18.85 | 1.48x |
| 4k | 12.18 | 18.26 | 1.50x |
| 8k | 11.14 | 21.92 | 1.97x |
| 16k | 9.51 | 11.03 | 1.16x |
| 32k | 7.36 | 7.27 | 0.99x (guard keeps it at plain speed) |
Prefill / TTFT identical between the tags (the draft priming is charged to prefill: +0.02-0.06 s).

## 5. Soak (flm serve, DFlash tag, thinking on, verify context never reloaded: FLM_DFLASH_VERIFY_RELOAD default 0)
3 x 2048-token generations (greedy, sampled, greedy) + 20 x 128-token requests back to back: 8246 tokens, all 23 requests complete and
non-empty, no dispatch timeout, no hang (grep for timeout/ERT errors in the server log: only words of the generated text). Long-generation
speeds 16.3 / 24.8 / 17.2 tok/s, short requests 27.7-53.6 tok/s. Log: e2e/serve_soak.log, e2e/soak.log.

## Files
Scripts: work/scripts/{e2e_rig.sh, e2e_tf.sh, e2e_bench.py, e2e_soak.py, e2e_tail.sh}; prompts ref/e2e_prompts.json (build_e2e_prompts.py),
sequences ref/seqs_long.json (gen_long_seqs.py); raw results work/e2e/*.
