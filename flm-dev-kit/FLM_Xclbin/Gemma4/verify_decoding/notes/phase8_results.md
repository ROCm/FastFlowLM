<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Phase 8: the fused 8-row verify in the gemma4e_dflash engine (2026-10-29)

IRON `FLM_DLL/detail/gemma4e_dflash`, branch feat/bolt-dflash: commits `b4068405d` (fused verify + lm_head_8), `d28e86fcd` (overlap/lazy commit/ranged flush), `a300556bd` (HRX stubs), `7f689ba6a` (adaptive guard).
Files: `gemma4e_dflash_verify.cpp` (new), `gemma4e_dflash_spec_state.hpp` (new, shared Spec state), `gemma4e_dflash_spec.cpp`, `gemma4e_dflash_detail.hpp`,
`gemma4e_dflash.cpp` (lazy decode streams), `tools/test_spec_engine.cpp`.

## What it does
- **Load** (`load_draft` -> `verify_load`): registers `verify.xclbin` and `lm_head_8.xclbin` from the tag's xclbin folder (`xclbins/<xclbin_folder>/`, files `verify.xclbin`,
  `verify_rtp_addresses.txt`, `lm_head_8.xclbin`, `lm_head_8_rtp_addresses.txt`; the two txt files are the build outputs `fused8_rtp_addresses.txt` / `rtp_addresses.txt` renamed).
  Anything missing, preemption mode or `FLM_DFLASH_VERIFY=seq` -> the sequential verify of Phase 5 (unchanged).
- **BOs**: 43 per-layer x BOs of `[8][D] rows | global RoPE table | SWA RoPE table | [8][256] pli | w_post_pli, layer_scalar` (80 KB each; static parts written once at load).
  Layer i writes its output rows into BO i+1 (kernel arg 3 = x BO of layer i+1), so the 42 runs chain on the device without host copies; the five taps are the rows of BOs 6, 12, 18, 24, 36.
  Weights: the decode BOs `proj_weights[i]`, `rms_weights[i]` and the engine KV buffers (global: row = position; sliding: the linear window buffer, replay `[S - base, ctx + 8 - base)`)
  are passed unchanged (no weight duplication; verify has its own hardware context).
- **Per cycle** (`verify_fused`): 8 embedding rows (padding rows repeat the last valid token, results ignored, their K/V rows are zeroed on commit), 8 RoPE positions per layer kind,
  8 x 42 pli rows on the host (AVX-512, `pli_layer_rows`, bit chain of tools/ple8_ref.py), `verify_prepare` (SWA compaction before the batch, hw context reload, 4 instruction streams, the 42-run
  runlist) which runs on another thread while the host draft computes, one runlist execute, taps copied from the BOs into the tap ring, `lm_head_8` on BO 42 (pre-norm rows, target norm) and
  un-interleave into `[8][vocab]` logits.
- **Draft head**: the 7 draft rows go through ONE `lm_head_8` dispatch (draft output_norm in the norm slot), argmax directly on the raw layout (parallel over the 7 rows).
- **Commit**: no decode-stream regeneration (lazy: `forward()` regenerates when `decode_seq_stale`), only the rejected rows (up to 8, padding rows included) are zeroed and flushed by byte range.
- **Env**: `FLM_DFLASH_VERIFY=seq` sequential verify, `=check` runs both every cycle and compares logits (prints per-row argmax differences with the sequential top-2 margin when `TEST_TRACE` is set,
  summary line in the speculation timing), `FLM_DFLASH_VERIFY_RELOAD=N` (default 200, 0 = never), debug: `FLM_DFLASH_VERIFY_STEP=1` (one run per layer with a 3 s timeout and the failing layer printed),
  `FLM_DFLASH_VERIFY_NLAYERS`, `FLM_DFLASH_STATS=1` (per layer rms / non-finite counts in step mode), `FLM_DFLASH_DEBUG=1`.
- Only the XRT runtime path is implemented (the verify context uses its own `xrt::device` handle); HRX/Windows builds fall back to the sequential verify at load (compile guard in commit; not built here).

## DEFECT FOUND: the verify xclbin stops answering after about 300 consecutive dispatches in one hardware context
Symptom: `ERT_CMD_STATE_TIMEOUT` (runlist) or `qds_device::wait() unexpected command state` (single run); after it the device refuses new contexts for ~60 s (`DRM_IOCTL_AMDXDNA_CREATE_HWCTX ... -22`) and recovers by itself.
Evidence (all deterministic):
| experiment | result |
|---|---|
| engine, fused verify, 42 layers per cycle (runlist and per-layer step mode) | dies on the 8th cycle at layer 10 (verify run #304-305), with two different prompts (ctx0 64 and 60) |
| same, only the first 20 layers per cycle | dies at verify run #300 (layer 19 of the 15th cycle) |
| 300 ms sleep between cycles | same run number (not wall time) |
| verify-only loop (no draft, no accept) | same |
| kernel module rebuilt before every layer (what the standalone harness does) | same, run #305 |
| one shared input BO and one shared output BO (no chaining) | same, run #300 |
| standalone harness `work/scripts/run_fused8.sh --layers 0 --cursors 64 --repeat 400` (no engine code at all) | `ERROR fused8: dispatch threw: qds_device::wait() unexpected command state`, ABORT |
| **hardware context rebuilt every 200 dispatches** (new `npu_xclbin_manager`, same BOs, ~3 ms) | 800 dispatches (40 cycles x 20 layers) and all later runs complete, no hang |
So it is count-based per context, independent of layer type, ctx, BO use and timing; the fused8 tests never ran more than ~300 dispatches of one xclbin in a process (hence not seen there).
Cause unknown (a counter/lock/token that drifts once per dispatch in the design is the obvious suspect: decode runs thousands of dispatches per context without it; compare with
the token/lock bookkeeping of the fused8 sequences). **Workaround in the engine**: `verify_prepare` rebuilds the verify hardware context when `runs_since_load + 42 > 200` (cost ~3 ms every 4-5
cycles). The kernel owner should find the leak; with it fixed the reload can be set to 0.

## Results (quiet box, `uptime` load < 1, real Bolt package, Q4_1 draft, local CI-equivalent xclbins; greedy)
Engine-level rig `tools/test_spec_engine.cpp` (`work/scripts/run_spec_test.sh`), plain greedy decode vs speculative decode with the fused verify:

| set | plain tok/s | spec tok/s | speedup | AL | stream equality |
|---|--:|--:|--:|--:|---|
| 12 chat prompts x 128 (ctx < 200) | 13.37 | **22.60** | **1.69x** | 3.17 | identical 4/12; divergence at a tie (margin <= 1 ulp) 5/12; beyond: 3/12 (margins 0.25, 0.25, 0.75 logit units) |
| 1017-token prompt x 160 | 12.58 | 15.41 | 1.22x | 2.91 | diverges at token 13 at a tie |
| 3917-token prompt x 160 | 11.59 | 9.31 | 0.80x | 2.18 | diverges at token 5 at a tie |
| 495 / 1117 / 2317-token prompts x 128 (before the overlap optimisation, so cycle +17 ms) | 12.7 / 12.6 / 12.1 | 15.5 / 9.9 / 8.2 | 1.22 / 0.78 / 0.68 | 3.02 / 2.02 / 1.81 | margins 0.375, tie, 1.5 |
AL on the same 12 chat prompts with the sequential verify (Phase 5): 3.09; llama.cpp (Q4_K_M target, BF16 draft, 40 prompts, 384 tokens): 3.71.
Restore scenario (checkpoint, restore, cold draft): the run after the restore is identical to plain, AL 3.14; the first run diverges at the same noise flip as prompt 0.
Compactions: prompts of 2317 and 3917 tokens cross 2048 and several compactions inside verify batches without error.

Per cycle at short context (ms): host draft 10.8, draft head 10.4 (lm_head_8 dispatch + pack/argmax), **verify 116.3** = inputs 1.5 + preparation (hidden behind the draft) 0.0 + **42 layers 102.3** +
taps 0.1 + lm_head_8 12.4, accept (8 argmax over 262144) 2.1, commit 0.03 -> about 140 ms for AL 3.17 tokens. Before the optimisations (commit 9.8, preparation 8.2 not overlapped,
draft head 12.0) the same set ran at 19.59 tok/s (1.46x). At 1k context the 42 layers cost 140 ms, at 4k 188 ms.

Check mode (sequential vs fused logits per cycle, 99 tokens): 280 rows, logits cosine >= 0.99 (typically 0.9995), argmax differs on 10 rows (3.6%); the sequential top-2 margins at those rows are 0 - 0.75.
Through FLM (`flm serve`, greedy via REST, 5 prompts x 128): speculation active (AL 3.5-4.3 on the coding prompts), 17.8 vs 11.1 tok/s over the requests (1.6x), texts equal on 2/5 (the others
diverge mid-stream at noise flips).


## Adaptive guard (`draft_ready`, env `FLM_DFLASH_GUARD=0` disables)
The standard `flm bench` (about 30 decoded tokens, low-acceptance synthetic prompt, AL 1.8-2.7) first showed the cost of long contexts: before the guard the DFlash tag decoded 2.8 tok/s at 32k
(plain 7.35), 5.4 at 16k (9.49), 8.4 at 4k (12.18). Model: a cycle costs ~125 + 0.016*ctx ms, a plain step ~74 + 0.0019*ctx ms, so speculation pays off only when AL > t_cycle/t_plain (1.9 at 1k, 2.3 at 4k,
3.6 at 16k, 4.7 at 32k). A running mean of AL (prior 2.6, weight 0.1 per cycle, reset per request; 5% below break-even tolerated; one probe cycle with weight 0.3 every 48 plain steps) switches speculation off
when it cannot win. With the guard: `flm bench` decode tok/s dflash vs base tag: 1k 12.47 vs 13.10, 2k 11.48 vs 12.78, 4k 9.91 vs 12.18, 8k 10.96 vs 11.11, 16k 9.33 vs 9.49, 32k 7.22 vs 7.35 (no gain, no big
loss; the bench prompt has low acceptance and only ~30 tokens, startup cycles dominate). Engine rig with the guard: 1017-token prompt 15.52 vs 12.51 tok/s (1.24x, 0 declines), 3917-token prompt 10.76 vs 11.60
(0.93x), 12 chat prompts 22.48 vs 13.36 (1.68x, 2 declines). The cost constants are fitted on this box (Strix, Bolt E4B) and live in `draft_ready`.

## What limits the speed
1. **Attention at context** (not the engine): the 8-token attention costs ~8x the single-token attention (K/V replayed once per token), so the verify layers grow from 102 ms (ctx < 200) to 140 ms
   (1k) and 188 ms (4k) while plain decode only goes 13.4 -> 12.6 -> 11.6 tok/s. Speedup at 1k is 1.22x (AL 2.9), below 1x at 4k. The plan gate (>= 1.6x at 1k) is met only at short context.
2. **Acceptance**: AL 3.17 on chat, 2.0-2.9 on the long-document prompts (summaries of random book text), 4-5 on code. The verify cycle of ~140 ms needs AL >= 3 for 1.5x.
3. Fixed cycle overhead outside the layers (~38 ms): lm_head_8 12.4 (9 ms stream + 3 ms pack/unpack), draft head 10.4, host draft 10.8 (not measured idle on this load: the drafter fork estimated 4-6 ms),
   accept 2.1. The NPU drafter (Phase 10) removes the host draft; the two lm_head_8 dispatches could be one 8-row dispatch if the draft head used the verify rows of the previous cycle (not possible: different inputs).
4. The verify core noise (1.4x the decode core error, bfp16 emulated mmul) flips greedy streams at ties and occasionally at margins up to 0.75 logit units (~6 ulp); long-context attention is noisier
   (global cosine 0.979 at ctx 4000 in the stage test): a flip with margin 1.5 was seen at ctx 2317. Distribution-exact sampling is unaffected in principle (the sampler sees the verify logits), but the
   output distribution is that of the verify path's slightly noisier logits.

## Remaining
- Find the dispatch-count hang in the fused8 design (see above), then drop the reload.
- Attention cost at long context (token-pair interchange, larger replay chunks, skipping MT2 staging), the compute-bound projection core (mmul<8,8,16> / epilogue scale-min), PLE overlap: all in FLM_Xclbin.
- Prefill-time draft priming costs ~190 ms at 1k ctx (once per request).
- Windows/HRX: the fused verify is XRT-only here.
- Wrapper-level logprobs/penalties are unchanged (they act on the verify logits rows through `_sample`).
