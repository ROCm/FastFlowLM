<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Stream D: acceptance length and host-side cycle time (2026-10-29)

Code: `FLM_DLL/detail/gemma4e_dflash`: drafter GEMM; async inject, draft-head re-scoring,
measured guard, tf_replay rig; verify preparation overlaps the draft head, dynamic GEMM schedule. Rigs: `tools/tf_replay.cpp` (teacher-forced / free AL), `work/scripts/run_d_seqs.py` (one sequence
per process, retries after a device reset), `work/scripts/eval_d.sh`, offline analysis `work/ref/al_analysis.py`,
`al_subvocab.py`, `al_rescore.py`, `al_rescore_q8.py`, `vocab_select.py`. Private builds: `work/dfork/build` (D), `work/dfork/build_head`
(HEAD before D, for A/B).

## 1. Where the AL gap comes from (evidence)

Teacher-forced (TF) replay = same token contexts for every variant: the 14 MT-Bench sequences of `work/ref/seqs.json` (llama.cpp greedy
continuations, **thinking on**, 400 tokens), drafts compared with the forced continuation, the same rule as `dflash_ref.replay_al`.
Engine dumps (`FLM_DFLASH_DUMP`) give the injected NPU taps and the draft hidden rows, so each factor can be swapped offline on the
identical cycles (1276 cycles):

| variant (same cycles) | TF AL | step |
|---|--:|---|
| fp32 reference draft on HF BF16 target taps | 3.734 | |
| fp32 reference draft on the NPU taps | 3.712 | (b) target taps noise: -0.6% |
| engine host draft hidden + exact fp32 head | 3.711 | (d) host draft numerics: 0 (hidden cos >= 0.9997 vs fp32 ref) |
| engine as shipped (NPU `lm_head_8` draft head) | 3.636 | (e) draft head: -2.0%; 13% of draft tokens differ from the exact head |
| emulated Q4 lm_head in fp32 | 3.697 | of the -2.0%, -0.4% is the Q4 weights, -1.6% the lm_head_8 arithmetic (bfp16 mmul) |
| NPU top-K re-scored exactly, K = 2..16 (emulated) | 3.711-3.714 | recovers all of (e) |

Other factors:
- (c) draft weights: offline sweep (S2.1) BF16 3.826 vs Q4_1 3.778 vs Q4_1 with `fc` BF16 3.829. `fc` is only used to inject taps
  (not in the per-cycle draft), so keeping it BF16 costs ~45 MB on disk and ~0.3 ms of injection per cycle, no draft time.
- (f) accept rule: the engine's TF AL tracks the reference per sequence (table in log.md); no off-by-one in positions after partial
  accepts (an error there would show as a large engine-vs-reference gap).
- (g) prompt priming: engine TF AL = reference TF AL on the same taps, so the prompt taps (incl. the last prefill row) are injected
  correctly; per sequence differences are within the head effect.
- (a) **chat template / thinking: the main reason the rig showed 3.17.** The engine rig `work/spec/ids.json` uses the chat template
  without `enable_thinking` and 128 tokens; llama.cpp's 3.71 was measured with thinking on (`--jinja`) and 384 tokens. Free-running
  greedy on the thinking-on prompts of seqs.json gives llama.cpp-like per-prompt AL (4.80, 3.81, 4.86, 4.86, 3.70, 3.24, 6.50,
  5.60, 3.84, 3.64, 2.39 on seqs 0-10; `work/dfork/free_think.log`). No engine bug behind the 3.17 vs 3.71 gap.

Rejected: a draft-head sub-vocabulary (to cut the 10 ms draft head). Frequencies from 17M corpus tokens: held-out coverage of
generations 90% at 32k ids; TF AL static 32k 3.156, static 64k 3.458, static 64k + every context id 3.630 vs full 3.711.

## 2. Changes and measured effect on acceptance

| config (TF, 14 seqs, own cycle boundaries) | TF AL | vs HEAD |
|---|--:|--:|
| HEAD (NPU lm_head_8 argmax) | 3.626 | |
| + top-8 re-scoring with the Q8_0 embedding rows (`FLM_DFLASH_RESCORE_K`, default 8) | 3.682 | +1.5% |
| + `fc` kept BF16 in the draft package (proposal, `work/dfork/pkg_fc`) | 3.745 | +3.3% |

(TF AL is deterministic to ~0.3% run to run; per sequence the cycle boundaries shift when a draft changes, so per-sequence numbers
move both ways while the mean improves.)

**Package proposal**: `configs/dflash.json` `"bf16_tensors": ["model.fc.weight"]` (the converter supports it; the host
drafter already reads BF16 blocks; checked: C++ vs fp32 reference on the same weights cos >= 0.999998, tokens 7/7 on the S5.2 cases).
Package grows 225 MB -> 270 MB. The package is not published yet, so this is the moment (D13).

## 3. Host side

- Drafter GEMM: bf16 scale/min pairs (-17% weight bytes), weight tile reused across 8-token tiles (prompt priming in
  64-row chunks), q|k|v and gate|up in one parallel region, 16 threads by default. Bit-identical outputs.
- Speculative injection: the verify rows' taps are injected on a background thread during the verify's `lm_head_8` and
  the caller's sampling; commit truncates (`truncate()`) to the kept rows. Inject wait at commit: ~0.05 ms (was ~1.4 ms on the path).
- Draft head re-scoring costs ~0.3 ms (56 Q8 rows dequantised + dotted) on top of the NPU dispatch.
- Guard: measured cycle and plain-step costs of the session replace the fitted constants once available.
- Sampling cost (FLM wrapper, per row): `Sampler::sample` top_k 64 = 0.08 ms, greedy 0.14 ms (`work/dfork/sbench`): not a bottleneck.
- Adaptive draft length: the verify is a fixed 8-row xclbin and the draft always computes 8 rows, so drafting fewer saves nothing;
  only the on/off guard matters.

## 4. Speed

The NPU was shared with the H/P/A forks for the whole session (plain decode read 4-6 tok/s instead of 13 during the final runs), so
absolute tok/s are depressed; comparisons below are between runs with the **same measured verify time** (same contention).

**40-prompt goldens set** (`work/ref/seqs.json`: MT-Bench 16 + SPEED-Bench 16 + long 4 (2-2.6k tokens) + multi 4, chat template with
thinking on, free-running greedy speculation, 256 tokens; `tf_replay free`, logs `work/dfork/q/gold_*.sum`):

| engine | cycles | AL | tok/s | ms/cycle | host draft | draft head | verify | accept | commit |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| HEAD (+H fix) | 2750 | 3.613 | 21.52 | 167.9 | 12.07 | 11.57 | 141.87 | 1.77 | 0.02 |
| D | 2686 | **3.708** | **23.05** | **160.9** | **6.66** | 11.66 | 141.49 | 0.51 | 0.02 |
| change | | +2.6% | **+7.1%** | -7.0 ms | -5.4 | +0.1 | (same) | -1.3 | |

(verify includes the 42 layers and lm_head_8; at these contexts the NPU part is unchanged by D. llama.cpp reference on the same
40 prompts: AL 3.71 greedy / 3.98 incl. the long ones at 384 tokens; the engine now matches it.)

**12 chat prompts** (`work/spec/ids.json`, no thinking, 128 tokens, engine rig, interleaved HEAD/D/HEAD/D; `work/dfork/q/chat_*.log`):

| run | plain tok/s | spec tok/s | ratio | AL |
|---|--:|--:|--:|--:|
| HEAD r1 / r2 | 4.22 / 4.97 | 10.16 / 10.72 | 2.41x / 2.16x | 3.17 |
| D r1 / r2 | 4.96 / 6.15 | 10.93 / 14.20 | 2.20x / 2.31x | 3.14 |

Too noisy under this contention to separate a 5-7% change; AL on this set is 3.17 vs 3.14 (free-running streams diverge at noise flips,
so a different trajectory; teacher-forced on fixed sequences D is +1.5%). On the earlier quiet box HEAD measured 13.33 -> 22.74 tok/s
(1.71x) on this set. Stream checks unchanged: 2-3 flips beyond ties per 12 prompts (verify-xclbin noise, same as HEAD).

Per-cycle at short context (6 chat prompts, quiet stretches, `work/dfork/ab_*.log`): host draft 10-13 -> 5.4-6.2 ms, accept 2.0 ->
0.5 ms, background injection wait 0.04 ms, verify preparation (5.7-7.5 ms CPU) hidden under the draft head; cycle ~137 -> ~130 ms.

Long prompts (495 / 1117 / 2317 tokens x 128, contention): spec 13.17 / 9.49 / 7.96 tok/s (HEAD) vs 16.47 / 12.26 / 9.86 (D) vs
16.58 / 12.74 / 10.01 (D + fc-BF16 package); the HEAD host draft was 16-44 ms per cycle at these contexts (per-cycle injection
streamed the weights with 8 threads), D 10.6-15.6 ms.

Prompt priming (first cycle after a prefill, charged to prefill): 1k prompt rows 100-140 ms in the engine; standalone injection
0.093-0.12 ms/row at 16 threads. The parallel GEMM saturates at ~570-720 GMAC/s (8+ threads, memory bound: 16 threads give the same
as 8 without pinning, `OMP_PROC_BIND=spread OMP_PLACES=cores` +25%); not pursued further (one-off per request).

## 5. Product notes
- FLM runs Bolt with `enable_think = false` unless the request sets `"think": true`; DFlash was trained and benchmarked with thinking
  on. Chat AL is ~3.2 without thinking and ~3.7-4 with it (same engine), which is the 3.17 vs 3.71 difference first reported.
- Draft-head re-scoring is controlled by `FLM_DFLASH_RESCORE_K` (default 8, 0 = the NPU argmax as before).
- Verify preparation order: `FLM_DFLASH_PREP_EARLY=1` restores the old order (before the host draft).
- Speculative injection: `FLM_DFLASH_ASYNC_INJECT=0` disables the background injection.
- `FLM_DFLASH_THREADS` (default 16 = physical cores) sets the host draft threads. On a loaded host the 16-thread draft is sensitive
  to stragglers (dynamic schedule mitigates; spikes of 18-65 ms were seen while other forks compiled / ran NPU tests).
