<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# gemma4e_dflash: DFlash speculative decoding for Nemotron-3.5-Bolt-E4B

Engine of `nemotron3.5-bolt-dflash:e4b` (FLM family `gemma4e-dflash`). It is a fork of `gemma4e_npu` (same Gemma-4-E4B
checkpoint, vision and audio paths unchanged) plus speculative decoding with the DFlash draft model
`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash`. `gemma4e_npu` and `gemma4e_flash` are not touched.

The DFlash tag needs `nemotron3.5-bolt:e4b` installed: the main weights, tokenizer and chat template come from that package,
the DFlash package (`Nemotron-3.5-Bolt-E4B-DFlash-NPU2`) holds only `config.json` and the draft weights `dflash_weight.q4nx`.
FLM passes the draft folder in the model config (`draft_model_path`); the wrapper calls `load_draft()`.

## The DFlash cycle

```
state : target K/V holds positions < pos0; id_last (sampled, no K/V row yet); the draft context holds positions < pos0
draft : block [id_last, mask x 7] at pos0..pos0+7 through the 5-layer draft (host, dflash_draft.cpp); K/V of each draft layer =
        its projections of c = rmsnorm(fc . [taps of target layers 5, 11, 17, 23, 35]) for the context positions (window 1024)
        plus the block itself (non-causal); drafts d1..d7 = argmax of the target lm_head over rows 1..7
verify: the target on [id_last, d1..d7] at pos0..pos0+7 in one pass (verify.xclbin + lm_head_8.xclbin) -> 8 logits rows, taps
accept: the wrapper samples row i with the user's sampler and accepts while the sample equals d_{i+1} (exact for any sampler)
commit: keep k+1 rows (context += k+1), zero the K/V rows of the rejected positions, add the taps of the kept rows to the draft context
```

## API

`causal_lm` is unchanged (the vtable must match the other engines). The engine implements the existing hooks
(`supports_speculation`, `speculate` = greedy draft + verify + accept + commit, `last_speculation_prime_us`) and non-virtual
methods the FLM wrapper reaches with `dynamic_cast<gemma4e_dflash*>`:

| Method | |
|---|---|
| `load_draft(dir)` | read `dir/config.json` and its `draft_weight` file; false = speculation stays off |
| `set_stop_tokens(ids)` | a verify batch is cut one past the first stop token inside the accepted prefix |
| `draft_ready()` | false when no draft, `FLM_DFLASH=0`, no room under `MAX_L`, the id log is out of step, or the guard says a cycle cannot win |
| `dflash_draft(last, drafts)` | up to 7 drafts; 0 = decline (state untouched) |
| `dflash_verify(last, drafts, n)` | `(n+1)` rows of `vocab_size()` bf16 logits, valid until the next call |
| `dflash_commit(n_keep)` | keep `n_keep` rows (accepted drafts + 1) |
| stats | `mean_accepted_length`, `draft_hit_rate`, `speculation_cycles`, `speculation_stats`, `speculation_timing`, `reset_speculation_stats` |

The draft context is cold after `clear_context`, `restore` and `write_state` (prefix-cache hits); it is rebuilt from the taps of
the last 1024 prompt rows at the first cycle (charged to prefill through `last_speculation_prime_us`).

## Files

| File | |
|---|---|
| `gemma4e_dflash.cpp`, `_detail.hpp`, `_sequence.*`, `gemma4e_prefill.*`, ... | the `gemma4e_npu` fork; taps captured in prefill and decode |
| `gemma4e_dflash_swa.hpp` | sliding-window K/V as a linear buffer of 2048 rows per plane (row = position - base), compacted by the host; decode and verify share it |
| `gemma4e_dflash_spec.cpp`, `_spec_state.hpp` | draft context, cycle, guard, stats; the sequential verify (8 decode passes) as fallback |
| `gemma4e_dflash_verify.cpp` | fused verify: one 42-run runlist of `verify.xclbin` on 8 rows, one `lm_head_8` dispatch for the verify rows and one for the draft rows |
| `gemma4e_dflash_rtp.hpp` | RTP addresses of the shipped `verify.xclbin` and `lm_head_8.xclbin` (CI ships only the `.xclbin`; the xclbin builds check their addresses against the same list) |
| `dflash_draft.{hpp,cpp}` | host drafter: Q4_1 weights (`fc` in BF16), AVX-512, 16 threads, fp32 activations |

The fused verify runs on both runtimes (XRT and HRX; the only runtime-specific piece is the ranged K/V flush in `gemma4e_dflash_bo.hpp`). Without it (preemption, a missing `verify.xclbin`) the tag decodes without speculation, at plain speed: the sequential verify (8 decode passes per cycle) is about 3x slower than plain decoding and runs only when `FLM_DFLASH_VERIFY=seq` asks for it.

## Environment variables

| Variable | Default | |
|---|---|---|
| `FLM_DFLASH` | 1 | 0 = no speculation (the tag behaves like `nemotron3.5-bolt:e4b`) |
| `FLM_DFLASH_VERIFY` | fused | `seq` = 8 sequential decode passes (also where no fused verify exists; debug / reference); `check` = run both every cycle and report differences |
| `FLM_DFLASH_ATTN` | x | `8` loads `verify_attn8.xclbin` (K/V replayed per token) for A/B runs, needs its `_rtp_addresses.txt` next to it |
| `FLM_DFLASH_GUARD` | 1 | 0 = always speculate (the guard stops speculating when the running acceptance length cannot beat plain decoding) |
| `FLM_DFLASH_THREADS` | 16 | host draft threads |
| `FLM_DFLASH_RESCORE_K` | 8 | the NPU draft head's top-K per row re-scored with the Q8_0 embedding rows (0 = off) |
| `FLM_DFLASH_ASYNC_INJECT` | 1 | add the verify rows to the draft context on a worker thread while `lm_head_8` and sampling run |
| `FLM_DFLASH_VERIFY_RELOAD` | 0 | rebuild the verify hardware context every N dispatches (safety switch; the hang it was added for is fixed) |
| `FLM_DFLASH_PROFILE` | off | per-phase draft timing |
| debug | | `FLM_DFLASH_TRACE`, `FLM_DFLASH_DUMP=<prefix>`, `FLM_DFLASH_DEBUG`, `FLM_DFLASH_STATS`, `FLM_DFLASH_CHECK_SWA`, `FLM_DFLASH_VERIFY_STEP`, `FLM_DFLASH_VERIFY_NLAYERS`, `FLM_DFLASH_PREP_EARLY`, `FLM_DFLASH_INJECT_CHUNK` |

Build option: `-DFLM_SWA_BUFFER_ROWS=640` makes the sliding buffer compact every ~80 tokens (stress test of the compaction).

## Tools (`tools/`, not part of the library)

| Tool | |
|---|---|
| `test_spec_engine.cpp` | prefill, then plain greedy vs speculative greedy streams (a flip at a top-1/top-2 margin of a few bf16 ulps is reported, not failed), acceptance length, per-phase timing; build line in the file header |
| `tf_replay.cpp` | teacher-forced acceptance length on fixed token sequences, and fused verify vs sequential verify |
| `test_swa_stream.cpp` | long greedy streams (thousands of tokens, several compactions) vs the unmodified `gemma4e_npu`; state save/restore round trips |
| `test_dflash_draft.cpp`, `bench_gemm.cpp`, `build_test_dflash_draft.sh` | host drafter vs `ref/draft_case.py` cases; GEMM microbenchmark |
| `ref/dflash_ref.py`, `ref/draft_case.py` | numpy/torch reference of the draft (from the BF16 GGUF or the converted package) |

## Results (Linux, XRT, quiet box; details in `FLM_Xclbin/Gemma4/verify_decoding/notes/e2e_results.md`)

| Context | Plain tok/s | DFlash, greedy, thinking off / on | DFlash, sampled (top_k 64, T 1), off / on |
|---|---|---|---|
| ~30 | 13.3 | 1.74x / 1.89x | 1.59x / 1.71x |
| 1k | 12.7 | 1.54x / 1.78x | 1.51x / 1.76x |
| 4k | 11.8 | 1.33x / 1.55x | 1.27x / 1.38x |
| 16k | 9.1 | 1.43x / 1.64x | 1.40x / 1.61x |

`flm serve`, 512 generated tokens, 3 prompts per context. The verify is not bit-identical to 8 decode steps (8-row cores round
differently): on real text its argmax agrees with sequential decoding on 97.3 % of rows, and a greedy DFlash stream can differ
from a plain greedy stream where the top two logits are within a few bf16 ulps.
