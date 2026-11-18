<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Gemma4 E4B 8-row verify layer (`verify.xclbin` of `nemotron3.5-bolt-dflash:e4b`)

The verify pass of DFlash speculative decoding for Nemotron-3.5-Bolt-E4B (a Gemma-4-E4B fine-tune). One cycle drafts 7 tokens
on the host, then runs the target model once on the 8 rows `[last token, 7 drafts]`. This folder builds the target's decoder
layer for those 8 rows: the same layer as the fused decode xclbin (`../decoding`), with a batch of `L_TOKENS = 8` token rows
instead of one. Every layer weight is streamed once per verify for all 8 rows, which is where the speedup comes from.

The engine is `FLM_DLL/detail/gemma4e_dflash` (see its README). The 8-row vocabulary projection is `../lm_head_8`.

## Build

```shell
source ../../setup.sh                         # the AIE toolchain (FLM_Xclbin/README.md); CI pins Peano
make bitstream MODEL_TYPE=GEMMA4_E4B          # the shipped design -> build/GEMMA4_E4B_fused8x_full_L8/xclbins/fused8.xclbin
make host                                     # run.exe, the host tests below
```

The xclbin is deployed as `verify.xclbin` (`FLM_Xclbin/kernel_assignment.md`, artifact
`xclbins-Gemma4_verify_decoding_GEMMA4_E4B`). The build compares its RTP addresses with `rtp_expected/fused8x_GEMMA4_E4B.txt`
and fails if they moved. The engine compiles these addresses in (`gemma4e_dflash_rtp.hpp`) because CI ships only the
`.xclbin`, so update both files together.

| Option | Values | Meaning |
|---|---|---|
| `DESIGN` | `fused8` (default) | the shipped 8-row layer |
| | `layer` | the decode layer copied from `../decoding`; with `L_TOKENS=1` it builds the decode xclbin (regression build) |
| | `projonly`, `projstage`, `rmsglu`, `ple8`, `attn8`, `attnx`, `attn8attn`, `attn8rope` | stage test designs, each with its own host test |
| `L_TOKENS` | `8` (default), `1` | rows per dispatch (the stage designs and `fused8` need 8) |
| `FUSED8_ATTN` | `x` (default), `8` | attention that reads every K/V row once per verify (`attnx`), or the first version that replays K/V once per token (A/B only) |
| `FUSED8_STAGE` | `full` (default), `nople`, `attn` | debug cut-offs that end the layer early; they do not drain the pipeline, never dispatch them in a loop |
| `ENABLE_CHESSCC` | `0` (default), `1` | Peano (shipped) or Chess |

Builds go to `build/<MODEL_TYPE>_fused8<ATTN>_<STAGE>_L<L>` (`fused8`) or `build/<MODEL_TYPE>_L<L>` (others), so they do not
overwrite each other.

## Design

The tile map is the decode layer's (16 Q4 projection cores in columns 0, 1, 6, 7; attention in column 2; norms, RoPE, GLU and PLE
in columns 3-5), changed only where 8 rows do not fit. Allocation table: `notes/fused8_allocation.md`.

| Stage | 8-row version |
|---|---|
| projections (qkv, o, up/gate, down) | 8-token Q4_1 core `kernels/q4_l8.h`: unpacks each weight block once and runs `aie::mmul<8,8,8>` with the tokens as rows; x replayed by a memtile phase machine (MT3), y de-permuted per head and token (MT4); DDR weight layout unchanged (shared with decode) |
| norms + residual | `kernels/rms_residual_l8.cc`: the four Gemma sandwich-norm phases with the residual `h[8][2560]` (40 KB) held in the L1 of the coreless tile (3,3); GLU moved to (4,4) |
| GLU | `kernels/glu_l8.cc` in rounds; the `[8][10240]` hidden lands in MT3 and is replayed for the down projection |
| q/k norm + RoPE | per token, 8-position cos/sin table from the host; q parked in MT4 |
| attention (`attnx`) | `kernels/attnx_score.cc` / `attnx_pv.cc`: K/V streamed from DDR once per verify, K multicast to 2 score tiles, V to 4 PV tiles; per-row causal / window masks; online softmax that rescales only when the running max grows by more than 8 |
| PLE | per-layer input `pli` computed on the host for the 8 tokens; gate (4,2), up (5,2), norm (5,3) with `kernels/ple8_*.cc` |

Engine contract (details in `notes/fused8_results.md`, `notes/attn_longctx.md`, `notes/swa_linear_buffer.md`):

- Kernel arguments: `x` BO (`[8][D]` rows, the RoPE table of the layer kind, `[8][256]` pli, `w_post_pli`, `layer_scalar`), the
  layer's weight BO, the norm BO, an output BO for the `[8][D]` result, the K/V cache.
- K/V: global layers keep row = position; sliding layers keep a linear window buffer (row = position - base, compacted by the
  host). The 8 new rows are appended before the replay. Rows past the context in the last replayed chunk must be finite (the
  engine keeps them zero).
- Every token-issuing shim BD has exactly one wait. One unconsumed task-completion token per dispatch fills the shim's queue
  after about 255 dispatches and the array stops answering (`notes/hang_rootcause.md`); the host sequences carry a token ledger
  that throws when a stream is unbalanced.

## Host tests (`run.exe -m GEMMA4_E4B -t <test>`)

The tests read the model from `q4nx_files/Gemma4-E4B-IT-NPU2/` (`host/models.h`): link it to the downloaded `nemotron3.5-bolt:e4b` package (`Nemotron-3.5-Bolt-E4B-NPU2`), whose weights the DFlash verify was measured with.

| Test | What it checks |
|---|---|
| `verify8` | one 8-row dispatch vs 8 sequential dispatches of the decode xclbin (the oracle), per layer type and cursor; `--poison`, `--repeat`, `--mode oracle` validates the harness itself |
| `fused8` | the fused layer vs the oracle on layers 0 / 5 / 40 / 35 (SWA, global, SWA-skip, global-skip) and a cursor sweep |
| `hammer8` | N consecutive dispatches in one hardware context, bit-identical results, latency min / p10 / median |
| `chain8` | all 42 layers on 8 rows vs 8 x 42 decode dispatches |
| `proj8`, `rmsglu`, `ple8` | the stage designs vs numpy references (`tools/*_ref.py`) |
| `attn8` | the attention stage designs, built separately from `host_attn8/` |

## Results (Peano, quiet box)

| | value |
|---|---|
| whole layer vs 8 sequential decode dispatches | 88 / 88 cases pass (cursors 0-4000, global layers also at 8k and 16k), cosine >= 0.9988 |
| 42 layers on 8 rows | 142 / 152 / 173 ms at context 1k / 4k / 16k (one decode token through the same rig: about 110 ms) |
| soak | 21,000 dispatches in one hardware context, no difference, no hang |
| real text, teacher-forced (engine) | argmax agrees with 8 sequential decodes on 97.3 % of rows (99.5 % outside near-ties), logits cosine 0.9995 |

End to end through `flm serve` (512 generated tokens, speedup over `nemotron3.5-bolt:e4b`): 1.6-1.9x at short context,
1.5-1.8x at 1k, 1.3-1.6x at 4k, 1.4-1.6x at 16k (`notes/e2e_results.md`).

## Notes

`notes/` holds the development record: the allocation table, per-stage results, the hang root cause, the attention redesign,
the end-to-end measurements. Script and log names in them refer to the development workspace and are not in this repo.
