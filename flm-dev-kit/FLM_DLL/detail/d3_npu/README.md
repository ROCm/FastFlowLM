<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# d3_npu

Engine for the Decision 3.0 decision models of vLLM Semantic Router (`vllm-sr/d3-lite`, package
`D3-Lite-NPU2`): the Qwen3.5-0.8B text tower read once per question, with **bidirectional full attention**,
returning the final-norm hidden state of the last token. No LM head, no decode, no conversation state. The
255-way code readout, the prompt and the answer objects belong to the application (FastFlowLM's
`AutoDecisionModel`, `src/common/decision`).

Public interface: `include/decision_model.hpp` (`decision_model`), `include/models/d3/d3_npu.hpp` (`d3_npu`).

**Status: text.** Every projection runs on the array: `mm_rtm2_glu.xclbin` (`xclbins-dequant_mm_bfp16_rtm2_glu_512x512x512`, Q4_K
dequantised in the core and multiplied in native bfp16) when the package's xclbin folder has it, else `mm.xclbin`
(`xclbins-dequant_mm_512x512x512`, Qwen3.5-0.8B's). With the bfp16 kernel the host skips its bfp16 pre-rounding. conv1d and the gated
delta rule run on the host (the delta rule on one contiguous state slab); token-serial stages do not amortize a bitstream switch at d3's
prompt lengths. d3:edge uses `mm_rtm2_glu.xclbin` (prefill 1.34-1.69x faster than on `mm.xclbin` at 64-4,096 tokens, same label accuracy).
**Attention** runs on the host for prompts under 1,536 tokens and on the array from 1,536 (`attn.xclbin` =
`xclbins-Qwen3_5_bidirectional_attention_prefill`, Qwen3.5's prefill attention made bidirectional), registered on
the first such prompt; if it cannot be registered, attention stays on the host. Host attention computes QK and PV in bf16
(`vdpbf16ps`; the engine builds with `-mavx512bf16`), so host results differ from the old fp32-probability path by bf16 rounding (top-1
agreement 98.8% on a 2,172-request text sample), and are bit-identical run to run.

**Attention modes** (decision_config.json `attention_mode`): `noncausal_full_attention` (lite / nano / mini: every
token sees every token of its prompt, the default) and `prefix_bidir` (d3-turbo, see below). The engine does not read
the mode itself: the application passes `d3_image_payload::state_len` per prompt for `prefix_bidir`.

**Images** (`encode_batch(ids, d3_image_payload*)`): the request's images, shared by every prompt. The vision tower
(`d3_vision.cpp`, a copy of `qwen3_5vl_npu`'s `Qwen3_5VLImageEncoder` with its parent dependency replaced by
`config.json` values; `vision_mm.xclbin` + `vision_attn.xclbin`, `vision_weight.q4nx`) is built on the first call with
images and runs once per call; its merger rows replace each prompt's image tokens in order. Positions are Qwen3.5's 3D
M-RoPE (`get_rope_index`): an image's tokens at (s, s + row, s + col), text resuming at s + max(grid_h, grid_w) / 2.
Vision attention runs on the host (bf16, same kernel as the text) for calls of at most 1,024 patches in total (`D3_VISION_HOST_ATTN_MAX`,
default 1024, 0 = never), which saves the two ~2.7 ms hw_context switches per layer that the array attention (its own xclbin) costs: one
784-patch image 131-137 -> 89 ms. Above that the array attention is faster and stays. The rotary-applied qkv buffer is synced back to the
device in both paths (the matmul reads the device copy). A segment encoded alone and inside a larger call can sit on different sides of the
threshold, so their rows are equally accurate but not bit-identical.

## prefix_bidir attention

`encode_batch(ids, &payload)` with `payload.state_len = {S_0, S_1, ...}` (one entry per prompt, appended to
`d3_image_payload`; `payload.images` may be empty, a text-only call is fine). For a prompt of L tokens with
0 < S < L, every full-attention layer lets the first S tokens (the state block) attend bidirectionally within the state
only, and the remaining L - S tokens (the question block) attend to all L tokens of the prompt, bidirectionally. The
Gated DeltaNet layers stay causal over the whole prompt, nothing else changes. `S <= 0` or `S >= L` (and an empty
`state_len`) is the unchanged full bidirectional attention, bit-identical to a call without a payload. Upstream
reference: `d3_runtime.PromptEncoder` with `StateBlocks` / `block_attention` (two unmasked SDPA calls per row), and
`d3_format.state_token_count` for S (the application computes it; boundary tokens belong to the question).
`state_len` is read through `void* payload`, so the application and `libd3_npu.so` must be built from the same
`d3_npu.hpp`. No virtual of `decision_model` was added.

- **Host** (`d3_cpu::attention`): two calls, state x state, then question x all (`causal = false`).
- **Array** (`attn.xclbin`, the same bitstream, no mask kernel): two runs of the kernel. Run A is the state over itself
  (`Q_chunks = ceil(S / 128)`, `L_valid = S`, its own K / V staging, rows past S zero), so it is exactly a standalone
  S-token prompt and its instruction sequence depends on S only. Run B is the question rows over every key
  (`q_base = o_base = S rows`, `Q_chunks = ceil((L - S) / 128)`, `L_valid = L`, key length `ceil128(L)`): the query chunk
  count and the key length are independent in `gen_bidirectional_attention`. A must run before B: it writes whole
  128-row chunks, i.e. rows [S, ceil128(S)) of the output belong to B. Sequences are cached per call on S (run A) and
  (S, L) (run B). q / o are used in place when the segment starts at row 0 and S + ceil128(L - S) rows fit the bucket,
  else staged.
- **Array or host**: by query-key pairs, S*S + (L - S)*L (L*L unsplit) against `D3_ATTN_ARRAY_MIN`^2. The default is the
  1,536-token break-even measured for lite / nano / mini; not re-swept for prefix_bidir or GQA 6.

**GQA group size.** The array schedule runs 4 query heads per pass (one per compute unit, all sharing one kv head).
Any group size runs: a kv head's heads go four at a time and a short last pass is padded with duplicates of its own
real heads (they compute and store identical rows to identical addresses). GQA 4 is one pass per kv head, as before;
GQA 6 (d3-turbo: 24 heads, 4 kv heads) is 8 passes per layer, 4 + 2 real heads per kv head, so about 2x the 16-head GQA 4
time instead of 1.5x. The host kernel packs 16 consecutive (row, head) vectors of one kv head per task, whatever the group.
Shapes the kernel cannot run (`head_dim != 256`, heads not a multiple of kv heads) throw; they used to be asserts, which a
release build (NDEBUG) compiles out.

**Value-head count.** `linear_num_v_heads_padded` (48 -> 64 for d3-turbo) is used only for the descriptor shapes of the
stored ssm tensors; all compute uses the real count. The padded `ssm_alpha/beta` slots are filled from what the tensor
holds (zero for the rest), never beyond it; the bf16 copies the host computes with must be exactly [Hv, D]. The host
delta rule maps value head hv to key head hv / (Hv / Hqk), correct for any ratio (3 for d3-turbo).

## State-once (`encode_state_once`)

Requests with many questions about one state block (d3-turbo's contract: a state, up to 255 questions) do not
recompute the state per question. `d3_npu::encode_state_once(const d3_state_request&)` (non-virtual, appended to
`d3_npu.hpp`; `d3_npu_abi_version()` = 2, `d3_npu_capabilities()` tell a caller what the library has) runs the state block once,
keeps what a continuation needs, and runs every question block as rows that continue from it:

- **Cache** (host heap, per call, `d3_state_cache`): per full-attention layer the state's K (after k_norm + RoPE) and V, bf16
  `[S][DK]` (d3-turbo: 16 layers, 64 KiB per state token); per GDN layer the last `conv_kernel - 1` rows of the in_proj_qkv output
  (the conv input) and the fp32 recurrent state `[Hv][DK][DV]` (d3-turbo: 48 layers, 147 MiB in all, independent of S).
- **Passes**: pass 0 = the state segment (`save`) plus the questions that fit into the padding of its last 512-row tile (up to
  `pack_rows`); later passes = question segments only (`dependent`), packed greedily. Row-wise stages are unchanged (packing is
  bit-exact). Question blocks get RoPE positions `next_pos + l` on all three axes (`next_pos` = S for text, the M-RoPE planner's
  position after the last state token when the state holds images).
- **GDN**: a question's conv input is `[cached tail | own rows]` instead of zero rows and its recurrent state starts from a
  copy of the cached one (`conv1d_silu_carry`, `gated_delta_net_carry`); the cache is read only.
- **Attention**: the state segment runs as before, unsplit bidirectional (array or host by `d3_attn_work(S, 0)`), and saves K / V.
  Question blocks always run on the host: `d3_cpu::attention_prefix` scores a block against two key ranges (the cached state K^T,
  built once per layer and pass and shared by every block, and the block's own keys) with the arithmetic of `d3_cpu::attention`
  over the joint key list, so a question's rows are **bit-identical to the per-row prefix_bidir path** (`encode_batch` with
  `state_len`) when both use host attention. `attn.xclbin` has no log-sum-exp output, so a state part and an own part cannot be
  merged on the array.
- **Numerics**: bit-identical to the per-row path with host attention on both sides; independent of the other questions in the
  call, their order and the pass packing (a question's arithmetic never reads another's rows). When the state is long enough for
  array attention the state rows carry the array's known noise (hidden cosine >= 0.998 vs host).
- **Readout rows**: `d3_question_block::readout_from_end` returns the final-norm hidden state of several rows per question (M3's box
  edges); empty = the last row.
- **Measured** (Strix Halo, shared host at load 40-60, so times are noisy; synthetic requests of question blocks of 110 +- 20 tokens,
  d3-turbo plain Q4_K package, default attention, wall seconds per request; per-row = M1 `encode_batch` + `state_len`, measured for the
  first 2-4 questions and scaled where marked):

  | S | N=1 | N=4 | N=16 | N=64 | N=255 |
  |---:|---|---|---|---|---|
  | 300 | 5.0 / 6.3 | 19.2 / 9.3 (2.1x) | 53-76 / 21 (2.5-3.7x) | 305* / 74 (4.1x) | 1,220* / 318 (3.8x) |
  | 1,024 | 10.4 / 12.4 | 57 / 15 (3.7x) | 243* / 26 (9.4x) | 917* / 88 (10.4x) | |
  | 4,096 | 51 / 40 | 202* / 54 (3.7x) | 635* / 73 (8.7x) | 3,505* / 110 (32x) | |

  (per-row / state-once; * = scaled from the measured questions; first-call figures at N=1 include one-time sequence setup in both modes).
  Peak resident of the process: 17.85 GB with and without state-once (S = 300, N = 255), 18.0 GB at S = 4,096 (the cache is 0.4 GB).
  Equivalence on the same runs: bit-identical to the per-row path whenever both use host attention (all S <= 1,024 runs, and S = 4,096 with
  `D3_ATTN_ARRAY_MIN=0`); with the default thresholds at S = 4,096 the per-row question rows run on the array and M2's on the host:
  cosine median 0.9995 (min 0.934 on one synthetic state where the array attention is itself 0.958 from the host result; M2 is closer to the
  host result than M1 in every compared row).
- **M3 / M4 additions (ABI revision 2, minor 1: `d3_npu_abi_minor()`, capabilities `D3_CAP_QUESTION_IMAGES | D3_CAP_VIDEO`):**
  - *Images inside question blocks* (`question_images` + `d3_question_block::image_ids`): the vision tower runs once per distinct image
    (entries with equal grid and pixels are merged by hash + memcmp, so an option image used by 100 questions costs one run), before the first
    pass; the merger rows replace the block's pad rows and the block's M-RoPE positions come from the planner started at the state's next
    position (`d3_positions.hpp`: each run `(cur, cur + row, cur + col)`, then `cur += max(gh, gw) / 2`; same as the state's images).
    Bit-identical to the per-row path (`encode_batch`, the state's and the block's images as the prompt's payload, `state_len`) with host attention.
  - *Video*: `video_token_id` makes `<|video_pad|>` runs visual runs like `image_token_id` runs (state and blocks). A video's temporal group is one
    `{grid_h, grid_w}` entry of `media->images` (pixels `[3][2][16][16]` per patch, 2 x 2 merge order); the vision tower needs no change (per-group
    attention, two-frame patch embedding). Rewriting the video pad ids to the image pad id (`video_token_id = -1`) gives bit-identical rows.
  - *Vision tower calls*: at most 65,536 patches (16,384 merged tokens) per call, whole segments (`D3_VISION_MAX_PATCHES`); a call costs about 0.45 s of
    fixed setup plus about 0.37 ms per patch (a 32-frame window of 16 groups x 784 patches: 4.7 s); one segment of 12,544 patches takes 9-11 s (attention is
    quadratic per segment), so many groups are cheaper than one big image; a segment costs about 14 ms on top (256 segments of 16 patches: 5.2 s). Segments of at
    least 576 patches gave bit-identical rows alone and in a call with others in all 25 checked; shorter ones (below the 512-key attention chunk) do not always:
    1 of 6 at 256 and 400 patches, 6 of 8 at 64 patches differ by 2-5 % relative L2, with the same error against an fp32 reference (a joint call is not an
    approximation of the single ones, both are equally close to the reference). 100k patches in one call (128 groups) work (34 s); the 65,536 limit is policy.
  - *Several states* (video windows) are several calls; a window of 3.4k tokens plus 15 moment blocks of 113 tokens (10 GEMM tiles) took 61 s on the loaded
    host (5.5 s per 512-row tile, 4.7 s of it vision), peak resident 19.5 GB over 8 windows with no growth.
  - Debug environment: `D3_VISION_TIMING=1`, `D3_VISION_DUMP=<file>`, `D3_VISION_FEED=<file>` (take the merger rows from a dump instead of the tower),
    `D3_POS_DUMP=<file>` (every segment's positions).

## Sizes

One engine, every size: the dimensions come from the package's `config.json`, the kernels are Qwen3.5's shared ones
(`mm.xclbin` is identical across sizes), so a new size is a package plus a catalog entry (`kernel_assignment`).
`d3:lite` (0.8B), `d3:nano` (`vllm-sr/d3-nano`, Qwen3.5-2B text tower, 24 layers, hidden 2,048), `d3:mini`
(`vllm-sr/d3-mini`, Qwen3.5-4B text tower, 32 layers, hidden 2,560, 16 key / 32 value linear-attention heads) and `d3:turbo`
(Qwen3.5-27B, below) are in use (`d3:ultra`, a turbo fine-tune, shares turbo's engine and kernels; `d3:flash`, the Qwen3.5-9B size, uses the unmodified engine with turbo's kernels; `d3:edge`, the 12-layer Qwen3.5-0.8B size, has full attention at layers 2, 5, 8, 9, 10 and 11, which the engine reads from `layer_types`). Nano and mini on the unmodified library (packages from the repo's GPTQ step; Strix Halo):

| | `d3:lite` | `d3:nano` | `d3:mini` |
|---|---:|---:|---:|
| 330 tokens x 1 question (ms) | 159 | 252 | 536 |
| 128 tokens x 16 questions (ms) | 638 | 1,110 | 2,386 |
| 32,768 tokens x 1 question (s) | 22.7 | 28.4 | 66.5 |
| model load (s) | 0.8 | 1.5-1.8 | 2.0-3.4 |
| peak resident at 32,768 tokens | | 5.3 GB | 8.6 GB |
| array attention from (tokens) | 1,536 | 1,536 (measured break-even ~1,550) | 1,536 (measured break-even ~1,536) |
| cosine vs BF16, final norm (median) | see the lite PR | 0.9982 (own GPTQ) | 0.9994 (own GPTQ) |

Mini is the first size whose linear-attention layers have more value heads than key heads (32 vs 16): the host delta
rule maps value head `hv` to key head `hv / (Hv / Hqk)` (the HF order). llama.cpp's GGUF and the q4nx converter keep the
value heads in a different order (tiled), which the converter undoes for 4B and up; a package built some other way must
store them in the HF order too.

d3-turbo (`vllm-sr/d3-turbo`, Qwen3.5-27B text tower: 64 layers = 48 linear + 16 full attention, hidden 5,120, FFN 17,408,
24 query / 4 kv heads = GQA 6, 16 key / 48 value linear-attention heads) runs on the same engine with `prefix_bidir` and
`encode_state_once` (library revision 2, minor 1; capabilities `D3_CAP_STATE_ONCE`, `D3_CAP_READOUT_ROWS`, `D3_CAP_QUESTION_IMAGES` and `D3_CAP_VIDEO`
= 15) Kernels: `mm.xclbin` and `attn.xclbin` as for the other sizes, `vision_mm.xclbin` /
`vision_attn.xclbin` of `qwen3.5:9b` (the 27B vision tower has the same width 1,152 and 27 layers; its merger outputs 5,120).
Local measurements (Strix Halo, host shared with other jobs, so times are upper bounds):

| | plain Q4_K | own GPTQ |
|---|---:|---:|
| package | 15.5 GB | 15.5 GB |
| text, 736 questions: cosine median (min) vs the iGPU BF16 `prefix_bidir` goldens | 0.9964 (0.963) | 0.99896 (0.981) |
| top-1 agreement | 91.03% | 95.65% |
| probability error median / max | 0.020 / 0.554 | 0.0107 / 0.502 |
| long prompts 6.2k-24.2k (5): cosine median, top-1 | 0.9946, 5/5 | 0.99802, 5/5 |
| images (60): cosine median (min), top-1 | 0.990 (0.942), 91.67% | 0.99414 (0.978), 91.67% |

The image rows are limited by the stock vision kernels (the NPU vision features are at relative L2 0.31 from the golden, half of
it in one massive-activation channel; d3-mini's shipped kernels measure 0.58 with the same method). Per layer the error grows
smoothly with depth (cosine 0.9995 at layer 0, 0.991 at layer 43, worst 0.978 at layer 50, 0.9993 at the final norm): quantization
noise, no engine step. Limits: 32,767 and 32,768 tokens accepted, 32,769 refused with `length_error`; load 8-17 s; peak resident
17-18.5 GB below 8k tokens, 25.0 GB at 32,768 tokens; the pass floor is 4-5 s (weight streaming) plus 7-12 ms per row. First
question TTFT (one question, loaded box): 330 tokens 5.5 s, 1k 11.9 s, 2k 23.4 s, 4k 42 s, 8k 84 s, 16k 186 s, 32k 313-430 s.
`D3_ATTN_ARRAY_MIN` stays 1,536 (the break-even is inside the +-30% noise). State-once speedups over per-row are in the
State-once section (up to 32x at S = 4,096 with 64 questions).

## Prompt length

Prompts up to the engine's `max_length` (default 32,768, FastFlowLM's limit). Buckets up to 4,096 rows (or the pack
rows) keep their scratch and GEMM instructions between calls; a longer prompt's are allocated for its call and
freed after it (`d3_prefill_context::trim`). A pass whose up/gate GEMM result would reach 2^31 bytes (the XRT shim cannot
allocate such a buffer: d3-turbo's is 2.28 GB at 32,768 rows, the limit is reached at about 30.7k rows) runs that GEMM as two
column parts (`d3_upgate_pairs`; `D3_BO_LIMIT_MB=<n>` lowers the limit for tests, the result is bit-identical). One question, Strix Halo:

| tokens | 512 | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 | 32,768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention on the host (ms) | 220 | 428 | 906 | 2,079 | 5,566 | 21,142 | 107,017 |
| this engine (ms) | 220 | 428 | 765 | 1,587 | 3,690 | 8,425 | 22,696 |

The array kernel alone: 7.5 ms per layer at 2,048 tokens, 29 ms at 4,096, 1.8 s at 32,768 (six attention layers).

## What is different from the chat engines

| | `qwen3_5vl_npu` (chat) | `d3_npu` |
|---|---|---|
| full-attention layers | causal | **bidirectional** (every query sees every key of its prompt; GDN layers stay causal) |
| state | KV cache + GDN state carried across calls | **none**: every prompt starts from zero conv / recurrent state |
| output | logits of the next token | final-norm hidden state of the last token |
| batching | one sequence | **packed batches**: several prompts' rows in one pass of up to 2,048 rows, four 512-row GEMM tiles (`set_pack_rows`) |

The prefill is `minicpm_v_4_7_npu`'s (same text tower, same weight layout, same host kernels), restructured around
`d3_batch`: row-wise stages (projections, norms, gates, GLU, residuals) run once over all rows of a pass;
conv1d, the delta rule and attention run per segment. Upstream reference: `d3_runtime.PromptEncoder`;
plain-PyTorch reference: not shipped with the dev kit.

## Numerical contract

- **Packing is bit-exact**: a prompt's row is the same whether it ran alone or packed with others, in any order
  (checked when this was written). Order independence and determinism are bit-exact.
- **Host build** (`PREFILL_MM=cpu`) reproduces a PyTorch model fed the same dequantized weights to cosine ≥ 0.9999
  at every checkpoint.
- **NPU build**: the array's bf16 MAC is bfp16-emulated and truncating, which is the engine's error floor (cosine
  ≈ 0.998 at the final norm against the same reference). Two host-side measures reduce it and are on by default:
  every GEMM input is rounded onto the bfp16 grid first (`d3_cpu::bfp16_round_rows`), and the GDN gate's silu runs
  on the host instead of fused into the projection (`rtp_act = 0`).
- **Array attention** (prompts of 1,536 tokens or more) keeps scores, max and exp in bf16 (host: fp32). Against host
  attention on the 1,078 Decision Index (DI2) questions of 1,024+ tokens, both run on the array: hidden cosine median
  0.9996 (min 0.9981), same top answer 97.8%. Neither is closer to upstream BF16: top-1 agreement with it 88.2% (array)
  vs 87.6% (host), gold accuracy +0.46 pt (95% CI −0.08 … +1.01).

## Build

```bash
make -j            # libd3_npu.so (+ copied to ../../build/lib)
make harness       # + ../../build/d3_npu/bin/d3_engine_run (tools/d3_engine_run.cpp)
make PREFILL_MM=cpu   # every projection on the host too: no xclbin, no device work
make RIG=1            # named checkpoint dumps (D3_RIG_DUMP=<dir>), named as the PyTorch reference names them
make PREFILL_PROFILE=1   # per-stage prefill times
make FLM_USE_HRX=1    # against the HRX runtime (hrx-integration/fetch-hrx-release.sh first)
```

## Run-time switches

| variable | effect |
|---|---|
| `D3_RIG_DUMP=<dir>` | (`RIG=1` builds) write each named checkpoint of a single-prompt call as `<name>.npy` |
| `D3_BFP_ROUND=0` | NPU builds: skip the bfp16 pre-rounding of GEMM inputs (for rig comparisons) |
| `D3_HOST_GEMMS=gdn_o,attn_o,down` | NPU builds: run those projections (the ones whose input is a gated product) on the host |
| `D3_BO_LIMIT_MB=<n>` | tests: largest device buffer one array run may write (default 2^31 - 16 MiB); lowering it forces the two-part up/gate GEMM |
| `FLM_PREFILL_THREADS=<n>` | host threads (default: one per physical core) |
| `D3_PAR_SPIN_US=<us>` | microseconds an idle host worker spins before it sleeps (default 30; `0` = sleep at once). The host loops run on `d3_par.hpp`'s own pool, not OpenMP (FLM issue #882: OpenMP workers spin through every NPU wait) |
| `D3_ATTN_ARRAY_MIN=<tokens>` | NPU builds: prompts whose attention work (query-key pairs) reaches that of a prompt of this many tokens run attention on the array (default 1,536 = prompts of 1,536 tokens or more when unsplit); `0` = never |

## Harness

`d3_engine_run <pkg_dir> <xclbin_root> <prompts.txt> <out.bin> [--pack N] [--group] [--reps R] [--max-length N] [--state-file F]`:
token-id prompts in (`<group> <id> <id> ...` per line), bf16 hidden rows out, CSV timing per call on stdout.
`--state-file` holds one state length S per prompt (0 = none): prefix_bidir through `d3_image_payload::state_len`.
`--state-once` (with `--group` and `--state-file`) sends each group (prompts sharing their first S tokens) through `encode_state_once`
instead; same output layout, so both modes' `out.bin` compare directly (`tools/compare_hidden.py`; synthetic groups from
`tools/make_state_once_prompts.py`).
`<xclbin_root>` holds `xclbins/D3-Lite-NPU2/mm.xclbin`.

`d3_req_run <pkg_dir> <xclbin_root> <list.txt> <out_dir> [--row-check K] [--rewrite-video] [--dup-qimages]` (`make req_harness`): whole requests from files
(`tools/req_file.py` writes them: state ids, state images / video groups, question images, questions with readout rows and `image_ids`) through
`encode_state_once`; output `<stem>.out` (bf16 rows). `d3_vision_bench <pkg_dir> <xclbin_root> 16x28x28[,...] [--check K] [--dump P]` (`make vision_bench`) times the
vision tower on random segments and compares a segment's rows alone and in a call.

## Tests (`tools/`)

| tool | needs | what it checks |
|---|---|---|
| `make test_cpu` (`tools/test_prefix_bidir_cpu.cpp`) | CPU | host two-call attention against a dense fp64 block-masked reference, GQA 1 / 4 / 6, S = 1, 127, 128, 129, L - S = 1, odd lengths (tolerance: bf16 rounding of the output, 4.2e-3 relative); state rows unchanged when question K / V change; S = 0 and S >= L are one unsplit call bit for bit; a query's result does not depend on the task shape; bad shapes throw. Gated delta rule against a naive fp64 version at 1, 2, 3 value heads per key head (16 key / 48 value heads), `_cpu_mm` with 48 outputs. 207 checks. |
| `make state_once_test` (`tools/test_state_once_cpu.cpp`) | CPU | `attention_prefix` bit-identical to the question rows of `d3_cpu::attention` over `[state; own]` keys (GQA 1 / 4 / 6, S = 0, 1, 15, 16, 17, 127, 128, 129, 4096, Lq = 1, 16, 17, 110) and within bf16 rounding of a dense fp64 reference; block results independent of scratch history; `conv1d_silu_carry` and `gated_delta_net_carry` (16 / 48, 16 / 32, 2 / 4 heads, states of 1 .. 600 rows) bit-identical to one pass over `[state \| question]`, including the tail and recurrent state handed on. 844 checks. |
| `make attn_array_test` (`tools/test_attn_array.cpp`) | NPU, attn.xclbin | `d3_attn_block::_array_attention` (one run unsplit, two runs prefix_bidir) against the host kernel for GQA 4 (16 heads) and GQA 6 (24 heads), (S, L) from (1, 129) to (4200, 4300) and unsplit 4,500, packed segments (staged q / o path), q ~ 0.6 * a visible key + noise so the softmax is peaked and a mask leak shows. Pass: cosine >= 0.999 per state / question part. Measured: cosine 0.9998-0.99999, relative RMS error 1.5-2.4% (the array's bf16 scores, the same as the unsplit path; it grows with L), state rows bit-identical to a standalone run of the state, repeated runs bit-identical. Prints array and host time per layer. |
| `make mm_big_test` (`tools/test_mm_big.cpp`) | NPU, mm.xclbin | the 32,768-row up/gate GEMM of a 27B-shaped layer (K 5120, N 34816, result 2.28 GB, past 2^31 bytes) on a synthetic weight, rows at the start, around 2^31 bytes and at the end against the host kernel. |
| `make positions_test` (`tools/test_positions_cpu.cpp`, `tools/fixtures`) | CPU | the M-RoPE planner (`d3_positions.hpp`) against transformers `get_rope_index`: five prompts of the FLM video agent (full positions) and 327 real prompts of the M3 / M4 golden sets (`gen_position_fixtures.py`: run starts / ends, resume points); the state planned alone plus the question block planned from the state's next position (images in blocks, video groups) equals the one-pass plan token by token. 1,015 checks. |
| `tools/check_prefix_bidir.py` | NPU, BF16 checkpoint, venv with torch + transformers | end to end: final-norm hidden states of `d3_engine_run --state-file` (host and array attention) against the upstream `PromptEncoder` forced to prefix_bidir, for prompts of 120 to 3,059 tokens and S = 1, 127, 128, 129, L / 2, L - 1; also the noncausal comparison of the same prompts, and that the engine follows the mask where it moves the reference. Runs on a checkpoint that was not trained with the mask (d3-mini): the probabilities are meaningless there, the hidden states are comparable. |

## Gotchas found bringing it up

- A host write to a BO must be flushed (synced) before the array writes that BO.
- XRT `buffer` copies do not own their memory: move an owning buffer, never copy it.
- The fused SiLU in `dequant_mm` is wrong for negative gate values.
