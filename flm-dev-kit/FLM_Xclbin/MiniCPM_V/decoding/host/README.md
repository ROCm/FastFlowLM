<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# Decode speed test (`make test`)

Runs the **24 decode layers of Qwen3.5-2B** on the NPU the way the FLM_DLL engine runs them, with a random
10K context, and times one decode step (token 10,241).

```bash
cd FLM_Xclbin/MiniCPM_V/decoding
source ~/setup.sh                 # only needed for make bitstream
make bitstream                    # build/QWEN3_5_2B/xclbins/layer.xclbin (MODEL_TYPE defaults to QWEN3_5_2B)
make test                         # 10K context, 20 timed steps after 3 warmup steps
make test ARGS="--per-layer"      # also time one attention layer and one GDN layer
make test ARGS="--len 128"        # another context length
make test ARGS="--check"          # correctness: every layer vs the fp32 CPU reference (~3 s)
make test ARGS="--check --kv-scale 8"   # same, with peaky attention scores
make build/story_ids.txt          # token ids of the story text (~13K), via tools/tokenize_ids.cpp
make test ARGS="--prompt-ids build/story_ids.txt --check --kv-quant all"   # real 10K context (~4 min) + 8-bit KV test
make test ARGS="--repeat 10"      # determinism: same step 10x from one saved state, both run styles
make test ARGS="--lm-head-in-layer"   # lm_head as a 25th run in layer.xclbin: logits vs CPU, then speed
```

Options (`./run.exe --help`): `--model-path` (default `/path/to/models/Qwen3.5-2B-NPU2`), `--xclbin`,
`--len` (cached tokens, default 10240), `--iters`, `--warmup`, `--per-layer`, `--check`, `--kv-scale`,
`--prompt-ids`, `--kv-quant none|int8|fp8|all`, `--repeat`, `--gap-ms`, `--lm-head <lm_head.xclbin>`,
`--lm-head-in-layer`, `--no-proj-mode`, `--seed`.

Output: ms/token and tok/s for the 24 layers (**no lm_head**), a check that the output is finite and non-zero,
and with `--per-layer` the time of one attention layer and one GDN layer.

## How it works

- `qwen3_5vl_model_def.hpp`, `qwen3_5vl_npu_sequence.hpp/.cpp` are copies of the engine's files
  (`FLM_DLL/detail/qwen3_5vl_npu/`, commit d8064aaf0), with only the full-engine header include removed.
  They hold the weight layout (Q4_K reorder) and the DMA instruction sequences. Change them here together with
  the kernel; the engine copy is shared by all Qwen3.5 sizes.
- `host.cpp` mirrors `qwen3_5vl_npu::Impl`: the same buffers per layer, `desc.load_attn_weights` /
  `desc.load_linear_weights`, `gen_set_context` + `gen_layer_seq`, and one run list per token
  (`set_layer_rtp` + 24 layers). RoPE cos/sin is set for position `len`.
- **Random context instead of a prefill:** each attention layer's K rows and V rows `0 .. len-1`
  (`len x DK`, `len x DV` bf16, uniform in [-1, 1]) are random; each GDN layer gets a random conv cache (bf16)
  and recurrent state (16 x 128 x 128 fp32, uniform in [-0.05, 0.05]). The input hidden state is random.
  The speed does not depend on the values, so no prefill is needed.
- Every step decodes the same token at the same position: the new K/V row `len` is simply rewritten.
- The program builds against `FLM_DLL/include` and FLM_DLL's q4nx reader (`makefiles/host.mk`), with the
  engine's compile flags. `../../host_common` is not used: it has no Q4_K support.

## Things that matter

- **"First linear layer" means every layer right after an attention layer** (`layer % 4 == 0`: 0, 4, 8, ...),
  not only layer 0. An attention layer writes the hidden state back to DDR at its end; the next GDN layer's
  sequence (`gen_layer_seq(..., is_first = true)`) sends it in again. Using the plain linear sequence for
  layer 4 left it waiting for input forever: `runlist failed execution (ERT_CMD_STATE_TIMEOUT)`.
- `--per-layer` runs each layer as its own run with a host wait, so its numbers add up to a bit more than the
  run-list number.
- The old standalone test (Q4_1 weights, one fixed layer) was removed
  from this copy; the original is still in `../../../Qwen3_5/qwen3_5_decoding/host/`.

## Correctness check (`--check`)

`cpu_ref.hpp/.cpp` is an fp32 CPU version of one decode step of a GDN layer and of an attention layer
(Hugging Face Qwen3.5 math), with the **same weights**: Q4_K / Q8_0 chunks decoded from `model.q4nx`, so
there is no quantization difference, only NPU rounding (bf16 activations, LUT exp, ...).

- Each layer runs on the NPU **alone**; afterwards its outputs are read back and compared with the CPU:
  GDN layer: the new recurrent state S (fp32) and conv cache; attention layer: the new K and V rows and the
  layer output x.
- The hidden state only leaves the NPU after an attention layer, so inside a block of 4 layers the CPU
  chains its own x; at each block start (layers 0, 4, 8, ...) it restarts from the NPU's x. Errors therefore
  grow a little inside a block and reset at the next one.
- Metrics: cosine, relative L2 error, max abs difference. `<-- low` marks cosine < 0.999.
- `--kv-scale N` widens the random K cache to [-N, N] so attention scores spread over about +-0.75 N:
  peaky attention, which is where softmax mistakes show.
- Conventions checked by the passing result: Q4_K w = S s q + M m (M stored negated), Q8_0 column-major,
  norms already include the +1, `ssm_a` = -exp(A_log), q_proj rows = 8 q heads then 8 gate heads, conv
  weights [4][6144] (row 3 = current token), conv cache rows oldest first, state S [head][dk][dv],
  K/V row t at t * 512 (V at the middle of the buffer), half-split RoPE on the first 64 dims.

- **Real context (`--prompt-ids FILE`):** instead of random K/V and state, the first `--len` tokens of FILE are
  decoded one by one on the NPU (CPU embedding from the Q8_0 lm_head table, the same 25-run list per token as the
  engine), which builds a real KV cache and real GDN states (10K tokens: ~3.7 min). Token `len` is then checked.
- **8-bit KV simulation (`--kv-quant`):** for each attention layer the CPU also runs attention with the cached
  K/V (and the new row) rounded to int8 (symmetric, one scale per token per KV head of 256 values) or fp8 e4m3
  (same scaling), and prints that against the bf16-KV CPU result: `attn core` (before o_proj) and `x out`.
- **The NPU layers of the checked token run back to back**, all CPU work comes after. See "Pauses inside a token".

## Pauses inside a token can corrupt the decode layer

`--repeat N --gap-ms G` restores one saved state N times, runs one decode step, and compares all caches and x
bit for bit. With both run styles and no pause: always identical. With a pause between per-layer runs:
1 and 10 ms identical; **50 ms: 4 of 5 repeats differ, 200 ms: 5 of 5** (everything from some layer on).
The design keeps state on the chip from one layer run to the next (x between GDN layers, cores' ping/pong
positions). During these tests another process (the user's other session, `dequant_mm.xclbin`) was using the
same NPU, and the NPU time-shares the whole array between processes: in a long enough pause the other design is
loaded and our on-chip state is wiped. Pauses between tokens are harmless because each token starts clean.
Not yet re-tested with the NPU to ourselves (idle alone may be harmless). Check `fuser -v /dev/accel/accel0`
before trusting a test. The engine runs a token as one run list; its per-layer preemption mode shares this risk.
The first version of `--check` did CPU work between layers and gave random failures (zero K/V rows, bad conv
caches) about 1 run in 5.

Baseline errors (Peano build of this folder, 10K context, seed 1234), worst per checkpoint type:

| checkpoint | kv-scale 1 | kv-scale 4 | kv-scale 8 |
|---|---|---|---|
| GDN state S | 0.999962 | same | same |
| GDN conv cache | 0.999455 | same | same |
| attention new K / V | 0.999036 / 0.999073 | same | same |
| attention output x | 0.998865 (layer 3) | 0.998799 (layer 3) | **0.997611 (layer 7)**, 0.998133 (layer 3) |

A kernel change must not make any of these worse (same seed, same options).

## Results (2026-11-02, Krackan test machine, power mode Performance, Peano build of this folder)

| cached tokens | 24 layers ms/token | tok/s (layers only) | attention layer | GDN layer |
|---|---|---|---|---|
| 128 | 16.66 | 60.0 | 607 us | 809 us |
| 4,096 | 19.01 | 52.6 | 954 us | 809 us |
| **10,240** | **22.17** | **45.1** | **1,509 us** | **823 us** |

The slope, 0.545 us per cached token, matches the end-to-end harness (0.553). End to end with the same
`layer.xclbin` is 35.49 ms/token at 10K, so lm_head + host work is about 13.3 ms per token.

## lm_head inside layer.xclbin (`--lm-head-in-layer`)

Plan and kernel side: `../PLAN_LM_HEAD_IN_LAYER.md`. The projection engine (16 cores) computes lm_head as one more
run of `layer.xclbin`, so a token no longer switches to `lm_head.xclbin` and back.

**One lm_head run** (`gen_lm_head_in_layer_seq`): mode RTPs = lm_head on the 16 projection cores and the RMS
core; x (the last layer's output, arg 0) and `model.norm.weight` (arg 2) to the RMS core; one weight transfer per
shim channel (arg 1); 248,320 bf16 logits come back on shim column 1, channel 0 (arg 4), in vocab order.

**Weight layout** (`reorder_cpy_lm_head_in_layer`, 588 MB instead of 540 MB):
- The cores' weight buffers hold one Q4_K chunk (4,736 B). A Q8_0 chunk (32 x 256, 8,704 B) does not fit, so each
  chunk is cut into two 32 x 128 halves: half h = scale bytes `[h*256, h*256+256)` + value bytes
  `[512 + h*4096, +4096)` = 4,352 B, padded to 4,736 B (the kernel reads `q80_half_block_t`, ignores the pad).
- One round = 512 rows = 16 cores x 32 rows. Shim column c (IT0, IT1, IT6, IT7), channel ch feeds 2 cores; its
  memory tile gives the 1st block of every pair to one core and the 2nd to the other. So the stream for
  `(c, ch)` holds, per round, row-block pair `p = round * 8 + c * 2 + ch`, as
  `for chunk, for half, for row block of the pair` (the same pairing as `reorder_cpy(..., 2)`).
- Each of the 8 streams is stored whole, one after another, so each channel is **one** linear transfer of
  73.5 MB (485 rounds) instead of 485 small ones; the BD length is a 32-bit word count, so it fits.

**Checks:** logits vs `cpu_ref::lm_head_logits` (fp32 final norm + Q8_0) on the NPU's own final x: cosine,
max error, top-1, top-5 overlap. Then the next step's layers alone vs right after an lm_head run must give the
same state checksum (nothing left behind on the array). Then speed: 24 layers with and without the lm_head run
in one run list, and lm_head alone.

| 10K, random context (2026-11-02) | |
|---|---|
| logits cosine vs CPU | 0.999985 (rel 5.5e-3), top-5 overlap 5/5; top-1 was a near-tie (9.88 vs 9.90) |
| next step after lm_head | same checksum |
| 24 layers, one run list | 22.24 ms |
| 24 layers + lm_head, one run list | **30.90 ms** (today with the xclbin switch: 34.8 ms, ANALYSIS.md section 15) |
| lm_head alone | 9.05 ms (65 GB/s on 8 channels) |

| 10K, real story context (`--prompt-ids build/story_ids.txt`, 2026-11-02) | |
|---|---|
| logits cosine vs CPU | 0.999995 (rel 3.5e-3), top-5 overlap 5/5 |
| top-1 | NPU 279 = CPU 279 = the real next token of the text |
| next step after lm_head | same checksum |
| 24 layers / + lm_head / lm_head alone | 22.22 / 31.34 / 9.36 ms (run to run: + lm_head 30.9-31.3 ms) |
