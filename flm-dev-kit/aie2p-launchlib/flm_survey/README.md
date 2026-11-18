<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# flm_survey: where FastFlowLM switches hw_contexts, and what warm switches would save

FastFlowLM gives each xclbin of a model its own hw_context, and every xclbin uses the whole array (8 columns,
32 cores). So whenever consecutive NPU commands go to different xclbins, the firmware switches the array between
hw_contexts. This survey counts those switches in real requests, measures what one costs, and estimates what
launchlib's warm switches (one hw_context, `../whisper_encoder` for the trial on Whisper) would save.

npu5 (Strix Halo), XRT 2.21.75, amdxdna 7.0.0-rc1, FastFlowLM v1.0.12 portable (XRT).

## Method

| Tool (`../launchlib/tools/`) | What |
|---|---|
| `hwctx_trace.c` | LD_PRELOAD wrapper of `ioctl`: logs every hw_context made (`DRM_AMDXDNA_CREATE_HWCTX`) and every command submission with its hw_context (`DRM_AMDXDNA_EXEC_CMD`) |
| `hwctx_survey.py` | for each model, `flm serve MODEL` with the tracer, then four requests: a short prompt (8 tokens out), a 1,006-1,080-token prompt (1 token out), a short prompt with 128 tokens out, and for vision models an image (1 token out) |
| `hwctx_switches.py` | per request: submissions, switches (two consecutive submissions to different hw_contexts), contexts, and the estimate |
| `ctx_switch_cost.py` | the cost of one switch between the xclbins of a model folder: a do-nothing command alternating between two hw_contexts against the same command in one (`../whisper_encoder/build/stream_probe.exe`) |

```
gcc -O2 -shared -fPIC ../launchlib/tools/hwctx_trace.c -o ../launchlib/build/hwctx_trace.so -ldl
FLM_TEST_DIR=<FastFlowLM>/.github/script FLM_MODEL_PATH=<models> \
  python3 ../launchlib/tools/hwctx_survey.py <flm> OUT <prompt.txt> <image.jpg> qwen3:0.6b qwen3.5:0.8b+image ...
python3 ../launchlib/tools/hwctx_switches.py OUT 2.5 0.5
python3 ../launchlib/tools/ctx_switch_cost.py ../whisper_encoder/build/stream_probe.exe <flm>/xclbins/Qwen3-0.6B-NPU2
```

The submission gaps in the trace do not measure a switch (FastFlowLM can submit before the previous command
ends; `hwctx_switches.py` prints the gap difference as a check, and it is near 0 for some models), so the cost
comes from `ctx_switch_cost.py`.

## What one switch costs

| Model folder | xclbins | switch (mean, min-max over the pairs) |
|---|---|---|
| Qwen3-0.6B | 4 | 2.39 ms (2.20-2.50) |
| Llama-3.2-1B | 4 | 2.49 ms (2.40-2.60) |
| Gemma3-1B | 5 | 2.53 ms (2.35-2.60) |
| LFM2-1.2B | 5 | 2.48 ms (2.35-2.60) |
| Qwen3.5-0.8B | 8 | 2.65 ms (2.50-2.90) |

The same command in one hw_context: 0.1 ms. The estimate below uses 2.5 ms per switch and, to be safe, 0.5 ms per
warm switch (0.75 ms in brackets): a Whisper launch with its warm switch takes ~0.19 ms with control packets
(32 cores, `../whisper_encoder` step 1); those of these models are not measured.

## Where the switches are

Every model switches in **prefill**: a fixed number per prompt, from one xclbin per kind of operation per layer
(e.g. Qwen3-0.6B, 28 layers: per layer the commands go to three of its four hw_contexts in the order
A, B x3, C, B x4, 4 switches per layer, 112 per prompt; the trace does not say which xclbin each context is). In
**decode**, the 9 models with an `lm_head.xclbin` (Gemma3, Gemma4, Qwen3.5, Qwen3.6, MiniCPM-V, Nemotron Bolt)
switch twice per token (to the LM head and back, ~5 ms per token); the others do not switch in decode.

## Time a warm switch would save (share of the request's time)

| Model | short prompt (time to first token) | 1k-token prompt | 128-token reply | image | switches per prompt | per decoded token |
|---|---|---|---|---|---|---|
| qwen3:0.6b | **38%** (33) | **26%** (23) | 13% (11) | - | 112 | 0 |
| gemma3:1b | **31%** (27) | 16% (14) | 19% (16) | - | 118 | 2 |
| qwen3.5:0.8b | **31%** (27) | **21%** (18) | **22%** (19) | **20%** (17) | 89 | 2 |
| lfm2:1.2b | **27%** (24) | 18% (16) | 5% (5) | - | 64 | 0 |
| llama3.2:1b | **27%** (24) | 15% (13) | 5% (5) | - | 64 | 0 |
| qwen2.5-it:3b | **27%** (24) | 19% (16) | 5% (4) | - | 144 | 0 |
| gemma4-it:e4b | **26%** (23) | 19% (17) | 8% (7) | **23%** (20) | 226 | 2 |
| nemotron3.5-bolt:e4b | **26%** (23) | 19% (17) | 9% (7) | **23%** (20) | 226 | 2 |
| nanbeige4.1:3b | **24%** (21) | 14% (12) | 4% (4) | - | 128 | 0 |
| qwen2.5vl-it:3b | **23%** (20) | 19% (17) | 5% (4) | 14% (12) | 144 | 0 |
| qwen3vl-it:4b | **22%** (19) | 14% (12) | 4% (3) | 15% (14) | 144 | 0 |
| gemma3:4b | **20%** (17) | 15% (13) | 10% (9) | 15% (13) | 150 | 2 |
| phi4-mini-it:4b | 18% (16) | 15% (13) | 4% (3) | - | 128 | 0 |
| qwen3.5:2b | 18% (16) | 15% (13) | 13% (12) | 14% (13) | 89 | 2 |
| minicpm-v-4.7:1b | 17% (15) | 8% (7) | 19% (17) | 11% (10) | 35 | 2 |
| qwen3.6-moe:35b-a3b | 14% (12) | 4% (4) | 9% (7) | 5% (4) | 212 | 2 |
| gpt-oss:20b | 8% (7) | 7% (7) | 3% (3) | - | 168 | 0 |
| gemma4-it:12b | 7% (6) | 5% (5) | 3% (3) | 9% (7) | 120 | 2 |

Decode speed alone (the models that switch per token; 2 x 2.0 ms saved per token): qwen3.5:0.8b 20%,
minicpm-v-4.7:1b 20%, gemma3:1b 16%, qwen3.5:2b 12%, gemma3:4b 8%, qwen3.6-moe 7%, Gemma4 E4B / Bolt 5%,
gemma4-it:12b 3%.

So: **more than 20% of the time to first token on 12 of 18 models** (small and mid-size, short prompts: chat
turns, tool calls, d3-style decisions), about 20% of decode speed on two, less on long prompts and on the 12B+
models (more compute per switch). For comparison, Whisper's warm switches save ~160 ms of ~600 ms per window.

## Plan (not started; needs approval)

1. **Feasibility on one model, Qwen3-0.6B (4 xclbins, prefill only, the largest share).** Build its four kernels
   (`attn`, `mm`, `layer`, `dequant`; `FLM_Xclbin/kernel_assignment.md` names their builds) with
   `--aie-generate-txn` and package them (`../launchlib/tools/make_package.py`; generalize
   `../whisper_encoder/tools/build_packages.sh`). Gate: every init write classified, `test_build` passes for every
   pair, and the measured warm switch per pair (`switch_cost`-style, control packets) is at most ~0.5 ms. Stop here
   if a kernel uses something the library does not model.
2. **One mechanism in FLM_DLL for every model, not per model** (Whisper's trial was per engine): in the shared
   runtime (`npu_xclbin_manager` / `npu_app`), when a model's xclbin folder has switch files, every app of the
   model runs in the first xclbin's hw_context, the runtime tracks which app the array holds, and a launch of
   another app gets the warm switch (current -> this) in front of its instructions (joined streams made once per
   pair used). No switch files, a hash mismatch, or `FLM_NO_SWITCH=1`: today's path. Fewer hw_contexts per model
   is a side effect (the 16-context limit).
3. **Validate Qwen3-0.6B**: greedy output token-identical to today's on npu_test's chat prompts and on long
   prompts; npu_test `chat`, `serving` and `prefix_cache` suites; time to first token and decode speed; a second
   process on the NPU (as Whisper's step 3).
4. **Second pilot with decode switches and vision: Qwen3.5-0.8B** (8 xclbins: GateDeltaNet prefill, LM head, vision).
5. **Shipping, model by model**: CI builds the packages and switch files with the xclbins (`kernel_assignment.json`
   artifacts), PR Check benches the touched models (XRT and HRX), Windows run; the same list as Whisper's step 4.

Open: warm switches of these kernels are not measured (the estimate's main uncertainty: 0.75 ms instead of 0.5 takes
2-5 points off, Whisper's ~0.2 ms would add some); with another process on the NPU, today's path is already slow
and unreliable (Whisper step 3), and this does not change that.
