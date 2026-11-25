<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# qwen3_asr_npu — Confucius4-R2T2 / Qwen3-ASR speech recognition

`libqwen3_asr_npu.so` runs NetEase Youdao's Confucius4-R2T2 (Qwen3-ASR-1.7B fine-tuned for streaming, append-only output):
a 24-layer audio encoder (0.32 B) and a Qwen3-1.7B text decoder. Public header: `include/models/qwen3_asr/qwen3_asr_npu.hpp`.

## Where each stage runs

| Stage | Where | Code |
|---|---|---|
| log-mel (128 bins, hop 160) | host, in FastFlowLM (the wrapper) | `FastFlowLM/src/common/AutoModel/modeling_qwen3_asr*.cpp` |
| audio encoder (conv x3, 24 pre-LN layers, 104-token window attention, projector) | matrix products (conv2/3, conv_out, qkv, o, fc1, fc2, proj1/2; ~99 % of the FLOPs) on the NPU (`audio_gemm.xclbin`, `FLM_Xclbin/bfp16_gemm`); conv1, LayerNorm, window attention, residuals on the host (AVX-512), BF16 rounding points of the reference | `audio/` |
| prompt prefill (system + audio tokens + text so far) | matrix products (q/k/v, o, up/gate, down) on the NPU (`prefill.xclbin`, `FLM_Xclbin/dequant_mm_mix_bfp16`), reading the mixed decode weights in place; attention on the NPU too for prompts of about 1024 tokens or more (`attn.xclbin`, the shared prefill attention); norms, RoPE, shorter attention, the KV cache and the LM head on the host (AVX-512) | `decoder/`, `prefill/` |
| decode (28 layers + tied lm_head) | NPU, `generic_decoding_layer` model type `QWEN3_ASR_1_7B_MIX` (Q4_1 + Q8_0) | `qwen3_asr_npu*.{hpp,cpp}` |

The prefill is the CPU decoder (`decoder/`) with its matrix products handed to `prefill/` (the `projections` interface): one
32-core call per product and layer, in jobs of up to 512 rows; the host packs the activations as block float and unpacks the bf16
results while the NPU runs the next job. The attention moves to `attn.xclbin` only when its work reaches that of a 1024-row
prompt: switching between the two kernels costs about 2.5 ms twice per layer, more than the CPU attention of shorter prompts.
The prefill hands its K/V rows to the NPU's striped KV cache with `kv_rows_copy`.

Every NPU call of the engine (decode, prefill products, encoder products) takes one lock (`npu_mu`): the next step's work runs
in the background during the decode, and a run of one hw_context while the decode runlist of another is in flight never
completes. Streaming: that background job encodes the next step's open window on the host (the encoder keeps a host copy of its
weights, 635 MB), hidden under the NPU decode; on the NPU it would wait for the decode, and so would the step. Offline: it
encodes and prefills the next ~60 s piece on the NPU.

## Weights: why Q8_0

4-bit round-to-nearest (Q4_0, Q4_1, Q4_K, even with a Q8_0 head) breaks this fine-tune: empty transcripts (Q4_1: 32 % of clips,
Q4_K: 29 %) or Chinese speech answered in English (Q4_0). Q8_0 is lossless (paired WER/CER vs BF16 on 120 LibriSpeech + 60 FLEURS-zh
clips: +0.00). A 4-bit body works only with `v_proj`, `ffn_down` and the head kept at 8 bit (Youdao's Q4_K_M does that); the decode
layer has one `QUANT_TYPE` per model, so that would need a mixed-precision kernel.

Package: `FLM_Q4NX_Converter -f qwen3-asr` (`configs/qwen3_asr.json`): `model.q4nx` (Q8_0 body and head, BF16 embedding, 2.45 GB),
`audio_weight.q4nx` (BF16 audio tower in the audio MM tile layout, 635 MB). `check_qwen3_asr.py` verifies a package against the HF checkpoint.

## Measured (this box, Ryzen AI 9 HX 370, 180-clip eval set, paired vs the BF16 PyTorch reference)

| | |
|---|---|
| English WER (120 clips) | 2.98 vs 2.89 (+0.09, 95% CI -0.09 .. +0.27) |
| Chinese CER (60 clips) | 8.96 vs 8.96 (+0.00) |
| decode | 26-29 ms/token with the mixed package (28 layers + head) |
| prompt prefill, decoder (NPU products) | 85 tokens 77-83 ms (host 121), 150 tokens 115 ms, 904 tokens 485 ms (host 1069), 2048 tokens 931 ms (host 2589; NPU attention from 1024 tokens) |
| encoder (host) | 21-35 ms per second of audio |
| encoder (NPU products) | 6.3 ms per second of audio on a 68 s clip (host 11), 8-10 ms/s at 7-18 s; 1 s clips ~44 ms/s (host ~21: the weights are read once per layer) |
| prefill + encoder, median clip (~7 s) | about 0.85 s |

NPU encoder, mixed package, 180 clips: English WER 2.76, Chinese CER 6.32 (host encoder on the same build: 2.85 / 5.88; the
paired 95% CI of the Chinese set is about ±5 points, so the two are not told apart). The NPU products keep A and W at bf16
precision: each is split into its bfp16 value and the scaled rest (more rows of A, more columns of W in the same call); plain bfp16
products drift the conv stack on quiet input.

## Debug switches

`QWEN3_ASR_NPU_LAYERS=n` runs only n decoder layers on the NPU (bisecting a hang). `QWEN3_ASR_CPU_DECODE=1` decodes on the host
(accuracy A/B of the NPU decode kernel). `QWEN3_ASR_CPU_ENCODER=1` runs the whole audio encoder on the host (also the case
when the package has no `audio_gemm.xclbin`); `QWEN3_ASR_NPU_STREAM_ENCODE=1` runs the streaming background encode on the NPU
as well (no host copy). `QWEN3_ASR_CPU_PREFILL=1` prefills the prompt on the host (also without
`prefill.xclbin`); `QWEN3_ASR_CPU_ATTENTION=1` keeps only the prompt attention there (also without `attn.xclbin`);
`QWEN3_ASR_NPU_ATTENTION_MIN=rows` moves the NPU attention threshold (0: always). `QWEN3_ASR_PROF=1` prints the prefill's
matmul / attention / rest split and the pack / NPU / unpack times of each NPU call. `QWEN3_ASR_GEMM_PROF=1` prints pack / NPU / unpack times of each encoder product. `TRACE_STEPS=1` makes `tools/qwen3_asr_engine_run` print the top-5 logits of the first steps.

## Test

```shell
cd FLM_DLL/detail/qwen3_asr_npu
make harness                      # libqwen3_asr_npu.so + tools/qwen3_asr_engine_run
build/qwen3_asr_npu/bin/qwen3_asr_engine_run <pkg> <xclbin_root> <goldens> <clip>... [--mel] [--out-ids f]
cd decoder && make && ./test_decoder_goldens <model.q4nx> <goldens> <clip>...   # CPU decoder alone, no NPU
cd audio/test && make run                                                         # CPU encoder alone, no NPU
cd audio/test && make run_npu XCLBIN=<audio_gemm.xclbin> WEIGHTS=... GOLDENS=...   # the encoder with its products on the NPU
```

## Gotchas

- The host must not write a buffer the NPU writes: the prefill logits
  come from a host buffer, not from `logits`.
- Only `proj_addr` (the proj tiles' IS_ATTN RTP) differs from the Q4_1 builds of the generic decode layer: 49664 here, 33280 there
  (`build/<MODEL>/xclbins/rtp_addresses.txt`).
- Build the decode xclbin with Peano `llvm-aie 22.0.0.2026090201+a36c62b9`; the older Peano in the venv crashes on `attn_kv.cc`.
- The stock HF `Qwen3ASRAudioEncoder` attends over the whole clip on CPU (no window mask); the reference used here applies the
  `cu_seqlens` window mask like the vLLM backend the model card numbers come from.
