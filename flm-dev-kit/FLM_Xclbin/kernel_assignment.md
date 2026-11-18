<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Kernel Assignment

Each table below maps a `model_tag` to its kernel folder and CI artifacts:

| Kernels | Artifacts | Compiler |
|---|---|---|
| kernel name | CI artifacts | compiler name |

## How to Test Your Kernel Code

Push or open a PR to the `main` branch under the `FLM_Xclbin` folder. GitHub Actions will build the modified kernels and publish the artifacts. To apply them locally:

1. Download the artifact from the GitHub Actions run.
2. Navigate to the xclbins directory for your platform and open the kernel folder for your `model_tag`:
   - **Linux:** `<install dir>/opt/fastflowlm/share/flm/xclbins/`
   - **Windows:** `<install dir>/xclbins/`
3. Replace the target `.xclbin` file with the downloaded one.

> **Note:** If the artifact filename differs from the original, rename it to match before replacing.

## Limitations (will fix in the future)

- Qwen3.5-Omni — no FastFlowLM model tag yet, so it is not in `kernel_assignment.json` and CI
  doesn't bench it. FastFlowLM ships its `Qwen3.5-OMNI-NPU2` folder; see
  [Qwen3.5-Omni](#qwen35-omni) for what CI builds for it. `layer`, `lm_head` and `mm` have no CI
  build yet.

---

## Models

- [LLaMA](#llama)
- [DeepSeek](#deepseek)
- [Qwen](#qwen)
- [Gemma](#gemma)
- [MedGemma](#medgemma)
- [TranslateGemma](#translategemma)
- [Qwen3.6 MoE](#qwen36-moe)
- [Qwen3.8 MTP](#qwen38-mtp)
- [Hy-Memory](#hy-memory)
- [MiniCPM-V](#minicpm-v)
- [GPT-OSS](#gpt-oss)
- [LiquidAI/LFM](#liquidailfm)
- [Microsoft/Phi](#microsoftphi)
- [Nanbeige](#nanbeige)
- [Hunyuan](#hunyuan)
- [Whisper](#whisper)
- [EmbeddingGemma](#embeddinggemma)
- [SmolVLA](#smolvla)

### LLaMA

#### `llama3.1:8b` → `Llama-3.1-8B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_4` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_LLAMA_3_1_8B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `llama3.2:1b` → `Llama-3.2-1B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_64_GQA_1_4` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_LLAMA_3_2_1B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `llama3.2:3b` → `Llama-3.2-3B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_3` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_LLAMA_3_2_3B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

### DeepSeek

#### `deepseek-r1:8b` → `Deepseek-R1-Distill-Llama-8B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_4` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_LLAMA_3_1_8B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `deepseek-r1-0528:8b` → `DeepSeek-R1-0528-Qwen3-8B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_4` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_QWEN3_8B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

### Qwen

#### `qwen2.5-it:3b` → `Qwen2.5-3B-Instruct-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Qwen2_5_prefill_attention_3b` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_0_w128x512` | Peano |
| `layer.xclbin` | `xclbins-Qwen2_5_decoding_3b_QWEN2_3B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `qwen2.5vl-it:3b` → `Qwen2.5-VL-3B-Instruct-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Qwen2_5_prefill_attention_3b` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-Qwen2_5_decoding_3b_QWEN2VL_3B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |
| `vision_attention.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN_2_5-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN2_5_3B-VISION-MM` | Peano |
| `vision_window_attention.xclbin` | `xclbins-VisionAttention_hostWindowAttention_while_array_QWEN2_5-VISION-WINDOW` | Peano |

#### `qwen3:0.6b` → `Qwen3-0.6B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_2` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_QWEN3_0_6B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `qwen3:1.7b` → `Qwen3-1.7B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_2` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_QWEN3_1_7B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `qwen3:4b` → `Qwen3-4B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_4` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q8_0_w128x512` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_QWEN3_4B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `qwen3:8b` → `Qwen3-8B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_4` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_QWEN3_8B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `qwen3-it:4b` → `Qwen3-4B-Instruct-2507-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_4` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q8_0_w128x512` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_QWEN3_4B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `qwen3-tk:4b` → `Qwen3-4B-Thinking-2507-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_4` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q8_0_w128x512` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_QWEN3_4B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `qwen3vl-it:4b` → `Qwen3-VL-4B-Instruct-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_4` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_QWEN3_4B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_4B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_4B-VISION-MM` | Peano |

#### `qwen3vl-flash:4b` → `Qwen3-VL-4B-Instruct-NPU2`

Same checkpoint as `qwen3vl-it:4b`, run by the `qwen3vl_flash` engine. Prefill
runs one fused overlay instead of separate `mm.xclbin` and `attn.xclbin`, so the
array is never reconfigured mid-layer. Columns 1-6 are the dequant+matmul half
and columns 0 and 7 a single attention compute unit, which keeps three matmul
columns on each of the two DDR channels.

| Kernels | Artifacts | Compiler |
|---|---|---|
| `fused_prefill.xclbin` | `xclbins-Qwen3VL_fused_prefill` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_QWEN3_4B` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_4B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_4B-VISION-MM` | Peano |

#### `qwen3.5:0.8b` → `Qwen3.5-0.8B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Qwen3_5_gated_attention_prefill` | Peano |
| `conv.xclbin` | `xclbins-Qwen3_5_conv1d_prefill` | Peano |
| `GateDeltaNet_prefill.xclbin` | `xclbins-Qwen3_5_gate_delta_net_prefill_hostDeltaNetPrefill` | Peano |
| `layer.xclbin` | `xclbins-Qwen3_5_qwen3_5_decoding_QWEN3_5_08B` | Peano |
| `lm_head.xclbin` | `xclbins-Qwen3_5_lm_head_npu_bin_QWEN3_5_0_8B` | Peano |
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_5_0_8B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_5_0_8B-VISION-MM` | Peano |

#### `qwen3.5:2b` → `Qwen3.5-2B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Qwen3_5_gated_attention_prefill` | Peano |
| `conv.xclbin` | `xclbins-Qwen3_5_conv1d_prefill` | Peano |
| `GateDeltaNet_prefill.xclbin` | `xclbins-Qwen3_5_gate_delta_net_prefill_hostDeltaNetPrefill` | Peano |
| `layer.xclbin` | `xclbins-Qwen3_5_qwen3_5_decoding_QWEN3_5_2B` | Peano |
| `lm_head.xclbin` | `xclbins-Qwen3_5_lm_head_npu_bin_QWEN3_5_2B` | Peano |
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_5_2B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_5_2B-VISION-MM` | Peano |

#### `qwen3.5:4b` → `Qwen3.5-4B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Qwen3_5_gated_attention_prefill` | Peano |
| `conv.xclbin` | `xclbins-Qwen3_5_conv1d_prefill` | Peano |
| `GateDeltaNet_prefill.xclbin` | `xclbins-Qwen3_5_gate_delta_net_prefill_hostDeltaNetPrefill` | Peano |
| `layer.xclbin` | `xclbins-Qwen3_5_qwen3_5_decoding_QWEN3_5_4B` | Peano |
| `lm_head.xclbin` | `xclbins-Qwen3_5_lm_head_npu_bin_QWEN3_5_4B` | Peano |
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_5_4B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_5_4B-VISION-MM` | Peano |

#### `qwen3.5:9b` → `Qwen3.5-9B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Qwen3_5_gated_attention_prefill` | Peano |
| `conv.xclbin` | `xclbins-Qwen3_5_conv1d_prefill` | Peano |
| `GateDeltaNet_prefill.xclbin` | `xclbins-Qwen3_5_gate_delta_net_prefill_hostDeltaNetPrefill` | Peano |
| `layer.xclbin` | `xclbins-Qwen3_5_qwen3_5_decoding_QWEN3_5_9B` | Peano |
| `lm_head.xclbin` | `xclbins-Qwen3_5_lm_head_npu_bin_QWEN3_5_9B` | Peano |
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_5_9B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_5_9B-VISION-MM` | Peano |

#### Qwen3.5-Omni

No model tag yet → `Qwen3.5-OMNI-NPU2`. Not in `kernel_assignment.json`, so CI neither assembles
nor benches it. `attn`, `conv` and `GateDeltaNet_prefill` are the same files the Qwen3.5 models
use. `—` means CI has no build target for that kernel yet; FastFlowLM ships a Chess build.

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Qwen3_5_gated_attention_prefill` | Peano |
| `conv.xclbin` | `xclbins-Qwen3_5_conv1d_prefill` | Peano |
| `GateDeltaNet_prefill.xclbin` | `xclbins-Qwen3_5_gate_delta_net_prefill_hostDeltaNetPrefill` | Peano |
| `layer.xclbin` | — | — |
| `lm_head.xclbin` | — | — |
| `mm.xclbin` | — | — |
| `mm_high.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_5-OMNI-VISION-MM-HIGH` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_5_OMNI-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_5-OMNI-VISION-MM` | Peano |
| `window_attention.xclbin` | `xclbins-VisionAttention_hostWindowAttention_while_array_QWEN3_5-OMNI-VISION-WINDOW` | Peano |

### Gemma

#### `gemma3:1b` → `Gemma3-1B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_256_GQA_1_4` | Peano |
| `dequant.xclbin` | `xclbins-Gemma3_small_gemma_text_dequant_npu_bin` | Peano |
| `layer.xclbin` | `xclbins-Gemma3_small_gemma_text_npu_bin` | Peano |
| `lm_head.xclbin` | `xclbins-Gemma3_small_gemma_text_lm_head_npu_bin` | Peano |
| `mm.xclbin` | `xclbins-Gemma3_small_gemma_text_prefill_mm_npu_bin` | Peano |

#### `gemma3:4b` → `Gemma3-4B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_256_GQA_1_2` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-Gemma3_gemma_npu_bin` | Peano |
| `lm_head.xclbin` | `xclbins-Gemma3_gemma_lm_head_npu_bin` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_GEMMA3_4B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA3_4B-VISION-MM` | Peano |

#### `gemma4-it:e2b` → `Gemma4-E2B-IT-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Gemma4_attention_DH_512_prefill` | Peano |
| `audio_conv1d.xclbin` | `xclbins-Gemma4_conv1d_prefill` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-Gemma4_decoding_GEMMA4_E2B` | Peano |
| `lm_head.xclbin` | `xclbins-Gemma4_lm_head_npu_bin_GEMMA4_E2B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |
| `swa.xclbin` | `xclbins-Gemma4_attention_DH_256_prefill` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_GEMMA4_E2B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA4-VISION-MM` | Peano |
| `vision_mm_high_precision.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA4-VISION-MM-HIGH` | Peano |

#### `gemma4-it:e4b` → `Gemma4-E4B-IT-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Gemma4_attention_DH_512_prefill` | Peano |
| `audio_conv1d.xclbin` | `xclbins-Gemma4_conv1d_prefill` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `dequant_mm_bfp.xclbin` | `xclbins-dequant_mm_Q4_1_bfp16_GEMMA4_E4B` | Peano |
| `layer.xclbin` | `xclbins-Gemma4_decoding_GEMMA4_E4B` | Peano |
| `lm_head.xclbin` | `xclbins-Gemma4_lm_head_npu_bin_GEMMA4_E4B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |
| `swa.xclbin` | `xclbins-Gemma4_attention_DH_256_prefill` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_GEMMA4_E4B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA4-VISION-MM` | Peano |
| `vision_mm_high_precision.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA4-VISION-MM-HIGH` | Peano |

#### `nemotron3.5-bolt:e4b` → `Nemotron-3.5-Bolt-E4B-NPU2`

NVIDIA Nemotron-3.5-Bolt-E4B: Gemma-4-E4B-it with more post-training, same text architecture (42 layers, hidden 2560,
vision and audio towers byte-identical to Gemma-4-E4B-it's). Same engine (`gemma4e_npu`) and the same kernels as
`gemma4-it:e4b`, in a folder of its own so this model's xclbins are built and shipped separately. The weights are
Q4_1 (converted from NVIDIA's Q4_K_M GGUF), as for `gemma4-it:e4b`.

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Gemma4_attention_DH_512_prefill` | Peano |
| `audio_conv1d.xclbin` | `xclbins-Gemma4_conv1d_prefill` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `dequant_mm_bfp.xclbin` | `xclbins-dequant_mm_Q4_1_bfp16_GEMMA4_E4B` | Peano |
| `layer.xclbin` | `xclbins-Gemma4_decoding_GEMMA4_E4B` | Peano |
| `lm_head.xclbin` | `xclbins-Gemma4_lm_head_npu_bin_GEMMA4_E4B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |
| `swa.xclbin` | `xclbins-Gemma4_attention_DH_256_prefill` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_GEMMA4_E4B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA4-VISION-MM` | Peano |
| `vision_mm_high_precision.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA4-VISION-MM-HIGH` | Peano |

#### `nemotron3.5-bolt-dflash:e4b` → `Nemotron-3.5-Bolt-E4B-DFlash-NPU2`

DFlash speculative decoding for Nemotron-3.5-Bolt-E4B (engine `gemma4e_dflash`, a fork of `gemma4e_npu`). The tag needs `nemotron3.5-bolt:e4b`
installed (`base_model`): the main weights, tokenizer and chat template come from that package, this package holds only the draft weights.
Its xclbin folder has the full base kernel set plus `lm_head_8.xclbin` (`Gemma4/lm_head_8`: the final vocabulary projection for 8 token rows per
dispatch, used for the verify rows and the draft rows) and `verify.xclbin` (`Gemma4/verify_decoding`: the decoder layer on the 8 verify rows,
`DESIGN=fused8`). Both builds check their RTP addresses against `rtp_expected/` because the engine compiles them in.

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Gemma4_attention_DH_512_prefill` | Peano |
| `audio_conv1d.xclbin` | `xclbins-Gemma4_conv1d_prefill` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `dequant_mm_bfp.xclbin` | `xclbins-dequant_mm_Q4_1_bfp16_GEMMA4_E4B` | Peano |
| `layer.xclbin` | `xclbins-Gemma4_decoding_GEMMA4_E4B` | Peano |
| `lm_head.xclbin` | `xclbins-Gemma4_lm_head_npu_bin_GEMMA4_E4B` | Peano |
| `lm_head_8.xclbin` | `xclbins-Gemma4_lm_head_8_GEMMA4_E4B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |
| `swa.xclbin` | `xclbins-Gemma4_attention_DH_256_prefill` | Peano |
| `verify.xclbin` | `xclbins-Gemma4_verify_decoding_GEMMA4_E4B` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_GEMMA4_E4B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA4-VISION-MM` | Peano |
| `vision_mm_high_precision.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA4-VISION-MM-HIGH` | Peano |

#### `nemotron3.5-bolt-dflash-flash:e4b` → `Nemotron-3.5-Bolt-E4B-DFlash-NPU2`

Single-turn flash serving with DFlash speculative decoding (engine `gemma4e_dflash_flash`: the `gemma4e_flash` fused prefill plus the
`gemma4e_dflash` draft/verify decode). Same package folder as `nemotron3.5-bolt-dflash:e4b` (and the same `base_model`,
`nemotron3.5-bolt:e4b`), so no new weights; the tag uses `fused_prefill.xclbin` instead of that tag's `attn`/`swa`/`mm`/`dequant`
prefill kernels, which is 9 NPU hardware contexts in total.

| Kernels | Artifacts | Compiler |
|---|---|---|
| `audio_conv1d.xclbin` | `xclbins-Gemma4_conv1d_prefill` | Peano |
| `fused_prefill.xclbin` | `xclbins-Gemma4_fused_prefill` | Peano |
| `layer.xclbin` | `xclbins-Gemma4_decoding_GEMMA4_E4B` | Peano |
| `lm_head.xclbin` | `xclbins-Gemma4_lm_head_npu_bin_GEMMA4_E4B` | Peano |
| `lm_head_8.xclbin` | `xclbins-Gemma4_lm_head_8_GEMMA4_E4B` | Peano |
| `verify.xclbin` | `xclbins-Gemma4_verify_decoding_GEMMA4_E4B` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_GEMMA4_E4B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA4-VISION-MM` | Peano |
| `vision_mm_high_precision.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA4-VISION-MM-HIGH` | Peano |

#### `gemma4e-flash:e2b` → `Gemma4-E2B-IT-NPU2`

Same checkpoint as `gemma4-it:e2b`, run by the `gemma4e_flash` engine. Gemma4
alternates four sliding-window layers to one full-attention layer and the two
use different head dims, so the standalone flow needs three prefill overlays per
layer — `mm.xclbin`, `attn.xclbin` (DH 512) and `swa.xclbin` (DH 256) — and swaps
between them constantly. `fused_prefill.xclbin` holds all three at once: columns
1-6 are the dequant+matmul array, column 0 the full-attention CU and column 7
the sliding-window CU, so the array is never reconfigured mid-layer.

| Kernels | Artifacts | Compiler |
|---|---|---|
| `audio_conv1d.xclbin` | `xclbins-Gemma4_conv1d_prefill` | Peano |
| `fused_prefill.xclbin` | `xclbins-Gemma4_fused_prefill` | Peano |
| `layer.xclbin` | `xclbins-Gemma4_decoding_GEMMA4_E2B` | Peano |
| `lm_head.xclbin` | `xclbins-Gemma4_lm_head_npu_bin_GEMMA4_E2B` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_GEMMA4_E2B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA4-VISION-MM` | Peano |
| `vision_mm_high_precision.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA4-VISION-MM-HIGH` | Peano |

#### `gemma4e-flash:e4b` → `Gemma4-E4B-IT-NPU2`

Same checkpoint as `gemma4-it:e4b`, run by the `gemma4e_flash` engine. Gemma4
alternates four sliding-window layers to one full-attention layer and the two
use different head dims, so the standalone flow needs three prefill overlays per
layer — `mm.xclbin`, `attn.xclbin` (DH 512) and `swa.xclbin` (DH 256) — and swaps
between them constantly. `fused_prefill.xclbin` holds all three at once: columns
1-6 are the dequant+matmul array, column 0 the full-attention CU and column 7
the sliding-window CU, so the array is never reconfigured mid-layer.

| Kernels | Artifacts | Compiler |
|---|---|---|
| `audio_conv1d.xclbin` | `xclbins-Gemma4_conv1d_prefill` | Peano |
| `fused_prefill.xclbin` | `xclbins-Gemma4_fused_prefill` | Peano |
| `layer.xclbin` | `xclbins-Gemma4_decoding_GEMMA4_E4B` | Peano |
| `lm_head.xclbin` | `xclbins-Gemma4_lm_head_npu_bin_GEMMA4_E4B` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_GEMMA4_E4B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA4-VISION-MM` | Peano |
| `vision_mm_high_precision.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA4-VISION-MM-HIGH` | Peano |

#### `gemma4-it:12b` → `Gemma4-12B-IT-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn_global_image.xclbin` | `xclbins-Gemma4_12B_QAT_attention_global_image` | Peano |
| `attn_global.xclbin` | `xclbins-Gemma4_12B_QAT_attention_DH_512_prefill` | Peano |
| `attn_sliding.xclbin` | `xclbins-Gemma4_12B_QAT_attention_DH_256_prefill` | Peano |
| `layer.xclbin` | `xclbins-Gemma4_12B_QAT_decoding` | Peano |
| `lm_head.xclbin` | `xclbins-Gemma4_12B_QAT_lm_head_npu_bin` | Peano |
| `dequant_mm.xclbin` | `xclbins-Gemma4_12B_QAT_dequant_mm` | Peano |
| `audio_image_mm.xclbin` | `xclbins-Gemma4_12B_QAT_audio_image_mm` | Peano |

### MedGemma

#### `medgemma:4b` → `Medgemma-4B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_256_GQA_1_2` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-Gemma3_gemma_npu_bin` | Peano |
| `lm_head.xclbin` | `xclbins-Gemma3_gemma_lm_head_npu_bin` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_GEMMA3_4B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA3_4B-VISION-MM` | Peano |

#### `medgemma1.5:4b` → `Medgemma-1.5-4B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_256_GQA_1_2` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-Gemma3_gemma_npu_bin` | Peano |
| `lm_head.xclbin` | `xclbins-Gemma3_gemma_lm_head_npu_bin` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_GEMMA3_4B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA3_4B-VISION-MM` | Peano |

### TranslateGemma

#### `translategemma:4b` → `Translategemma-4B-Instruct-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_256_GQA_1_2` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-Gemma3_gemma_npu_bin` | Peano |
| `lm_head.xclbin` | `xclbins-Gemma3_gemma_lm_head_npu_bin` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_GEMMA3_4B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GEMMA3_4B-VISION-MM` | Peano |

### Qwen3.6 MoE

#### `qwen3.6-moe:35b-a3b` → `Qwen3.6-35B-A3B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Qwen3_6_gated_attention_prefill` | Peano |
| `conv.xclbin` | `xclbins-Qwen3_5_conv1d_prefill` | Peano |
| `GateDeltaNet_prefill.xclbin` | `xclbins-Qwen3_5_gate_delta_net_prefill_hostDeltaNetPrefill` | Peano |
| `layer.xclbin` | `xclbins-Qwen3_6_qwen3_6_moe_decoding_QWEN3_6_35B_A3B` | Peano |
| `lm_head.xclbin` | `xclbins-Qwen3_5_lm_head_npu_bin_QWEN3_5_2B` | Peano |
| `dequant_mm.xclbin` | `xclbins-dequant_mm_256x512x512` | Peano |
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_6_35B-VISION-ATTENTION` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_6_35B-VISION-MM` | Peano |

### Qwen3.8 MTP

#### `qwen3.8-mtp:27b` → `Qwen3.8-27B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attention_layer.xclbin` | `xclbins-Qwen3_8_gated_attention_verify` | Peano |
| `attention_prefill.xclbin` | `xclbins-Qwen3_8_gated_attention_prefill` | Peano |
| `conv.xclbin` | `xclbins-Qwen3_8_conv1d_prefill` | Peano |
| `GateDeltaNet_prefill.xclbin` | `xclbins-Qwen3_8_gate_delta_net_prefill` | Peano |
| `deltanet_layer.xclbin` | `xclbins-Qwen3_8_gate_delta_net_verify` | Peano |
| `MTP_layer.xclbin` | `xclbins-Qwen3_8_MTP_decoding` | Peano |
| `dequant_mm.xclbin` | `xclbins-Qwen3_8_dequant_mm` | Peano |
| `lm_head.xclbin` | `xclbins-Qwen3_8_lm_head_npu_bin` | Peano |
| `lm_head_8.xclbin` | `xclbins-Qwen3_8_lm_head_8` | Peano |
| `vision_attn.xclbin` | `xclbins-Qwen3_8_vision_attn` | Peano |
| `vision_mm_a.xclbin` | `xclbins-Qwen3_8_vision_mm` | Peano |
| `vision_mm_b.xclbin` | `xclbins-Qwen3_8_vision_mm` | Peano |

### Hy-Memory

#### `hy-memory:30b-a3b` → `Hy-Memory-30B-A3B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `dense_layer.xclbin` | `xclbins-Hy_mem_30BA3B_dense_decoding` | Peano |
| `moe_layer.xclbin` | `xclbins-Hy_mem_30BA3B_moe_decoding` | Peano |
| `lm_head.xclbin` | `xclbins-Hy_mem_30BA3B_lm_head_npu_bin` | Peano |
| `prefill_overlay.xclbin` | `xclbins-Hy_mem_30BA3B_prefill_overlay` | Peano |

### MiniCPM-V

#### `minicpm-v-4.7:1b` → `MiniCPM-V-4.7-1B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Qwen3_5_gated_attention_prefill` | Peano |
| `attn_bfp16c.xclbin` | `xclbins-Qwen3_5_gated_attention_prefill_chunked` | Peano |
| `conv.xclbin` | `xclbins-Qwen3_5_conv1d_prefill` | Peano |
| `GateDeltaNet_prefill.xclbin` | `xclbins-Qwen3_5_gate_delta_net_prefill_hostDeltaNetPrefill` | Peano |
| `layer.xclbin` | `xclbins-Qwen3_5_qwen3_5_decoding_QWEN3_5_08B` | Peano |
| `layer_lm_sl.xclbin` | `xclbins-MiniCPM_V_decoding_QWEN3_5_08B_SL` | Peano |
| `lm_head.xclbin` | `xclbins-Qwen3_5_lm_head_npu_bin_QWEN3_5_0_8B` | Peano |
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `mm_rtm2_glu.xclbin` | `xclbins-dequant_mm_bfp16_rtm2_glu_512x512x512` | Peano |
| `vision_attn.xclbin` | `xclbins-Qwen3_8_vision_attn` | Peano |
| `vision_attn_kv.xclbin` | `xclbins-MiniCPM_V_vision_attn_kv` | Peano |
| `vision_mm_a.xclbin` | `xclbins-Qwen3_8_vision_mm` | Peano |
| `vision_mm_b.xclbin` | `xclbins-Qwen3_8_vision_mm` | Peano |
| `vision_mm_c.xclbin` | `xclbins-MiniCPM_V_vision_mm_c` | Peano |
| `vision_mm_p.xclbin` | `xclbins-MiniCPM_V_vision_mm_p` | Peano |

The tower uses `vision_mm_p` (design C on packed bfp16 operands) and `vision_attn_kv` when they are present.
`vision_mm_c` / `vision_mm_a` / `vision_mm_b` / `vision_attn` are the fallbacks (`FLM_MINICPM_V_VISION_MM_P=0`,
`FLM_MINICPM_V_VISION_ATTN_KV=0`). `vision_mm_a` is also the single-context home of the attention.

#### `minicpm-v-4.7-moe:35b-a3b` → `MiniCPM-V-4.7-35B-A3B-NPU2`

Text = the Qwen3.6-35B-A3B kernels, vision = the MiniCPM-V-4.7-1B / Qwen3.8 tower kernels; no kernel of its own.

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Qwen3_6_gated_attention_prefill` | Peano |
| `conv.xclbin` | `xclbins-Qwen3_5_conv1d_prefill` | Peano |
| `GateDeltaNet_prefill.xclbin` | `xclbins-Qwen3_5_gate_delta_net_prefill_hostDeltaNetPrefill` | Peano |
| `layer.xclbin` | `xclbins-Qwen3_6_qwen3_6_moe_decoding_QWEN3_6_35B_A3B` | Peano |
| `lm_head.xclbin` | `xclbins-Qwen3_5_lm_head_npu_bin_QWEN3_5_2B` | Peano |
| `dequant_mm.xclbin` | `xclbins-dequant_mm_256x512x512` | Peano |
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `vision_attn.xclbin` | `xclbins-Qwen3_8_vision_attn` | Peano |
| `vision_mm_a.xclbin` | `xclbins-Qwen3_8_vision_mm` | Peano |
| `vision_mm_b.xclbin` | `xclbins-Qwen3_8_vision_mm` | Peano |

#### `minicpm-ocr:1.0` → `MiniCPM-OCR-1.0-NPU2`

Text = the Qwen3.5-0.8B kernels, vision = the Qwen3.8 tower kernels, as `minicpm-v-4.7:1b`; GatedDeltaNet prefill is
its own rewritten kernel. The faster paths come from `FLM_Xclbin/minicpm_ocr/`: decode with the lm_head inside the
layer (`layer_with_lm_head`, and `layer_with_lm_head_kvb` reading K/V as bfp16; matrix type `QWEN3_5_08B_KVB` =
`make bitstream KV_BFP16=1`), bfp16 prefill attention (`attn_bfp16`), bfp16 prefill mm with the GLU on the array
(`mm_bfp16_glu`) and `vision_overlay` (vision mm + vision attention in one xclbin). The engine opens each of these when
present and falls back to `attn` / `layer` + `lm_head` / `mm` / `vision_attn` + `vision_mm_a` (env switches turn them
off); `vision_mm_b` is always needed. The overlay's switch table and layout are compiled into the engine
(`FLM_DLL/detail/minicpm_ocr_npu/image/npu/`); after any change to the overlay or the two vision folders run
`make engine` there and commit the copies. `make bitstream` fails if they differ (`check_engine`). No conv kernel:
conv1d runs on the host.

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-Qwen3_5_gated_attention_prefill` | Peano |
| `attn_bfp16.xclbin` | `xclbins-minicpm_ocr_gated_attention_prefill` | Peano |
| `GateDeltaNet_prefill.xclbin` | `xclbins-minicpm_ocr_gate_delta_net_prefill` | Peano |
| `layer.xclbin` | `xclbins-Qwen3_5_qwen3_5_decoding_QWEN3_5_08B` | Peano |
| `layer_with_lm_head.xclbin` | `xclbins-minicpm_ocr_decoding_QWEN3_5_08B` | Peano |
| `layer_with_lm_head_kvb.xclbin` | `xclbins-minicpm_ocr_decoding_QWEN3_5_08B_KVB` | Peano |
| `lm_head.xclbin` | `xclbins-Qwen3_5_lm_head_npu_bin_QWEN3_5_0_8B` | Peano |
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `mm_bfp16_glu.xclbin` | `xclbins-minicpm_ocr_dequant_mm_prefill` | Peano |
| `vision_attn.xclbin` | `xclbins-Qwen3_8_vision_attn` | Peano |
| `vision_mm_a.xclbin` | `xclbins-Qwen3_8_vision_mm` | Peano |
| `vision_mm_b.xclbin` | `xclbins-Qwen3_8_vision_mm` | Peano |
| `vision_overlay.xclbin` | `xclbins-minicpm_ocr_vision_overlay` | Peano |

### GPT-OSS

#### `gpt-oss:20b` → `GPT-OSS-20B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-GPTOSS_gpt_prefill_attn_v1_npu_bin` | Peano |
| `dequant.xclbin` | `xclbins-GPTOSS_gpt_dequant_mxfp4_npu_bin` | Peano |
| `expert.xclbin` | `xclbins-GPTOSS_gpt_prefill_expert_mm` | Peano |
| `layer.xclbin` | `xclbins-GPTOSS_gpt_npu_bin` | Peano |
| `mm.xclbin` | `xclbins-GPTOSS_gpt_prefill_mm_GPTOSS-MAIN` | Peano |
| `short_seq_mm.xclbin` | `xclbins-GPTOSS_gpt_prefill_mm_GPTOSS-SHORT-SEQ` | Peano |

#### `gpt-oss-sg:20b` → `GPT-OSS-Safeguard-20b-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-GPTOSS_gpt_prefill_attn_v1_npu_bin` | Peano |
| `dequant.xclbin` | `xclbins-GPTOSS_gpt_dequant_mxfp4_npu_bin` | Peano |
| `expert.xclbin` | `xclbins-GPTOSS_gpt_prefill_expert_mm` | Peano |
| `layer.xclbin` | `xclbins-GPTOSS_gpt_npu_bin` | Peano |
| `mm.xclbin` | `xclbins-GPTOSS_gpt_prefill_mm_GPTOSS-MAIN` | Peano |
| `short_seq_mm.xclbin` | `xclbins-GPTOSS_gpt_prefill_mm_GPTOSS-SHORT-SEQ` | Peano |

### LiquidAI/LFM

#### `lfm2:1.2b` → `LFM2-1.2B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_64_GQA_1_4` | Peano |
| `conv.xclbin` | `xclbins-LiquidAI_conv1d` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_0_w128x512` | Peano |
| `layer.xclbin` | `xclbins-LiquidAI_lfm_npu_bin_LFM_1_2B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `lfm2:2.6b` → `LFM2-2.6B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_64_GQA_1_4` | Peano |
| `conv.xclbin` | `xclbins-LiquidAI_conv1d` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_0_w128x512` | Peano |
| `layer.xclbin` | `xclbins-LiquidAI_lfm_npu_bin_LFM_2_6B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `lfm2-trans:2.6b` → `LFM2-2.6B-Transcript-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_64_GQA_1_4` | Peano |
| `conv.xclbin` | `xclbins-LiquidAI_conv1d` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_0_w128x512` | Peano |
| `layer.xclbin` | `xclbins-LiquidAI_lfm_npu_bin_LFM_2_6B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `lfm2.5-it:1.2b` → `LFM2.5-1.2B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_64_GQA_1_4` | Peano |
| `conv.xclbin` | `xclbins-LiquidAI_conv1d` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_0_w128x512` | Peano |
| `layer.xclbin` | `xclbins-LiquidAI_lfm_npu_bin_LFM_1_2B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `lfm2.5-tk:1.2b` → `LFM2.5-1.2B-Thinking-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_64_GQA_1_4` | Peano |
| `conv.xclbin` | `xclbins-LiquidAI_conv1d` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_0_w128x512` | Peano |
| `layer.xclbin` | `xclbins-LiquidAI_lfm_npu_bin_mixed` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

### Microsoft/Phi

#### `lfm3:1.7b` → `LFM3-1.7B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_4` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_0_w128x512` | Peano |
| `layer.xclbin` | `xclbins-LiquidAI_lfm3_npu_bin_LFM3_1_7B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

#### `phi4-mini-it:4b` → `Phi4-mini-Instruct-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_3` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_PHI4_4B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

### Nanbeige

#### `nanbeige4.1:3b` → `Nanbeige4.1-3B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn.xclbin` | `xclbins-generic_prefill_attention_attention_DH_128_GQA_1_4` | Peano |
| `dequant.xclbin` | `xclbins-Dequant_Q4_1_w128x512` | Peano |
| `layer.xclbin` | `xclbins-generic_decoding_layer_NANBEIGE_3B` | Peano |
| `mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_GENERIC` | Peano |

### Hunyuan

#### `hy-mt2-flash:1.8b` → `Hy-MT2-1.8B-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `layer.xclbin` | `xclbins-Hy_MT2_1.8B_decoding` | Peano |
| `prefill_overlay.xclbin` | `xclbins-Hy_MT2_1.8B_prefill_overlay` | Peano |

### Confucius4-R2T2 (Qwen3-ASR)

#### `r2t2:1.7b` → `Confucius4-R2T2-NPU2`

Decode only: the prompt is prefilled on the host (AVX-512), the audio encoder is host code too. Mixed decode weights
(`QWEN3_ASR_1_7B_MIX`): Q4_1 for q/k/o/gate/up and the LM head, Q8_0 for v_proj and down_proj (4-bit everywhere breaks this
fine-tune); the package's `npu_q41.q4nx` holds the Q4_1 copies. `QWEN3_ASR_1_7B` is the all-Q8_0 build (packages without that file).

| Kernels | Artifacts | Compiler |
|---|---|---|
| `layer.xclbin` | `xclbins-generic_decoding_layer_QWEN3_ASR_1_7B_MIX` | Peano |

### Whisper

#### `whisper-v3:turbo` → `Whisper-V3-Turbo-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `decoder_mv.xclbin` | `xclbins-Whisper_decoder_mv_dequant_mv` | Peano |
| `encoder_attn.xclbin` | `xclbins-Whisper_encoder_attention_888` | Peano |
| `encoder_dequant.xclbin` | `xclbins-Whisper_encoder_dequant` | Peano |
| `encoder_mm.xclbin` | `xclbins-Whisper_encoder_MM` | Peano |
| `whisper_head.xclbin` | `xclbins-Whisper_whisper_lm_head_npu_bin` | Peano |

### EmbeddingGemma

#### `embed-gemma:300m` → `Embedding-Gemma-300M-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `attn_full_mask.xclbin` | `xclbins-Embedding_Gemma_gemma_embed_attention_full_mask` | Peano |
| `mm.xclbin` | `xclbins-Embedding_Gemma_MM` | Peano |
| `mv.xclbin` | `xclbins-Embedding_Gemma_mv` | Peano |
| `sliding_attn.xclbin` | `xclbins-Embedding_Gemma_gemma_embed_attention_sliding_non_causal_mask` | Peano |

### Decision 3.0 (d3)

#### `d3:lite` → `D3-Lite-NPU2`

The d3 decision encoder (vllm-sr/d3-lite: the Qwen3.5-0.8B text tower, bidirectional full attention, no LM
head) runs every projection on `mm.xclbin`; conv1d and the gated delta rule run on the host, and so does
attention for short prompts. Prompts at or over the engine's cutover use `attn.xclbin`, Qwen3.5's prefill attention
made bidirectional, registered on the first such prompt (`FLM_DLL/detail/d3_npu`). Requests with images run Qwen3.5-0.8B's vision tower on
`vision_mm.xclbin` / `vision_attn.xclbin`, registered on the first image request.

| Kernels | Artifacts | Compiler |
|---|---|---|
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `attn.xclbin` | `xclbins-Qwen3_5_bidirectional_attention_prefill` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_5_0_8B-VISION-MM` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_5_0_8B-VISION-ATTENTION` | Peano |

#### `d3:nano` → `D3-Nano-NPU2`

The same engine and the same text kernels as `d3:lite` (vllm-sr/d3-nano: the Qwen3.5-2B text tower, 24 layers,
hidden 2048); the engine reads its dimensions from the package. Requests with images run Qwen3.5-2B's vision tower
(24 layers, width 1024, output 2048; the blocks are stock Qwen3.5-2B's, only the merger is d3-nano's).

| Kernels | Artifacts | Compiler |
|---|---|---|
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `attn.xclbin` | `xclbins-Qwen3_5_bidirectional_attention_prefill` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_5_2B-VISION-MM` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_5_2B-VISION-ATTENTION` | Peano |

#### `d3:mini` → `D3-Mini-NPU2`

The same engine and the same text kernels as `d3:lite` and `d3:nano` (vllm-sr/d3-mini: the Qwen3.5-4B text tower, 32
layers, hidden 2560, 16 heads / 4 kv heads, 16 key / 32 value linear-attention heads); the engine reads its dimensions
from the package. Requests with images run Qwen3.5-4B's vision tower (24 layers, width 1024, output 2560; every tensor,
merger included, is stock Qwen3.5-4B's).

| Kernels | Artifacts | Compiler |
|---|---|---|
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `attn.xclbin` | `xclbins-Qwen3_5_bidirectional_attention_prefill` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_5_4B-VISION-MM` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_5_4B-VISION-ATTENTION` | Peano |

#### `d3:turbo` → `D3-Turbo-NPU2`

The same engine and the same text kernels as `d3:lite`, `d3:nano` and `d3:mini` (vllm-sr/d3-turbo: the Qwen3.5-27B text
tower, 64 layers = 48 linear + 16 full attention, hidden 5120, 24 heads / 4 kv heads, 16 key / 48 value linear-attention
heads), with `prefix_bidir` attention (the state block attends within itself, the question block to everything) and the
state-once API; the engine reads its dimensions from the package. `attn.xclbin` is the same bidirectional bitstream as the
other d3 sizes (GQA 6 runs as 8 passes of 4 heads). Requests with images run Qwen3.5-27B's vision tower (27 layers, width
1152, output 5120) on the `qwen3.5:9b` vision kernels, which have the same tower shape (the merger output width is a
host-side parameter), so no new vision kernel is built.

| Kernels | Artifacts | Compiler |
|---|---|---|
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `attn.xclbin` | `xclbins-Qwen3_5_bidirectional_attention_prefill` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_5_9B-VISION-MM` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_5_9B-VISION-ATTENTION` | Peano |

#### `d3:ultra` → `D3-Ultra-NPU2`

vllm-sr/d3-ultra is a new fine-tune of the d3-turbo architecture (identical config, tokenizer and engine); it uses exactly the same four kernels as `d3:turbo`.
New in the host-side format only: `prompt_layout` `header-v1` and `temperature_by_questions`.

| Kernels | Artifacts | Compiler |
|---|---|---|
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `attn.xclbin` | `xclbins-Qwen3_5_bidirectional_attention_prefill` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_5_9B-VISION-MM` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_5_9B-VISION-ATTENTION` | Peano |

#### `d3:flash` → `D3-Flash-NPU2`

vllm-sr/d3-flash is the Qwen3.5-9B size of the d3 family (32 layers, hidden 4,096, FFN 12,288, 16 query / 4 kv heads = GQA 4, 16 key / 32 value linear-attention
heads; `noncausal_full_attention`, no new host-side format). Its vision tower is `qwen3.5:9b`'s (merger out 4,096), so it uses the same four kernels as `d3:turbo`
and `d3:ultra`; no kernel is built for it.

| Kernels | Artifacts | Compiler |
|---|---|---|
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `attn.xclbin` | `xclbins-Qwen3_5_bidirectional_attention_prefill` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_5_9B-VISION-MM` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_5_9B-VISION-ATTENTION` | Peano |

#### `d3:edge` → `D3-Edge-NPU2`

vllm-sr/d3-edge is the Qwen3.5-0.8B size of the d3 family, pruned to 12 layers (hidden 1,024, FFN 3,584, 8 query / 2 kv heads = GQA 4, 16 key / 16 value
linear-attention heads, tied embedding; `noncausal_full_attention`, no new host-side format). Its layer pattern is not every 4th layer: full attention sits at
layers 2, 5, 8, 9, 10 and 11, and the engine reads it from `layer_types`. Its vision tower is `qwen3.5:0.8b`'s (12 layers, width 768, merger out 1,024), so it uses
that model's two vision kernels; no kernel is built for it.

| Kernels | Artifacts | Compiler |
|---|---|---|
| `mm.xclbin` | `xclbins-dequant_mm_512x512x512` | Peano |
| `attn.xclbin` | `xclbins-Qwen3_5_bidirectional_attention_prefill` | Peano |
| `vision_mm.xclbin` | `xclbins-MM_host7_mt_variant_bias_QWEN3_5_0_8B-VISION-MM` | Peano |
| `vision_attn.xclbin` | `xclbins-VisionAttention_hostAttention_whoel_array_QWEN3_5_0_8B-VISION-ATTENTION` | Peano |

### SmolVLA

#### `smolvla:500m` → `SmolVLA-500M-NPU2`

| Kernels | Artifacts | Compiler |
|---|---|---|
| `expert.xclbin` | `xclbins-SMOLVLA_expert` | Peano |
| `text.xclbin` | `xclbins-SMOLVLA_text` | Peano |
| `vision.xclbin` | `xclbins-SMOLVLA_vision` | Peano |
