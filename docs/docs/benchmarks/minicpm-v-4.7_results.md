---
layout: docs
title: MiniCPM-V 4.7
parent: Benchmarks
nav_order: 15
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance of MiniCPM-V 4.7 on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v1.0.8.  
> - Under FLM's default NPU power mode (Performance)    
> - Newer versions may deliver improved performance.
> - Fine-tuned models show performance comparable to their base models.   

---

### **Test System 1:** 

AMD Ryzen™ AI 5 340 (Kraken Point) with 64 GB DRAM; performance is comparable to other Kraken Point systems.

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/bench/minicpm_v47_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/bench/minicpm_v47_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **MiniCPM-V-4.7-1B**    | XDNA2    | 47.7 | 45.9 | 43.0 | 32.7 | 31.6 | 23.8 | 

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **MiniCPM-V-4.7-1B**    | XDNA2    | 1504.7 | 1624.0 | 1648.2 | 1808.8 | 1941.1 | 1666.3 |

---

### 🚀 Prefill TTFT with Image Input (Seconds)

Prefill time-to-first-token (TTFT) for MiniCPM-V-4.7-1B on NPU (FastFlowLM) with different image resolutions.

**Mid Resolution Images:**

| Model        | HW  | 720p (1280×720) | 1080p (1920×1080) | 
|--------------|-----------|----------------:|------------------:|
| MiniCPM-V-4.7-1B  | XDNA2 | 2.1 | 2.5 |

**High Resolution Images:**

| Model        | HW  | 2K (2560×1440) | 4K (3840×2160) |
|--------------|-----------|---------------:|---------------:|
| MiniCPM-V-4.7-1B  | XDNA2 | 2.5 | 2.5 |

> This test uses a short prompt: “Describe this image.”

> TTFT stays flat from 1080p up: MiniCPM-V splits large images into at most 9 tiles, so higher resolutions add no extra image tokens.

---

---

## MiniCPM-V-4.7-35B-A3B (`minicpm-v-4.7-moe:35b-a3b`)

The Mixture-of-Experts model (35B parameters, about 3B active per token). Same text engine and NPU kernels as
[Qwen3.6-35B-A3B](qwen3.6_results/), MiniCPM's own picture handling.

> **Note:**
> - Results are based on FastFlowLM v1.0.12, under FLM's default NPU power mode (Performance).
> - One run per cell; this box is shared and decoding speed varies by about ±15 % between runs.

### **Test System 2:**

AMD Ryzen™ AI MAX+ PRO 395 (Strix Halo) with 128 GB DRAM.

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **MiniCPM-V-4.7-35B-A3B** | XDNA2 | 16.2 | 16.1 | 15.2 | 14.6 | 13.1 | 10.9 |

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **MiniCPM-V-4.7-35B-A3B** | XDNA2 | 148 | 219 | 278 | 288 | 284 | 269 |

### 🚀 Prefill TTFT with Image Input (Seconds)

| Model        | HW  | 720p (1280×720) | 1080p (1920×1080) | 2K (2560×1440) | 4K (3840×2160) |
|--------------|-----------|----------------:|------------------:|---------------:|---------------:|
| MiniCPM-V-4.7-35B-A3B | XDNA2 | 7.8 | 9.1 | 8.8 | 8.4 |

> This test uses a short prompt: “Describe this image.” TTFT stays flat from 1080p up: pictures are split into at most 9 tiles.

---

### 🎯 Accuracy

The weights are converted from the llama.cpp Q4_K_M release, with the token embedding and the output head taken from the BF16
checkpoint (see the [model card](https://huggingface.co/FastFlowLM/MiniCPM-V-4.7-35B-A3B-NPU2)). Greedy decoding; "BF16" is the
upstream checkpoint run with Hugging Face Transformers on the CPU.

| **Check** | **FastFlowLM (NPU)** | **llama.cpp Q4_K_M (same GGUF)** | **BF16** |
|-----------|---------------------:|---------------------------------:|---------:|
| GSM8K, first 100 test problems (thinking off) | 98 / 100 | 97 / 100 | 99 / 100 |
| DocVQA, 50 validation questions (ANLS) | 97.50 | 97.42 | 97.42 |

GSM8K allows up to 1024 new tokens. Paired with the BF16 model on those 100 problems, the NPU build is 1.0 point lower (95% confidence interval −2.4 to +4.4 points; 2 problems only BF16 solves, 1 only the NPU build). With 50–100 questions these checks cannot tell a loss of a point or two from none; they show that nothing is broken.
