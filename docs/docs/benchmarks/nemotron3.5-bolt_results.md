---
layout: docs
title: Nemotron Bolt
parent: Benchmarks
nav_order: 21
---

## ⚡ Performance and Efficiency Benchmarks

NVIDIA Nemotron-3.5-Bolt-E4B (`nemotron3.5-bolt:e4b`) is a post-trained Gemma-4-E4B-it and runs on the same engine and
kernels as `gemma4-it:e4b`, so it performs like it. This page reports the measurements for the Bolt package.

> **Note:**
> - Results are based on FastFlowLM 1.0.10 plus the Nemotron Bolt changes (a local build), under FLM's default NPU
>   power mode (Performance). Newer versions may deliver improved performance.
> - The Gemma 4 E4B row is measured in the same session on the same build.

---

### **Test System:**

AMD Ryzen™ AI MAX+ PRO 395 (Strix Halo) with 128 GB DRAM (HP ZBook Ultra G1a).

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**            | **HW**    | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|----------------------|-----------|-------:|-------:|-------:|-------:|--------:|--------:|
| **Nemotron Bolt E4B**| NPU (FLM) | 12.8   | 12.5   | 11.9   | 10.9   | 9.2     | 7.2     |
| Gemma 4 E4B          | NPU (FLM) | 12.8   | 12.5   | 11.9   | 10.9   | 9.3     | 7.2     |

> Nemotron Bolt supports up to 64k tokens of context (the default is 32k).

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**            | **HW**    | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|----------------------|-----------|-------:|-------:|-------:|-------:|--------:|--------:|
| **Nemotron Bolt E4B**| NPU (FLM) | 438    | 542    | 606    | 643    | 611     | 516     |
| Gemma 4 E4B          | NPU (FLM) | 439    | 546    | 609    | 644    | 612     | 516     |

### 🚀 Time to First Token (Seconds)

| **Model**            | **HW**    | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|----------------------|-----------|-------:|-------:|-------:|-------:|--------:|--------:|
| **Nemotron Bolt E4B**| NPU (FLM) | 2.2    | 3.6    | 6.4    | 12.1   | 25.4    | 60.1    |

---

## 🎯 Accuracy

The weights are converted from NVIDIA's Q4_K_M release (see the [model card](https://huggingface.co/FastFlowLM/Nemotron-3.5-Bolt-E4B-NPU2)).

### Task accuracy: GSM8K (thinking off, greedy)

| **Run** | **Problems** | **Nemotron Bolt E4B** |
|---------|-------------:|----------------------:|
| FastFlowLM CI bench, XRT | 50 | 94.0% |
| FastFlowLM CI bench, HRX | 50 | 94.0% |

A paired check against the BF16 model (first 100 GSM8K test problems, one prompt, up to 512 new tokens): the NPU build
solves 87, the BF16 model solves 90 (difference −3.0 points, 95% confidence interval −8.9 to +2.9; 6 problems only BF16
solves, 3 only the NPU build). With 100 problems this cannot tell a small loss from none.

### Closeness to the BF16 model

Teacher-forced over 6,317 positions of the BF16 model's own answers to 53 prompts (CPU simulation of the quantized weights):

| **Weights** | **Top-1 agreement with BF16** | **Mean KL(BF16 ‖ quantized)** |
|-------------|------------------------------:|------------------------------:|
| NVIDIA Q4_K_M as released | 94.7% | 0.044 |
| FastFlowLM package (Q4_K_M converted to Q4_1) | 94.1% | 0.054 |
| BF16 quantized straight to Q4_1 | 92.9% | 0.076 |

The simulation keeps the tied output head at 8 bits; on the NPU it is 4-bit (Q4_1), as for Gemma 4 E4B.
