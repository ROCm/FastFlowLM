---
layout: docs
title: LiquidAI/LFM3
parent: Benchmarks
nav_order: 22
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on the FastFlowLM release that adds LFM3-1.7B (v1.0.11).
> - Under FLM's default NPU power mode (Performance)  
> - Prefill and every decoder layer (conv, sliding-window attention, full attention) and the LM head run on the NPU.
> - Newer versions may deliver improved performance.

---

### **Test System:** 

AMD Ryzen™ AI MAX+ PRO 395 (Strix Halo) with 128 GB DRAM; performance is comparable to other Strix Halo and Strix Point systems.

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **LFM3-1.7B**  | XDNA2    | 41	|40	|39	|36	|31	|24|

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **LFM3-1.7B**  | XDNA2    | 1092 | 1543 | 1467 | 1397 | 1346 | 1223 |

---

## 🎯 Accuracy

LFM3-1.7B runs the vendor's quantization-aware-distilled Q4_0 weights. The NPU computes in bf16, so its logits differ slightly from a
PyTorch reference run on the same quantized weights.

| **Test**                                   | **PyTorch reference** | **FLM on NPU** |
|--------------------------------------------|----------------------:|---------------:|
| WikiText-2 perplexity (6.1k tokens, 8 windows) | 19.23 | 20.16 |
| GSM8K, first 150 test questions (greedy, thinking off) | n/a | 81.3% |

> **Note:** 
> - The CPU engine that shares the weights reaches 87.3% on the same 150 GSM8K questions; attention on the NPU accounts for the difference.
> - `LFM3_DECODE=conv` runs the attention layers on the host (conv layers and LM head stay on the NPU): 84.0% on the same questions, at a lower decoding speed (about 12 TPS at 32k).
> - The default (all NPU) is the faster mode, especially at long context.
