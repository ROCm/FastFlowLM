---
layout: docs
title: Gemma 4
parent: Benchmarks
nav_order: 3
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v0.9.39.
> - Under FLM's default NPU power mode (Performance)  
> - Newer versions may deliver improved performance.
> - Fine-tuned models show performance comparable to their base models. 

---

### **Test System 1:** 

AMD Ryzen™ AI 7 350 (Kraken Point) with 32 GB DRAM; performance is comparable to other Kraken Point systems.

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/bench/gemma4_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/bench/gemma4_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Gemma 4 E2B**  | NPU (FLM)    | 20.5	| 19.8	| 18.6	| 16.9	| 13.1 |	9.6 |

> OOC: Out Of Context Length  
> Each LLM has a maximum supported context window. For example, the gemma4:1b model supports up to 32k tokens.

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Gemma 4 E2B**   | NPU (FLM)    | 689 |	874 |	1019 |	1009 |	939 |	719|

---

### 🚀 Prefill TTFT with Image (Seconds)

| **Model**        | **HW**       | **Image** |
|------------------|--------------------|--------:|
| **Gemma 4 E2B**   | NPU (FLM)    | 1.7|

> This test uses a short prompt: “Describe this image.”
