---
layout: docs
title: Gemma 3
parent: Benchmarks
nav_order: 3
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v0.9.20.
> - Under FLM's default NPU power mode (Performance)  
> - Test system spec: AMD Ryzen™ AI 7 350 (Krakan Point) with 32 GB DRAM. 
> - Newer versions may deliver improved performance. 

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/gemma3_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/gemma3_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **Hardware**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |**64k** | **128k** |**Model**|
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------|
| **Gemma 3 1B**  | NPU (FLM)    | 40.0	| 39.3	|	38.1	|	35.8	|	32.6	|	26.7|	OOC|	OOC| **Gemma 3 1B**  | 
| **Gemma 3 4B**  | NPU (FLM)    | 18.0	| 17.8	| 17.6	| 17.1	| 16.1	| 14.5 |	13.2 |	11.2 |**Gemma 3 4B**  |

> OOC: Out Of Context Length  
> Each LLM has a maximum supported context window. For example, the gemma3:1b model supports up to 32k tokens.

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **Hardware**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |**Model**|
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|---------|
| **Gemma 3 1B**   | NPU (FLM)    | 1004 |	1321|	1528 |	1645 |	1657 |	1596|**Gemma 3 1B**  |
| **Gemma 3 4B**   | NPU (FLM)    | 528 |	654 |	738 |	754 |	739 |	673|**Gemma 3 4B**  |

---

### 🚀 Prefill TTFT with image (Seconds)

| **Model**        | **Hardware**       | **Image** |
|------------------|--------------------|--------:|
| **Gemma 3 4B**   | NPU (FLM)    | 4.3|

> This test uses a short prompt: “Describe this image.”
