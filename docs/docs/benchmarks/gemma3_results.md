---
layout: docs
title: Gemma 3
parent: Benchmarks
nav_order: 3
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v0.9.31.
> - Under FLM's default NPU power mode (Performance)  
> - Newer versions may deliver improved performance.
> - Fine-tuned models show performance comparable to their base models. 

---

### **Test System 1:** 

AMD Ryzen™ AI 7 350 (Kraken Point) with 32 GB DRAM; performance is comparable to other Kraken Point systems.

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/bench/gemma3_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/bench/gemma3_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |**64k** | **128k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|
| **Gemma 3 1B**  | XDNA2    | 41.1	| 40.5	|	39.5	|	37.3	|	33.6	|	27.9|	OOC|	OOC|
| **Gemma 3 4B**  | XDNA2    | 18.2	| 18.0	| 17.8	| 17.3	| 16.3	| 14.8 |	13.2 |	11.2 |

> OOC: Out Of Context Length  
> Each LLM has a maximum supported context window. For example, the gemma3:1b model supports up to 32k tokens.

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Gemma 3 1B**   | XDNA2    | 1004 |	1321|	1546 |	1720 |	1785 |	1755|
| **Gemma 3 4B**   | XDNA2    | 528 |	654 |	771 |	881 |	936 |	926|

---

### 🚀 Prefill TTFT with Image (Seconds)

| **Model**        | **HW**       | **Image** |
|------------------|--------------------|--------:|
| **Gemma 3 4B**   | XDNA2    | 2.6|

> This test uses a short prompt: “Describe this image.”
