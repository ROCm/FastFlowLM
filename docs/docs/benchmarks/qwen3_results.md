---
layout: docs
title: Qwen 3
parent: Benchmarks
nav_order: 2
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance of Qwen 3 on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v0.9.31.  
> - Under FLM's default NPU power mode (Performance)    
> - Newer versions may deliver improved performance.
> - Fine-tuned models show performance comparable to their base models.   

---

### **Test System 1:** 

AMD Ryzen™ AI 7 350 (Kraken Point) with 32 GB DRAM; performance is comparable to other Kraken Point systems.

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/bench/qwen3_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/bench/qwen3_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Qwen 3 0.6B**  | XDNA2    | 66.5 | 57.5 | 44.5 | 31.0 | 19.6 | 14.1|
| **Qwen 3 1.7B**  | XDNA2    | 40.2 | 35.8 | 30.8 | 23.7 | 16.4 | 12.5|
| **Qwen 3 4B**    | XDNA2    | 19.6 | 18.1 | 16.3 | 13.7 | 10.6 | 8.5| 
| **Qwen 3 8B**    | XDNA2    | 11.9 | 11.5 | 11.1 | 10.4 | 8.7 | 7.2|

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Qwen 3 0.6B**  | XDNA2    | 1494 |	2003 |	2165 |	1981 |	1485 |	907 |
| **Qwen 3 1.7B**  | XDNA2    | 956 |	1263 |	1434 |	1411 |	1143 |	768|
| **Qwen 3 4B**    | XDNA2    | 509 |	582 |	615 |	576 |	448 |	303| 
| **Qwen 3 8B**    | XDNA2    | 357 |	435 |	457 |	442 |	367 |	260|

---

### 🚀 Prefill TTFT with Image Input (Seconds)

Prefill time-to-first-token (TTFT) for Qwen3-VL-4B on NPU (FastFlowLM) with different image resolutions.

**Mid Resolution Images:**

| Model        | HW  | 720p (1280×720) | 1080p (1920×1080) | 
|--------------|-----------|----------------:|------------------:|
| Qwen3-VL-4B  | XDNA2 |            3.3 |               7.4 |


**High Resolution Images:**

| Model        | HW  | 2K (2560×1440) | 4K (3840×2160) |
|--------------|-----------|---------------:|---------------:|
| Qwen3-VL-4B  | XDNA2 |           13.7 |             41.2 |

> This test uses a short prompt: “Describe this image.”