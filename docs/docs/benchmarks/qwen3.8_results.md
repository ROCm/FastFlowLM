---
layout: docs
title: Qwen 3.8
parent: Benchmarks
nav_order: 14
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance of Qwen 3.8 on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v1.0.7.  
> - Under FLM's default NPU power mode (Performance)    
> - Newer versions may deliver improved performance.
> - Fine-tuned models show performance comparable to their base models.   
> - `qwen3.8-mtp:27b` decodes with its built-in MTP draft head, so decoding speed is **prompt-dependent** — predictable text (code, structured output, long reasoning chains) accepts more drafts and decodes faster than highly creative text. See the [model card](https://fastflowlm.com/docs/models/qwen/#-model-card-qwen38-27b) for how speculation works.
> - Benchmarks for this model are capped at 8k context.

---

### **Test System 1:** 

AMD Ryzen™ AI 5 340 (Kraken Point) with 64 GB DRAM; performance is comparable to other Kraken Point systems.

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/bench/qwen38_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/bench/qwen38_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|
| **Qwen3.8-27B**    | NPU (FLM)    | 1.06 | 1.86 | 1.50 | 1.32 | 

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|
| **Qwen3.8-27B**    | NPU (FLM)    | 104.94 | 110.56 | 113.05 | 112.81 | 

---

### 🚀 Time to First Token (TTFT, Seconds, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|
| **Qwen3.8-27B**    | NPU (FLM)    | 9.35 | 17.62 | 34.34 | 68.71 | 

---

### 🚀 Prefill TTFT with Image Input (Seconds)

Prefill time-to-first-token (TTFT) for Qwen3.8-27B on NPU (FastFlowLM) with different image resolutions.

**Mid Resolution Images:**

| Model        | HW  | 720p (1280×720) | 1080p (1920×1080) | 
|--------------|-----------|----------------:|------------------:|
| **Qwen3.8-27B**  | NPU (FLM) | 15.4 | 34.7 |



> This test uses a short prompt: “Describe this image.”
