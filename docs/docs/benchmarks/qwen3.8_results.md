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
| **Qwen3.8-27B**    | XDNA2    | 2.04 | 2.23 | 2.70 | 2.31 | 

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|
| **Qwen3.8-27B**    | XDNA2    | 104.94 | 110.56 | 113.05 | 112.81 | 

---

### 🚀 Time to First Token (TTFT, Seconds, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|
| **Qwen3.8-27B**    | XDNA2    | 9.35 | 17.62 | 34.34 | 68.71 | 

---

### 🚀 Prefill TTFT with Image Input (Seconds)

Prefill time-to-first-token (TTFT) for Qwen3.8-27B on NPU (FastFlowLM) with different image resolutions.

**Mid Resolution Images:**

| Model        | HW  | 720p (1280×720) | 1080p (1920×1080) | 
|--------------|-----------|----------------:|------------------:|
| **Qwen3.8-27B**  | XDNA2 | 15.4 | 34.7 |



> This test uses a short prompt: “Describe this image.”

---

### 🚀 DFlash 2 draft: decode speed against the MTP head (TPS, greedy, 160 new tokens, short prompts)

`qwen3.8-dflash:27b` is the same model decoded with z-lab's [DFlash 2 draft](https://fastflowlm.com/docs/models/qwen/#-model-card-qwen38-27b-with-the-dflash-2-draft) instead of the built-in MTP head. Both rows of a prompt ran the same engine build and kernels on the same machine, one request at a time, thinking off.

| **Prompt** | **`qwen3.8-mtp:27b`** | **`qwen3.8-dflash:27b`** | **Speedup** |
|---|--------:|--------:|--------:|
| Hash map explanation + Python code | 2.95 | 3.09 | 1.05x |
| TCP vs UDP | 3.39 | 3.48 | 1.03x |
| Short story (creative) | 1.71 | 1.96 | 1.14x |
| Train catch-up word problem | 4.12 | 4.91 | 1.19x |
| French Revolution bullets | 2.42 | 2.89 | 1.19x |
| SQL query + explanation | 4.10 | 4.42 | 1.08x |
| C linked-list reversal | 3.57 | 5.25 | 1.47x |
| Grade-school math word problem | 3.40 | 4.24 | 1.25x |
| Translation to French and German (60 tokens) | 2.65 | 2.30 | 0.87x |
| JSON object (three planets) | 3.98 | 4.87 | 1.22x |
| **Mean** | **3.23** | **3.74** | **1.16x** |

On a short code prompt the draft keeps about 6.6 of every 7 drafted tokens (the model then emits 7.6 tokens per step); on open prose it keeps about 3. One step costs about 1.2 s with either draft: 0.9 s to check the 8 positions and about 0.27 s to draft. Gains come from the draft predicting more of what the model will say, so they follow the text, as with the MTP head.
