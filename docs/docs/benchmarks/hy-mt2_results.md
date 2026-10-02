---
layout: docs
title: Hy-MT2
parent: Benchmarks
nav_order: 13
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:**
> - Results are based on FastFlowLM v1.0.5.
> - Under FLM's default NPU power mode (Performance)
> - Newer versions may deliver improved performance.
> - Hy-MT2 is a dedicated translation model tuned for short, single-turn requests, so it is benchmarked up to 16k max context length rather than the longer sweeps used for general chat models.

---

### **Test System:**

AMD Ryzen™ AI 9 370 (Strix Point) with 32 GB DRAM; performance is comparable to other Strix Point and Strix Halo systems.

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/bench/hy_mt2_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/bench/hy_mt2_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|--------:|
| **Hy-MT2-1.8B**  | NPU (FLM)    | 52.2 | 46.2 | 38.9 | 28.4 | 19 |

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|--------:|
| **Hy-MT2-1.8B**  | NPU (FLM)    | 1,068 | 1,051 | 1,013 | 912 | 723 |


