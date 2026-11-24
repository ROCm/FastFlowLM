---
layout: docs
title: Hy-MT3
parent: Benchmarks
nav_order: 14
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:**
> - Measured on FastFlowLM v1.0.14 (Q4_K weights); prefill speed is prompt tokens divided by time to first token.
> - Under FLM's default NPU power mode (Performance)
> - Newer versions may deliver improved performance.
> - Hy-MT3 is a dedicated translation model tuned for short, single-turn requests, so it is benchmarked up to 8k context length. It runs on the same NPU kernels as [Hy-MT2](../hy-mt2_results/), so the two perform alike.

---

### **Test System:**

AMD Ryzen™ AI 9 370 (Strix Point) with 32 GB DRAM; performance is comparable to other Strix Point and Strix Halo systems.

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|
| **Hy-MT3-1.8B**  | XDNA2    | 42.3 | 40.5 | 38.4 | 33.8 |

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|
| **Hy-MT3-1.8B**  | XDNA2    | 1,636 | 1,569 | 1,376 | 1,120 |
