---
layout: docs
title: Hy-Memory
parent: Benchmarks
nav_order: 14
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:**
> - Results are based on FastFlowLM v1.0.13 (pre-release build).
> - Under FLM's default NPU power mode (Performance)
> - Newer versions may deliver improved performance.
> - Prefill and decode speeds are the median of 3 runs, taken from the server's own timing, with a different prompt for every run so the prefix cache cannot help.

---

### **Test System:**

AMD Ryzen™ AI Max+ (Strix Halo) with 128 GB DRAM; performance is comparable to other Strix Halo systems.

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**            | **HW**   | **1k** | **2k** | **4k** | **8k** | **16k** |
|----------------------|----------|-------:|-------:|-------:|-------:|--------:|
| **Hy-Memory-30B-A3B** | XDNA2    | 19.3   | 18.4   | 16.9   | 14.5   | 11.4    |

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**            | **HW**   | **1k** | **2k** | **4k** | **8k** | **16k** |
|----------------------|----------|-------:|-------:|-------:|-------:|--------:|
| **Hy-Memory-30B-A3B** | XDNA2    | 181    | 249    | 259    | 223    | 192     |

Time to first token: 5.4 s at 1k, 15.7 s at 4k, 37 s at 8k, 83 s at 16k.
