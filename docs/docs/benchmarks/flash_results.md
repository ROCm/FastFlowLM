---
layout: docs
title: Flash Models
parent: Benchmarks
nav_order: 22
---

## ⚡ Flash Model Benchmarks

This section reports the performance of **Flash** models on NPU with FastFlowLM (FLM).

> **Note:**
> - Under FLM's default NPU power mode (Performance)
> - Newer versions may deliver improved performance.
> - Flash models are **single-turn** with a **fixed 1k-token context**, so they are benchmarked at a short (128-token) prompt, the case Flash is tuned for, rather than the long-context sweeps used for general chat models.
> - Flash only changes how **prefill** runs. **Decode speed matches the non-flash model**, so decoding numbers are taken from the corresponding benchmark page.

---

### **Test System:**

AMD Ryzen™ AI 9 370 (Strix Point) with 32 GB DRAM; performance is comparable to other Strix Point and Strix Halo systems.

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, @ 128-token prompt)

| **Model**               | **HW**    | **128** |
|-------------------------|-----------|--------:|
| **gemma4e-flash:e2b**   | NPU (FLM) | ~390 |
| **gemma4e-flash:e4b**   | NPU (FLM) | ~256 |
| **qwen3vl-flash:4b**    | NPU (FLM) | ~350 |
| **hy-mt2-flash:1.8b**   | NPU (FLM) | TBD |

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ 1k context)

| **Model**               | **HW**    | **1k** | **Source** |
|-------------------------|-----------|-------:|------------|
| **gemma4e-flash:e2b**   | NPU (FLM) | 22.6 | [Gemma 4](../gemma4_results/) (`gemma4-it:e2b`) |
| **gemma4e-flash:e4b**   | NPU (FLM) | 12.6 | [Gemma 4](../gemma4_results/) (`gemma4-it:e4b`) |
| **qwen3vl-flash:4b**    | NPU (FLM) | 19.6 | [Qwen3](../qwen3_results/) (Qwen 3 4B) |
| **hy-mt2-flash:1.8b**   | NPU (FLM) | 52.2 | [Hy-MT2](../hy-mt2_results/) |

> Context is fixed at 1k tokens for every flash model, so 1k is the only applicable decode starting point.
