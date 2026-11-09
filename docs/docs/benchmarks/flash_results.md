---
layout: docs
title: Flash Models
parent: Benchmarks
nav_order: 22
---

## ⚡ Flash Model Benchmarks

This section reports the performance of **Flash** models on NPU with FastFlowLM (FLM).

> **Note:**
> - Results are based on FastFlowLM v1.0.12 (main build), under FLM's default NPU power mode (Performance).
> - Newer versions may deliver improved performance.
> - Flash models are **single-turn** with a **fixed 1k-token context**, so they are benchmarked with short prompts (64 to 1,024 tokens) rather than the long-context sweeps used for general chat models.
> - Flash only changes how **prefill** runs. **Decode speed matches the non-flash model**, so decoding numbers are taken from the corresponding benchmark page.
> - Prompt lengths are the server-reported prompt tokens, chat template included (the 1,024 case is 1,023 tokens). Text-only prompts, no system prompt, median of 7 runs.
> - The 64-token and 128-token cases take about the same time, so prefill TPS at 64 tokens is correspondingly lower.

---

### **Test System:**

AMD Ryzen™ AI 9 370 (Strix Point) with 32 GB DRAM; performance is comparable to other Strix Point and Strix Halo systems.

---

### 🚀 Time to First Token (TTFT, seconds, with different prompt lengths)

| **Model**               | **HW**    | **64** | **128** | **256** | **512** | **1024** |
|-------------------------|-----------|-------:|--------:|--------:|--------:|---------:|
| **gemma4e-flash:e2b**   | XDNA2 | 0.20 | 0.22 | 0.36 | 0.73 | 1.53 |
| **gemma4e-flash:e4b**   | XDNA2 | 0.33 | 0.34 | 0.62 | 1.20 | 2.57 |
| **nemotron3.5-bolt-dflash-flash:e4b** | XDNA2 | 0.32 | 0.33 | 0.60 | 1.18 | 2.53 |
| **qwen3vl-flash:4b**    | XDNA2 | 0.24 | 0.24 | 0.45 | 0.85 | 1.83 |
| **hy-mt2-flash:1.8b**   | XDNA2 | 0.13 | 0.14 | 0.23 | 0.43 | 0.84 |

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

Prompt tokens divided by TTFT (TTFT includes generating the first token).

| **Model**               | **HW**    | **64** | **128** | **256** | **512** | **1024** |
|-------------------------|-----------|-------:|--------:|--------:|--------:|---------:|
| **gemma4e-flash:e2b**   | XDNA2 | 322 | 593 | 721 | 705 | 668 |
| **gemma4e-flash:e4b**   | XDNA2 | 193 | 373 | 412 | 427 | 398 |
| **nemotron3.5-bolt-dflash-flash:e4b** | XDNA2 | 201 | 392 | 427 | 434 | 405 |
| **qwen3vl-flash:4b**    | XDNA2 | 271 | 526 | 570 | 601 | 558 |
| **hy-mt2-flash:1.8b**   | XDNA2 | 490 | 923 | 1,093 | 1,197 | 1,214 |

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ 1k context)

| **Model**               | **HW**    | **1k** | **Source** |
|-------------------------|-----------|-------:|------------|
| **gemma4e-flash:e2b**   | XDNA2 | 22.6 | [Gemma 4](../gemma4_results/) (`gemma4-it:e2b`) |
| **gemma4e-flash:e4b**   | XDNA2 | 12.6 | [Gemma 4](../gemma4_results/) (`gemma4-it:e4b`) |
| **nemotron3.5-bolt-dflash-flash:e4b** | XDNA2 | 18.2 | [Nemotron Bolt DFlash](../nemotron3.5-bolt-dflash_results/) (`nemotron3.5-bolt-dflash:e4b`) |
| **qwen3vl-flash:4b**    | XDNA2 | 19.6 | [Qwen3](../qwen3_results/) (Qwen 3 4B) |
| **hy-mt2-flash:1.8b**   | XDNA2 | 52.2 | [Hy-MT2](../hy-mt2_results/) |

> Context is fixed at 1k tokens for every flash model, so 1k is the only applicable decode starting point.
>
> `nemotron3.5-bolt-dflash-flash:e4b` decodes with speculation, so its speed depends on the text: the value above is `flm bench` (about 30 decoded tokens, mean of 3 runs, 15 to 24 tokens/s between runs). Code and structured answers reach 33 to 40 tokens/s, open prose about 20 (see its [model card](../../models/nemotron3.5-bolt-dflash/#-model-card-nvidia-nemotron-35-bolt-e4b--dflash-flash)). The three rows of this model were measured on an AMD Ryzen™ AI MAX+ PRO 395 (Strix Halo) with the same method; on that machine `gemma4e-flash:e4b` gives 0.32 / 0.32 / 0.58 / 1.17 / 2.52 s.
