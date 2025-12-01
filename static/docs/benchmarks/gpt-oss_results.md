---
layout: docs
title: gpt-oss
parent: Benchmarks
nav_order: 4
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v0.9.20.
> - Under FLM's default NPU power mode (Performance)  
> - Test system spec: AMD Ryzen™ AI 7 350 (Krakan Point) with 32 GB DRAM. 
> - Newer versions may deliver improved performance. 

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/gpt-oss_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/gpt-oss_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **Hardware**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |**64k** | **128k** |**Model**|
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------|
| **gpt-oss-20b**  | NPU (FLM)    | 18.2|	17.9|	17.3|	16.2|	14.4|	11.8|	8.7|	5.7| **gpt-oss-20b**  | 

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **Hardware**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |**Model**|
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|---------|
| **gpt-oss-20b**  | NPU (FLM)    | 198|	286|	354|	342|	267|	173| **gpt-oss-20b**  | 