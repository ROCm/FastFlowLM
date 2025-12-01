---
layout: docs
title: LiquidAI/LFM2
parent: Benchmarks
nav_order: 5
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v0.9.19.
> - Under FLM's default NPU power mode (Performance)  
> - Test system spec: AMD Ryzen™ AI 7 350 (Krakan Point) with 32 GB DRAM. 
> - Newer versions may deliver improved performance. 

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **Hardware**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |**Model**|
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|---------|
| **LFM2-1.2B**  | NPU (FLM)    | 41.5|	40.6|	39.6|	37.1|	33.3|    28| **LFM2-1.2B**  | 

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **Hardware**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |**Model**|
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|---------|
| **LFM2-1.2B**  | NPU (FLM)    | 1384|	 1781|	2038|	1905|	1503|	1024| **LFM2-1.2B**  | 