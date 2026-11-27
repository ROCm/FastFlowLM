---
layout: docs
title: Inkling
parent: Benchmarks
nav_order: 24
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:**
> - Inkling-2-Mini is supported from FastFlowLM v1.0.14, the version these numbers were measured on (`flm bench`, 2 iterations, Q4_K weights); prefill speed is prompt tokens divided by time to first token.
> - Under FLM's default NPU power mode (Performance)
> - Newer versions may deliver improved performance.
> - Inkling-2-Mini is a mixture-of-experts model (26.4B parameters, 3.4B active) whose 12 middle layers run three times per token.

---

### **Test System:**

AMD Ryzen™ AI Max+ PRO 395 (Strix Halo) with 128 GB DRAM, Linux. The model needs about 20 GB of memory, so use a system with 64 GB or more.

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Inkling-2-Mini**  | XDNA2    | 11.6 | 11.5 | 11.3 | 10.8 | 9.7 | 8.3 |

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Inkling-2-Mini**  | XDNA2    | 742 | 975 | 955 | 1,080 | 1,029 | 931 |

---

### 🖼️ Image encoding

The vision tower runs its matrix multiplications and attention on the NPU: about 0.3 s for a 640 × 480 image (192 image tokens), 0.6 s at 525 tokens and 7.6 s at the 4,096-token maximum (2560 × 2560 pixels).

---

### 🎯 Accuracy

Mean KL divergence of the next-token distribution against the BF16 model, 32 × 512 tokens (lower is better):

| Text | Q4_K | Q4_K with the unsloth imatrix (shipped) |
| :--- | ---: | ---: |
| wikitext-2 | 0.0184 | 0.0162 |
| code and documentation | 0.0135 | 0.0115 |

Image embeddings: relative L2 error 0.11 to 0.15 against the BF16 reference (48 to 4,096 image tokens); the BF16 reference itself is 0.06 away from FP32.
