---
layout: docs
title: Confucius4-R2T2
parent: Benchmarks
nav_order: 23
---

## ⚡ Accuracy and Performance Benchmarks

This section reports the accuracy and the speed of Confucius4-R2T2 on the NPU with FastFlowLM (FLM).

> **Note:**
> - Results are based on a FastFlowLM build of the `r2t2` family (v1.0.13), under FLM's default NPU power mode (Performance).
> - The audio encoder and the prompt processing run on the CPU, the 28 decoder layers run on the NPU; the CPU numbers below depend on the host CPU.
> - Newer versions may deliver improved performance.

---

### **Test System:**

AMD Ryzen™ AI 9 HX 370 (Strix Point), 24 CPU threads, 59 GB DRAM.

---

### 🎯 Accuracy (relative to the original BF16 model)

The same 180 recordings were transcribed by the original BF16 model (PyTorch, CPU) and by FLM through `v1/audio/transcriptions`: 120 English utterances (LibriSpeech test-clean, up to 20 s) and 60 Mandarin utterances (FLEURS cmn_hans_cn). Error rates are against the reference text after lower-casing, removing punctuation and writing Chinese numerals as digits; the difference is paired per utterance (95% bootstrap interval).

| **Language** | **Metric** | **BF16 original** | **FLM (NPU)** | **Difference** |
|--------------|------------|------------------:|--------------:|---------------:|
| English      | WER (%)    | 2.89 | 2.81 | −0.09 (−0.31 … +0.09) |
| Mandarin     | CER (%)    | 8.96 | 6.37 | −2.60 (−9.84 … +1.37) |

The NPU decode reads mixed precision: Q4_1 for q/k/o/gate/up and the output head, Q8_0 for v_proj and down_proj (the prompt is processed on the CPU with Q8_0 throughout). 4-bit quantization of every matrix (Q4_0, Q4_1, Q4_K) breaks this model (empty transcripts, or Mandarin answered in English); keeping v_proj and down_proj at 8 bit does not change its accuracy, and decodes ~24 % faster than Q8_0 everywhere. Mandarin CER is dominated by number and punctuation style (digits vs Chinese numerals), not by recognition errors, which is why the absolute values are higher than the English ones and why their difference is not significant.

For recordings cut at 0.5 s … 60 s (around the encoder's 1 s chunk and 8 s window edges), the FLM transcripts are identical to the BF16 original's in all 18 cases.

---

### 🚀 Speed

Prompt processing (audio encoder + prefill) per recording length, and decoding speed:

| **Audio length** | **Encode + prefill** | **per second of audio** | **Decoding (tokens/s)** |
|------------------|---------------------:|------------------------:|------------------------:|
| 0 – 4 s          | 0.13 s | 43 ms | 38.5 |
| 4 – 8 s          | 0.21 s | 35 ms | 38.4 |
| 8 – 12 s         | 0.31 s | 30 ms | 38.0 |
| 12 – 20 s        | 0.45 s | 28 ms | 37.8 |

End to end through the server (normal request): a 68 s recording takes about 8 s, a 205 s recording 22 s, a 410 s recording 42 s (about 0.1 × real time; recordings longer than a minute are transcribed in pieces of about a minute). Streaming (stream=true): 0.45 × real time at 480 ms chunks, 0.75 × at 240 ms, about 0.9 × at 200 ms, 1.05 × at the paper's 160 ms; long recordings stay at the same rate.

---

### 🧠 Memory

About 5 GB resident after loading, about 7 GB while transcribing a minute-long recording.
