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
| English      | WER (%)    | 2.89 | 2.98 | +0.09 (−0.08 … +0.26) |
| Mandarin     | CER (%)    | 8.96 | 8.96 | +0.00 |

The decoder is stored as **Q8_0**: it is lossless for this model, while 4-bit quantization (Q4_0, Q4_1, Q4_K) is not (empty transcripts, or Mandarin answered in English). Mandarin CER is dominated by number and punctuation style, not by recognition errors, which is why the absolute value is higher than the English one.

For recordings cut at 0.5 s … 60 s (around the encoder's 1 s chunk and 8 s window edges), the FLM transcripts are identical to the BF16 original's in all 18 cases.

---

### 🚀 Speed

Prompt processing (audio encoder + prefill) per recording length, and decoding speed:

| **Audio length** | **Encode + prefill** | **per second of audio** | **Decoding (tokens/s)** |
|------------------|---------------------:|------------------------:|------------------------:|
| 0 – 4 s          | 0.46 s | 144 ms | 28.7 |
| 4 – 8 s          | 0.76 s | 127 ms | 28.7 |
| 8 – 12 s         | 1.09 s | 112 ms | 28.6 |
| 12 – 20 s        | 1.64 s | 106 ms | 28.4 |

End to end through the server (normal request): a 68 s recording takes about 10 s, a 205 s recording 28 s, a 410 s recording 53 s (about 0.13 × real time; recordings longer than a minute are transcribed in pieces of about a minute). Streaming (stream=true): 0.5 × real time at 480 ms chunks, 0.8 × at 240 ms, 0.9 × at 200 ms, 1.1 × at the paper's 160 ms; long recordings stay at the same rate.

---

### 🧠 Memory

About 5 GB resident after loading, about 7 GB while transcribing a minute-long recording.
