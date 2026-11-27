---
layout: docs
title: Confucius4-R2T2
parent: Benchmarks
nav_order: 23
---

## ⚡ Accuracy and Performance Benchmarks

This section reports the accuracy and the speed of Confucius4-R2T2 on the NPU with FastFlowLM (FLM).

> **Note:**
> - Results are based on a FastFlowLM build of the `r2t2` family (v1.0.14), under FLM's default NPU power mode (Performance).
> - The audio encoder, the prompt processing and the 28 decoder layers run on the NPU.
> - Newer versions may deliver improved performance.

---

### **Test System:**

AMD Ryzen™ AI 9 HX 370 (Strix Point), 24 CPU threads, 59 GB DRAM.

---

### 🎯 Accuracy (relative to the original BF16 model)

**LibriSpeech test-clean**, all 2620 utterances (5.4 h, 52.6k words), transcribed by the original BF16 model (PyTorch, CPU) and by FLM. Word error rate against the reference text after lower-casing and removing punctuation (numbers are not normalized: "1847" and "eighteen forty seven" count as different words for both); the difference is paired per utterance (95% bootstrap interval).

| **Model** | **WER (%)** | **Word errors** | **Difference** |
|-----------|------------:|----------------:|---------------:|
| BF16 original | 2.22 | 1178 | |
| FLM (NPU)     | 2.30 | 1220 | +0.08 (+0.02 … +0.15) |

Without the utterances that contain digits, the two are 2.11 and 2.14.

**English and Mandarin**, 180 recordings: 120 English utterances (LibriSpeech test-clean, up to 20 s) and 60 Mandarin utterances (FLEURS cmn_hans_cn). Error rates are against the reference text after lower-casing, removing punctuation and writing Chinese numerals as digits; the difference is paired per utterance (95% bootstrap interval).

| **Language** | **Metric** | **BF16 original** | **FLM (NPU)** | **Difference** |
|--------------|------------|------------------:|--------------:|---------------:|
| English      | WER (%)    | 2.89 | 2.81 | −0.09 (−0.41 … +0.21) |
| Mandarin     | CER (%)    | 8.96 | 9.30 | +0.34 (−0.27 … +1.49) |

One Mandarin recording is answered in English by both the BF16 original and FLM; without it the Mandarin CER is 5.98 (BF16) and 6.32 (FLM).

The NPU reads the decoder in mixed precision, for the prompt and for decoding alike: Q4_1 for q/k/o/gate/up and the output head, Q8_0 for v_proj and down_proj. The audio encoder runs on the NPU in bfp16 (block floating point) from its BF16 weights. 4-bit quantization of every matrix (Q4_0, Q4_1, Q4_K) breaks this model (empty transcripts, or Mandarin answered in English); keeping v_proj and down_proj at 8 bit does not change its accuracy, and decodes ~24 % faster than Q8_0 everywhere. Mandarin CER is dominated by number and punctuation style (digits vs Chinese numerals), not by recognition errors, which is why the absolute values are higher than the English ones and why their difference is not significant.

For recordings cut at 0.5 s … 60 s (around the encoder's 1 s chunk and 8 s window edges), the FLM transcripts are identical to the BF16 original's in all 18 cases.

---

### 🚀 Speed

Prompt processing (audio encoder + prefill) per recording length, and decoding speed:

| **Audio length** | **Encode + prefill** | **per second of audio** | **Decoding (tokens/s)** |
|------------------|---------------------:|------------------------:|------------------------:|
| 0 – 4 s          | 0.13 s | 42 ms | 38.4 |
| 4 – 8 s          | 0.15 s | 27 ms | 38.3 |
| 8 – 12 s         | 0.20 s | 21 ms | 38.0 |
| 12 – 20 s        | 0.25 s | 17 ms | 37.9 |

End to end through the server (normal request): a 68 s recording takes about 8 s, a 205 s recording 23 s, a 410 s recording 46 s (about 0.11 × real time; recordings longer than a minute are transcribed in pieces of about a minute).

Streaming (stream=true), time to process / length of the audio (below 1 keeps up with live speech):

| **Chunk size** | 160 ms | 200 ms | 240 ms | 320 ms | 480 ms | 1000 ms | 2000 ms |
|----------------|-------:|-------:|-------:|-------:|-------:|--------:|--------:|
| 11 – 18 s recordings | 0.88 | 0.70 | 0.62 | 0.50 | 0.37 | 0.22 | |
| 205 s recording      | | | | | 0.43 | | 0.18 |

---

### 🧠 Memory

About 3.2 GB resident after loading, about 3.6 GB after the first request, at most about 4.5 GB while transcribing a 5.5-minute recording (4.1 GB as one request, 4.5 GB streamed at 480 ms chunks).
