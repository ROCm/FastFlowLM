---
layout: docs
title: Whisper
parent: Benchmarks
nav_order: 23
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the speech-to-text performance of **whisper-v3:turbo** (whisper-large-v3-turbo) on NPU with
FastFlowLM (FLM).

> **Note:**
> - "Before" is the Whisper engine and kernels shipped up to FLM v1.0.12; "now" is the release that includes
>   FastFlowLM#941.
> - Under FLM's default NPU power mode (Performance).
> - Whisper works on 30 s windows: the encoder runs once per window (and on every chunk shorter than 30 s), then the
>   decoder writes the text token by token. Both run on the NPU.
> - Same runner, same FLM build, only the Whisper engine and kernels differ between the two columns.

---

### **Test System:**

AMD Ryzen™ AI MAX+ PRO 395 (Strix Halo); performance is comparable to other Strix Halo and Strix Point systems.

---

### 🚀 Speed

| **Metric** | **Before** | **Now** |
|---|---:|---:|
| Encoder, per 30 s window | 1889 ms | **606 ms** |
| Decoder, per token | 20.3 ms | 20.0 ms |
| 5-minute recording (308 s, 15 windows), wall time | 44.7 s | **25.3 s** |
| 6.4 s clip, wall time | 2.51 s | **1.16 s** |

Real-time factor (processing time / audio time, lower is faster) on the FLEURS evaluation set:

| **Set** | **Audio** | **Before** | **Now** |
|---|---:|---:|---:|
| English, 20 clips | 194 s | 0.259 | **0.130** |
| English, 3 files of ~80 s | 252 s | 0.140 | **0.082** |
| English, 1 file of 308 s | 308 s | 0.146 | **0.082** |
| German | 139 s | 0.203 | **0.109** |
| Spanish | 122 s | 0.230 | **0.123** |
| French | 105 s | 0.269 | **0.144** |
| Japanese | 142 s | 0.215 | **0.121** |
| Chinese | 104 s | 0.275 | **0.148** |
| Arabic | 107 s | 0.292 | **0.171** |

> **Note:** On Windows the encoder's CPU steps run on one thread; expect about 0.7 s per 30 s window there
> (RTF 0.17 on the English clips, 0.11 on the 308 s file).

---

## 🎯 Accuracy

FLEURS test recordings (84 files, 7 languages), word error rate (character error rate for Japanese and Chinese),
lower is better:

| **Set** | **Metric** | **Before** | **Now** |
|---|---|---:|---:|
| English, 20 clips | WER | 3.39 % | 3.39 % |
| English, 3 files of ~80 s | WER | 4.68 % | 4.68 % |
| English, 1 file of 308 s | WER | 12.48 % | 11.83 % |
| German | WER | 6.05 % | 6.51 % |
| Spanish | WER | 2.92 % | 2.92 % |
| French | WER | 6.61 % | 6.61 % |
| Japanese | CER | 5.27 % | 6.03 % |
| Chinese | CER | 11.59 % | 11.59 % |
| Arabic | WER | 15.51 % | 15.51 % |
| **Pooled** | | **7.50 %** | **7.53 %** |

> **Note:** The German and Japanese differences are one file each (one word, one compound). 79 of the 83 short files
> are transcribed identically.

---

## 💾 Memory

| | **Before** | **Now** |
|---|---:|---:|
| Whisper-only server (`flm serve --asr whisper-v3:turbo`), after transcribing | 1.0-1.1 GB | 2.3-2.4 GB |
| ... with `WHISPER_CACHE_DEQUANT=0` | | 1.0 GB |

The extra memory holds the encoder's dequantised weights. With `WHISPER_CACHE_DEQUANT=0` they are dequantised for
every window instead (about 0.8 s per 30 s window).
