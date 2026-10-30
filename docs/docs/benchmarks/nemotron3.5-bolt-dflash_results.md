---
layout: docs
title: Nemotron Bolt DFlash
parent: Benchmarks
nav_order: 22
---

## ⚡ Performance and Efficiency Benchmarks

`nemotron3.5-bolt-dflash:e4b` is NVIDIA Nemotron-3.5-Bolt-E4B (`nemotron3.5-bolt:e4b`) decoded with NVIDIA's DFlash speculative-decoding
draft. Each step the draft proposes 7 tokens, the NPU checks all 8 positions in one pass (a dedicated 8-token version of the fused
decoding layer) and the model keeps the longest prefix it agrees with plus one token of its own.

> **Note:**
> - FastFlowLM 1.0.12 (local build of the release branch), FLM's default NPU power mode (Performance), Linux.
> - The baseline is `nemotron3.5-bolt:e4b` on the same build, same prompts, same settings.

---

### **Test System:**

AMD Ryzen™ AI MAX+ PRO 395 (Strix Halo) with 128 GB DRAM.

---

### 🚀 Decoding Speed (`flm serve`, 512 generated tokens, 3 prompts per context length)

Speedup over `nemotron3.5-bolt:e4b`; the "AL" columns are the tokens kept per step (acceptance length).

| **Prompt context** | **Bolt E4B (TPS)** | **Greedy, think off** | **Greedy, think on** | **Sampled, think off** | **Sampled, think on** |
|--------------------|------------------:|----------------------:|---------------------:|-----------------------:|----------------------:|
| ~30 tokens         | 13.3              | 1.74x (AL 2.87)       | 1.89x (AL 2.95)      | 1.59x (AL 2.59)        | 1.71x (AL 2.66)       |
| 1k                 | 12.7              | 1.54x (AL 2.49)       | 1.78x (AL 2.86)      | 1.51x (AL 2.44)        | 1.76x (AL 2.74)       |
| 4k                 | 11.8              | 1.33x (AL 2.10)       | 1.55x (AL 2.52)      | 1.27x (AL 2.01)        | 1.38x (AL 2.35)       |
| 16k                | 9.1               | 1.43x (AL 2.03)       | 1.64x (AL 2.31)      | 1.40x (AL 1.97)        | 1.61x (AL 2.27)       |

"Sampled" is the model's default sampler (temperature 1.0, top_k 64, top_p 0.95). Per prompt the speedup ranges from 1.1x to 3.1x (a
coding prompt reaches 40.6 tokens/s). The long-context prompts summarize long documents, which the draft predicts less well than chat.

### 🚀 `flm bench` (about 30 decoded tokens per context length)

| **Model**              | **HW**    | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------------|-----------|-------:|-------:|-------:|-------:|--------:|--------:|
| **Bolt E4B DFlash**    | NPU (FLM) | 17.9   | 18.9   | 18.3   | 21.9   | 11.0    | 7.3     |
| Bolt E4B               | NPU (FLM) | 13.2   | 12.8   | 12.2   | 11.1   | 9.5     | 7.4     |

`flm bench` decodes only a few tokens of one prompt per length, so it is a coarse measure of speculative decoding. At 32k the engine
turns speculation off because it would not be faster. Prefill speed and time to first token are those of `nemotron3.5-bolt:e4b`.

---

## 🎯 Accuracy

### Draft quality (acceptance length, greedy, thinking on)

| **Setup** | **Tokens kept per step** |
|-----------|-------------------------:|
| NVIDIA's report (MT-Bench / SPEED-Bench, draft length 7) | 3.88 / 3.61 |
| llama.cpp with the draft, 40 MT-Bench and SPEED-Bench prompts | 3.71 |
| FastFlowLM on the NPU, same 40 prompts | 3.71 |

### Agreement with token-by-token decoding

The NPU checks 8 positions in one pass. Teacher-forced on real text (3,840 positions at 1k, 4k and 16k tokens of context), the token it
picks at each position matches token-by-token decoding of `nemotron3.5-bolt:e4b`:

| **Context** | **Same top-1 token** | **Different by more than a near-tie** | **Mean cosine of the logits** |
|-------------|---------------------:|--------------------------------------:|------------------------------:|
| 1k          | 97.9%                | 0.7%                                  | 0.9995                        |
| 4k          | 97.6%                | 0.5%                                  | 0.9995                        |
| 16k         | 96.5%                | 0.3%                                  | 0.9995                        |

A "near-tie" is a top-1 / top-2 logit margin of 2 bf16 rounding steps or less. With greedy decoding the text can therefore differ from
`nemotron3.5-bolt:e4b` at such near-ties; both are valid greedy continuations of the model. With sampling, every token is drawn from
the model's distribution.

### Stability

8,246 tokens over 23 back-to-back requests (2,048-token generations and short requests, greedy and sampled) and more than 80,000
8-token checks in one session ran without an error or a stall.
