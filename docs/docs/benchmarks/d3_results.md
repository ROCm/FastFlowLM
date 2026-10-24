---
layout: docs
title: Decision 3.0 (d3)
parent: Benchmarks
nav_order: 20
---

## ⚡ d3-lite on the NPU

Accuracy and latency of `d3:lite` (vllm-sr/d3-lite v3.1.3, package D3-Lite-NPU2) served by FastFlowLM 1.0.11.

> **Note:**
> - AMD Ryzen AI MAX+ PRO 395 (Strix Halo, XDNA 2 NPU), 128 GB, Ubuntu 24.04, XRT, FLM's default NPU power mode (Performance).
> - Latency is per request at the engine (no HTTP), median of 20 runs after warm-up; HTTP adds about a millisecond.

### Accuracy against the upstream BF16 model

| Set | Questions | Upstream BF16 | NPU (FLM) |
|---|---:|---:|---:|
| Decision Index public suite, stratified subset (gold accuracy) | 6,824 | 61.2% | 60.5% |
| Same questions, top-1 agreement with BF16 | 7,226 | -- | 90.5% |
| Hand-built contract set, top-1 agreement with BF16 | 311 | -- | 93.2% |

Paired gold accuracy, upstream BF16 minus NPU: +0.69 pt (95% CI [+0.10, +1.28] pt).
Prompts, token ids and answer objects are byte-identical to the upstream runtime.

### Latency

| Request | Median (ms) |
|---|---:|
| 1 question, 128 tokens | 109 |
| 1 question, 330 tokens | 151 |
| 1 question, 1,024 tokens | 430 |
| 1 question, 2,048 tokens | 765 |
| 1 question, 4,096 tokens | 1587 |
| 1 question, 8,192 tokens | 3690 |
| 1 question, 16,384 tokens | 8425 |
| 1 question, 32,768 tokens | 22696 |
| 4 questions, 330 tokens each | 536 |
| 16 questions, 330 tokens each | 2067 |

First request after `flm serve` is ready: 191 ms (steady state 168 ms over HTTP).

Rows of 2,048 tokens and more: median of 15 runs (32,768: 3). From 1,536 tokens attention runs on the NPU (below
that the CPU is faster); the first such request also loads the attention kernel (about 0.1 s) and from then on the
decision model holds two NPU hardware contexts instead of one.

### Images

Accuracy on the Decision Index public vision suite (the 10 benchmarks that can be rebuilt without gated data; 60
questions each, images capped at 1.6 MP as the suite's `decide` engine sends them):

| Benchmark | Upstream BF16 | NPU (FLM) | Same answer |
|---|---:|---:|---:|
| CV-Bench | 81.7% | 85.0% | 91.7% |
| BLINK | 51.7% | 58.3% | 78.3% |
| RealWorldQA | 68.3% | 71.7% | 88.3% |
| CharXiv | 65.0% | 70.0% | 90.0% |
| InfographicVQA | 78.3% | 80.0% | 90.0% |
| Mind2Web | 61.7% | 63.3% | 83.3% |
| KIE (CORD + FUNSD) | 91.7% | 93.3% | 98.3% |
| Moderation (Hateful Memes) | 55.0% | 53.3% | 95.0% |
| R-Bench-M | 31.7% | 26.7% | 71.7% |
| MMMU-Pro vision | 16.7% | 15.0% | 58.3% |
| **All 600** | **60.2%** | **61.7%** | **84.5%** |

Paired gold accuracy, NPU minus upstream BF16: +1.50 pt (95% CI [−0.92, +3.92] pt). Image grids and input token
counts equal upstream's on every question; the processor's pixels are within 2 grey levels of upstream's.

Latency with and without an image, at the same total input length (one question, over HTTP, median of 5; 8,192
tokens: 3 runs; 16,384 and 32,000: 1):

| Input tokens | Text only (ms) | Text + one image (ms) | Image |
|---:|---:|---:|---|
| 516 | 266 | 483 | 640 × 480 (300 tokens) |
| 1,028 | 448 | 658 | 640 × 480 (300 tokens) |
| 2,052 | 830 | 1,651 | 1.6 MP (1,600 tokens) |
| 4,100 | 1,803 | 2,641 | 1.6 MP |
| 8,196 | 3,798 | 4,641 | 1.6 MP |
| 16,386 | 8,733 | 9,560 | 1.6 MP |
| 31,994 | 21,466 | 22,477 | 1.6 MP |

An image costs a fixed amount on top of its tokens (decoding, resizing and the vision tower): about 0.2 s at
640 × 480 and 0.8 s at 1.6 MP. All questions of a request share one image encoding: a 1.6 MP image with 8 questions
takes 6.4 s, not 8 × 1.5 s. The first request with images loads the vision tower (about 40 ms) and two more NPU
hardware contexts.
## ⚡ d3-nano on the NPU

Accuracy and latency of `d3:nano` (vllm-sr/d3-nano v3.1.3, package D3-Nano-NPU2), same machine and settings as above.

> **Note:**
> - Latency is per request at the engine (no HTTP), median of 3 runs in each of 2 processes; the NPU is bimodal per process (about ±15%).
> - Prompts are token ids of the given length. Rows of 7,869 tokens and more are real documentation text, median of 3 requests (32,768: 1 request).

### Accuracy against the upstream BF16 model

| Set | Questions | NPU (FLM) |
|---|---:|---:|
| Hand-built request set (up to 4,096 tokens), top-1 agreement with BF16 | 725 | 95.7% |
| Same set, final hidden state cosine to BF16 (median) | 725 | 0.998 |
| Long prompts, 845 to 31,491 tokens, top-1 agreement with BF16 | 30 | 90.0% |
| Long prompts, final hidden state cosine to BF16 (min over lengths) | 30 | 0.992 |
| Decision Index public suite, seeded round-robin sample (3,240 requests, 42 benchmarks): mean benchmark score | 3,240 | 55.5% (BF16: 56.7%) |
| Same requests, top-1 agreement with BF16 (choice answers) | 24,956 | 96.5% |
| Decision Index public vision suite, 60 seeded questions from each of 10 benchmarks: accuracy | 600 | 64.0% (BF16: 64.3%) |
| Same image questions, top-1 agreement with BF16 | 600 | 84.3% |

Prompts, token ids and answer objects follow the upstream runtime. On the Decision Index sample the mean benchmark
score is 1.25 points below the upstream BF16 model (standard error 0.58; about 65 to 80 requests per benchmark, so a
single benchmark is noisy). This is a sample, not the full suite, and a different one from the d3-lite subset.
The vision subset leaves out Winoground (gated), as for d3-lite; paired accuracy, BF16 minus d3-nano, is +0.3 points (95% CI -2.3 to +2.8). The GPTQ calibration prompts (300 synthetic questions)
contain no Decision Index data.

### Images (d3-nano)

Image requests agree with the upstream BF16 model on 56 of 60 answers (93.3%; 31 hand-built requests with 64 × 64 to
2000 × 1500 images, one to several images each; median probability difference 0.026). Latency over HTTP, one question,
median of 3: 1.3 s with a 1,024 × 768 image (876 input tokens), 2.4 s with two images, 3.0 s with a 2000 × 1500 image
(downscaled to the 1.6 MP budget). A request with 8 questions about one image takes 5.0 s (one image encoding). PNG, JPEG and WebP data URLs
are accepted; web addresses, file paths, corrupt images and videos are refused with HTTP 422.

### Latency

| Request | Median (ms) |
|---|---:|
| 1 question, 128 tokens | 210 |
| 1 question, 330 tokens | 252 |
| 1 question, 1,024 tokens | 584 |
| 1 question, 2,048 tokens | 1162 |
| 1 question, 4,096 tokens | 2470 |
| 1 question, 7,869 tokens | 5262 |
| 1 question, 15,854 tokens | 11431 |
| 1 question, 31,491 tokens | 27084 |
| 1 question, 32,768 tokens | 28429 |
| 4 questions, 330 tokens each | 796 |
| 16 questions, 330 tokens each | 2889 |

Attention runs on the NPU from 1,536 tokens, as for d3-lite (measured break-even for d3-nano: about 1,550 tokens).
At 31,491 tokens that is 3.3× faster than attention on the CPU (27 s against 91 s) with no measurable change in
accuracy. A prompt of 32,769 tokens is refused.

## ⚡ d3-mini on the NPU

Accuracy and latency of `d3:mini` (vllm-sr/d3-mini v3.1.3, package D3-Mini-NPU2), same machine and settings as above.

> **Note:**
> - Latency is per request at the engine (no HTTP), median of 3 runs in each of 2 processes; the NPU is bimodal per process (about ±15%). Rows of 16,116 tokens and more are real documentation text, one request each.

### Accuracy against the upstream BF16 model

| Set | Questions | NPU (FLM) |
|---|---:|---:|
| Hand-built request set (up to 4,096 tokens), top-1 agreement with BF16 | 725 | 97.0% |
| Same set, final hidden state cosine to BF16 (median) | 725 | 0.9994 |
| Long prompts, 835 to 31,545 tokens, top-1 agreement with BF16 | 30 | 96.7% |
| Long prompts, final hidden state cosine to BF16 (min over lengths) | 30 | 0.988 |
| Decision Index public suite, seeded round-robin sample (800 requests, 42 benchmarks): mean benchmark score | 800 | 61.3% (BF16: 62.6%) |
| Same requests, top-1 agreement with BF16 (choice answers) | 7,028 | 96.8% |

Prompts, token ids and answer objects follow the upstream runtime. On the Decision Index sample the mean benchmark
score is 1.25 points below the upstream BF16 model (standard error 0.75; about 19 requests per benchmark, so a single
benchmark is very noisy). This is a sample, not the full suite, and the same sampling as for d3-nano, not the d3-lite
subset. The GPTQ calibration prompts (300 synthetic questions) contain no Decision Index data.

### Images (d3-mini)

Image requests agree with the upstream BF16 model on 58 of 60 answers (96.7%; 31 hand-built requests with 64 × 64 to
2000 × 1500 images, one to several images each; median probability difference 0.024). PNG, JPEG and WebP data URLs
are accepted; web addresses, file paths, corrupt images and videos are refused with HTTP 422.

### Latency

| Request | Median (ms) |
|---|---:|
| 1 question, 128 tokens | 473 |
| 1 question, 330 tokens | 536 |
| 1 question, 1,024 tokens | 1315 |
| 1 question, 2,048 tokens | 2597 |
| 1 question, 4,096 tokens | 5280 |
| 1 question, 16,116 tokens | 25900 |
| 1 question, 31,545 tokens | 62900 |
| 1 question, 32,768 tokens | 66500 |
| 4 questions, 330 tokens each | 1767 |
| 16 questions, 330 tokens each | 6575 |

Attention runs on the NPU from 1,536 tokens, as for d3-lite (measured break-even for d3-mini: about 1,536 tokens; at
4,096 tokens the NPU attention makes a request 17% faster, 5.3 s against 6.3 s). A prompt of 32,769 tokens is refused.
