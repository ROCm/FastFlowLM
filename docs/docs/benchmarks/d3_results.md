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
| Decision Index public suite, seeded round-robin sample (1,600 requests, 42 benchmarks): mean benchmark score | 1,600 | 62.8% (BF16: 63.8%) |
| Same requests, top-1 agreement with BF16 (choice answers) | 12,656 | 97.0% |
| Decision Index public vision suite, 60 seeded questions from each of 10 benchmarks: accuracy | 600 | 69.2% (BF16: 71.7%) |
| Same image questions, top-1 agreement with BF16 | 600 | 89.8% |

Prompts, token ids and answer objects follow the upstream runtime. On the Decision Index sample the mean benchmark
score is 0.99 points below the upstream BF16 model (standard error 0.57; about 37 requests per benchmark, so a single
benchmark is noisy). This is a sample, not the full suite, and the same sampling as for d3-nano, not the d3-lite
subset. The vision subset leaves out Winoground (gated), as for d3-lite and d3-nano; paired accuracy, BF16 minus d3-mini,
is +2.5 points (95% CI +0.5 to +4.5): images cost a little accuracy (largest gaps on Mind2Web, MMMU-Pro, CV-Bench and
CharXiv, about 60 questions each), while the answers agree with BF16 on 89.8% of the questions. The GPTQ calibration
prompts (300 synthetic questions) contain no Decision Index data.

### Images (d3-mini)

Image requests agree with the upstream BF16 model on 58 of 60 answers (96.7%; 31 hand-built requests with 64 × 64 to
2000 × 1500 images, one to several images each; median probability difference 0.024). Latency over HTTP, one question,
median of 3: 2.1 s with a 1,024 × 768 image (876 input tokens), 3.8 s with two images, 4.4 s with a 2000 × 1500 image
(downscaled to the 1.6 MP budget). A request with 8 questions about one image takes 10.8 s (one image encoding). PNG,
JPEG and WebP data URLs are accepted; web addresses, file paths, corrupt images and videos are refused with HTTP 422.

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

## ⚡ d3-turbo on the NPU

Accuracy and latency of `d3:turbo` (vllm-sr/d3-turbo v1.1, Qwen3.5-27B backbone, prefix-bidirectional attention, package
D3-Turbo-NPU2, Q4_K with GPTQ), same machine and settings as above.

> **Note:**
> - Latency in the table below is end to end over HTTP (`flm serve --decision d3:turbo`, the release libraries and
>   xclbins of the pull request build) on an otherwise idle machine: median and range of 5 requests after one warm-up
>   request. A request is one state block plus one block per question: the state is computed once and every question
>   runs against it (state-once).
> - Memory: 16.6 GB of weights; the server process peaks at 17.4 GB for prompts up to 4,096 tokens, 22.4 GB at 24,000 and
>   26.3 GB at 32,768 tokens (a fresh server of the pull request build, one request). In a soak run of 1,000 mixed requests (text, image
>   and long-prompt requests, several questions per request, repeated requests and malformed bodies), first one after the
>   other and then the same schedule from 4 concurrent clients next to a streaming chat model, there were 0 status and 0
>   answer mismatches between the two runs (905 answered, 95 refused with 422 as expected) and no crash or hang; the server
>   process went from 18.1 GB after the load to a transient peak of 29.4 GB and settled at 21.9 GB.

### Accuracy against the upstream BF16 model

| Set | Questions | NPU (FLM) |
|---|---:|---:|
| Hand-built request set (up to 4,096 tokens), top-1 agreement with BF16 | 736 | 95.7% |
| Same set, final hidden state cosine to BF16 (median) | 736 | 0.9990 |
| Long prompts, 6,144 to 24,576 tokens, top-1 agreement with BF16 | 5 | 100% (5 of 5) |
| Long prompts, final hidden state cosine to BF16 (min over lengths) | 5 | 0.998 |
| Context limit: 32,767 and 32,768 tokens answered (top-1 agreement), 32,769 refused | 2 | 100% (2 of 2) |
| Image requests (31 hand-built requests, 64 × 64 to 2000 × 1500 images), top-1 agreement with BF16 | 60 | 93.3% |
| Box questions on images (29 requests): median edge difference to BF16 / median IoU with BF16's box | 99 boxes | 0.006 / 0.91 |
| Decision Index public suite, seeded round-robin sample: mean benchmark score | 782 requests, 42 benchmarks | 77.18 (BF16: 76.12; +1.06 pt, standard error 0.48) |
| Same requests, top-1 agreement with BF16 (choice answers) | 7,026 | 98.9% |
| Decision Index public vision suite, 60 seeded questions from each of 10 benchmarks: accuracy | 600 | 79.0% (BF16: 81.0%; paired difference +2.0 pt, 95% CI -0.3 to +4.3) |
| Same image questions, top-1 agreement with BF16 | 600 | 88.8% |

The reference is the upstream runtime on a GPU in BF16 (goldens frozen before the NPU runs). Prompts, token ids and answer
objects follow the upstream runtime (error answers, `usage.input_tokens` and the shape of every answer are identical on all
request sets below); probabilities differ by the quantization and the bf16 kernels. The text sets were measured through
`flm serve` (212 requests, 735 answered questions: top-1 95.7%, mean largest probability difference 0.021); the cosine row
is the engine-level measurement of the same set. The image set misses the 93% agreement floor of the smaller models at the
engine level (91.7%): the vision tower's noise (the NPU's vision features differ from BF16's by about 0.31 relative L2, the
vision kernels of d3-mini by 0.58) dominates, so the judge is the Decision Index vision subset above.

### Images (d3-turbo)

Image requests agree with the upstream BF16 model on 56 of 60 answers (93.3%; 31 hand-built requests; mean largest
probability difference 0.042). Latency of an image request on an idle machine (HTTP, median of 5 after a warm-up): one question about a 1,024 × 768 image 7.6 s (range 7.55 to 7.66 s), two 1,024 × 768 images 14.6 s (14.58 to 14.63 s), one 2000 × 1500 image 15.3 s (15.23 to 15.33 s; the image is scaled to 1.6 megapixels), 8 questions about one 1,024 × 768 image 10.6 s (10.57 to 10.66 s). The first request with an image also loads the vision tower. Box questions on an image (one to eight per request, 29 hand-built requests) took 3 to 20 s per request
(median 11 s) on the idle machine, see the table below.

### Latency

| Request | Median (s) | Range (s) |
|---|---:|---:|
| 1 question, 128 tokens | 2.85 | 2.83 to 2.88 |
| 1 question, 330 tokens | 3.22 | 3.18 to 3.24 |
| 1 question, 1,024 tokens | 6.86 | 6.78 to 6.88 |
| 1 question, 4,096 tokens | 27.4 | 27.3 to 27.6 |
| 1 question, 32,768 tokens | 314 | 314 to 315 |
| 4 questions, one 330-token state | 6.09 | 6.06 to 6.14 |
| 16 questions, one 330-token state | 9.52 | 9.52 to 9.58 |
| 1 box question, one 1,024 × 768 image | 7.62 | 7.61 to 7.66 |
| 4 box questions, one image | 10.45 | 10.42 to 10.52 |

A request of n tokens has n tokens in all (state block plus question block); the 4 and 16 question requests are 597 and
1,247 tokens in all (state 330 tokens, the rest are the question blocks). A 32,768-token request is 32,767 tokens by FastFlowLM's
count. On the idle machine a pass over the weights costs about 2.8 s, then 5 to 10 ms per token (more at longer prompts); a
further question about the same state costs 0.4 to 1 s.

Earlier development measurements (plain Q4_K package, an NPU shared with other jobs, so upper bounds), time to the answer of one
question: 64 tokens 5.6 s, 330 tokens 5.5 s, 1,024 tokens 11.9 s, 2,048 tokens 23.4 s, 4,096 tokens 42 s, 8,192 tokens 84 s,
16,384 tokens 186 s, 32,768 tokens 430 s (the first timing of the 64 layers alone at 32,768 tokens: 313 s). A pass over the weights
costs about 4 to 5 s whatever the prompt (the 16.5 GB of weights stream through the NPU), plus 7 to 12 ms per token row. Before
the state was computed once, 16 questions about a 330-token state took 62 s; against one pass per question the state-once engine is 2.5 times faster at a 300-token state with 16 questions, 9.4 times at
1,024 tokens and 8.7 times at 4,096 tokens (64 questions: 4.1, 10.4 and 32 times). A first request loads the weights (about
17 s) and, with images, the vision tower.

Attention runs on the NPU from 1,536 tokens, as for the smaller models (the break-even was measured on the 27B backbone only
within the noise of the shared NPU, about ±30%).

### Video, box and track (development measurements)

Measured through `flm serve --decision d3:turbo` (HTTP, GPTQ package, a local engine build with images in question blocks and
video windows) against the frozen BF16 goldens of the same requests (upstream runtime, one row per question and one row per
video moment). The seconds per request were measured again with the pull request build on an idle machine (one pass over
each request set; requests that are refused before any work, such as a box question without a video, are left out): they
agree with the first measurements on the shared machine, because the time is spent on the NPU.

| Set | Requests | Answers / boxes compared | Result against BF16 | Seconds per request |
|---|---:|---:|---|---|
| Plain questions about a video (2 fps, up to 32 frames) | 11 | 45 | top-1 100 %, mean max difference 0.016 | 4 to 27 (median 18) |
| Box at a video moment | 11 | 22 boxes | edge difference median 0.031, IoU 0.77 | 7 to 41 (median 22) |
| Track | 13 | 120 moments | edge difference median 0.017, IoU 0.84, `visible` decision agrees on 99 % | 7 to 103 (median 14) |
| Box on images | 29 | 99 boxes | edge difference median 0.006, IoU 0.91 | 3 to 20 (median 11) |
| `assets` (state, options, instructions) | 15 | 26 answers, 7 boxes | top-1 100 %, box IoU median 0.98 | 11 to 114 (median 28) |

A box differs from the BF16 box by the vision tower's noise (a few hundredths of the image size) and a confidence moves by about the
same amount. Time per window of a track (one state of up to 32 frames, 16 moments): about 36 s; a 60 second clip at 2 updates per
second is 8 windows and 120 moments: 278 s (peak memory 19.5 GB), at 1 update per second 145 s. A box at one moment of a 60 second
clip: 29 s, a plain question about it: 28 s.

Limits, as measured: a 61 second clip is refused (`422`), a request may take at most 256 video moments and 16,384 video input
tokens (`422`; `D3_MOMENT_ROWS` changes the first), `long_context: true` is refused. Videos need the FFmpeg shared libraries of the
machine (loaded at run time); without them an encoded video is refused with `422` and frame lists still work.
