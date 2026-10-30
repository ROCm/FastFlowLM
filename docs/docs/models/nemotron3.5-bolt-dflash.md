---
layout: docs
title: Nemotron Bolt DFlash
nav_order: 18
parent: Models
---

## 🧩 Model Card: [NVIDIA-Nemotron-3.5-Bolt-E4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB) with the [DFlash draft](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash)

- **Type:** Any-to-Text, with speculative decoding
- **What it is:** the same model as [`nemotron3.5-bolt:e4b`](../nemotron3.5-bolt/), decoded faster: NVIDIA's DFlash draft proposes up to 7
  tokens at a time and the NPU checks all of them in one pass
- **Speed:** 1.3x to 1.9x the decode speed of `nemotron3.5-bolt:e4b`, from short prompts to 16k tokens of context
  ([benchmarks](https://fastflowlm.com/docs/benchmarks/nemotron3.5-bolt-dflash_results/))
- **Think / Conversational mode / Tool Calling / Vision / Audio:** as `nemotron3.5-bolt:e4b`
- **Packages:** two downloads. The main weights are the `nemotron3.5-bolt:e4b` package (`Nemotron-3.5-Bolt-E4B-NPU2`, about 9 GB); the
  draft is [`Nemotron-3.5-Bolt-E4B-DFlash-NPU2`](https://huggingface.co/FastFlowLM/Nemotron-3.5-Bolt-E4B-DFlash-NPU2) (0.27 GB).
  `flm pull nemotron3.5-bolt-dflash:e4b` downloads whichever is missing
- **Draft:** 5 layers, 0.36B parameters, reads features of 5 layers of the main model; Q4_1 with its feature projection in BF16
- **Max Context Length:** 64k tokens
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))
- **License:** [OpenMDW-1.1](https://openmdw.ai/license/1-1/)
- **FastFlowLM:** 1.0.12 or later

▶️ Run with FastFlowLM in PowerShell:

```shell
flm run nemotron3.5-bolt-dflash:e4b
```

Server mode:

```shell
flm serve nemotron3.5-bolt-dflash:e4b
```

🧪 **Smoke test** (one clear answer, long enough for many speculative steps):

```shell
flm run nemotron3.5-bolt-dflash:e4b
>>> List the integers from 1 to 40 separated by commas, then write the word done.
```

Expected: `1, 2, 3, ..., 40` followed by `done`, at a visibly higher speed than with `nemotron3.5-bolt:e4b`.

📝 **Note:**

- Every step drafts 7 tokens and checks them together; the model keeps the longest prefix it agrees with plus one token of its own.
  Sampling works as usual: each checked position is sampled with your settings (temperature, top_k, top_p, penalties) and compared with
  the draft, so the output follows the main model's distribution.
- The gain depends on how predictable the text is: code, lists and structured answers gain the most (up to about 3x), open-ended prose
  the least. Thinking on (`"think": true`) gains more than thinking off.
- Greedy output can differ from `nemotron3.5-bolt:e4b` at near-ties: the NPU checks 8 positions in one pass, which rounds differently
  from decoding one token at a time. 97% of checked positions pick exactly the same token as token-by-token decoding (99.5% when the two
  best tokens are not within a hair of each other).
- Speculation works on both NPU runtimes (XRT and HRX) with the same speed and output.
- `FLM_DFLASH=0` in the environment turns speculation off: the tag then decodes exactly like `nemotron3.5-bolt:e4b`.
- Speculation pauses while a tool-call or JSON grammar constrains the output, and whenever it would not be faster (for example at very
  long contexts); it resumes by itself.
- `flm remove nemotron3.5-bolt-dflash:e4b` removes only the draft package; `nemotron3.5-bolt:e4b` stays installed.
