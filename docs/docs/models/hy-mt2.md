---
layout: docs
title: Hy-MT2
nav_order: 14
parent: Models
---

## 🧩 Model Card: [hy-mt2:1.8b](https://huggingface.co/tencent/Hy-MT2-1.8B)

- **Type:** Text-to-Text (Translation)
- **Think:** No
- **Tool Calling Support:** No
- **Base Model:** [tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B)
- **Quantization:** Q4_0
- **Max Context Length:** 1k tokens
- **Default Context Length:** 1k tokens (fixed, see note below)

▶️ Run with FastFlowLM in PowerShell:

```shell
flm run hy-mt2:1.8b
```

⚠️ **Note:** **Starting from FLM v1.0.6**, Hy-MT2 is served as a single-turn model: every request starts from a clean KV state, and the context length is fixed at **1k tokens** — context-length overrides are ignored. This matches how the model is meant to be used, one short translation request at a time.

> Earlier versions allowed longer contexts. The [benchmark results](https://fastflowlm.com/docs/benchmarks/hy-mt2_results/) were measured on v1.0.5 and therefore sweep up to 16k.

📖 Prompt Guide

**Prompt Format**

Hy-MT2 is a dedicated translation model, not a general-purpose chat model — it has no default system prompt. There are two ways to prompt it:

**Option 1: instruction + text in a single user message**

    将以下文本翻译为{TARGET_LANG}，注意只需要输出翻译后的结果，不要额外解释：

    {TEXT}

or, in English:

    Translate the following segment into {TARGET_LANG}, without additional explanation.

    {TEXT}

**Option 2: instruction as the system prompt, text as the user message**

Pin the translation instruction as the system prompt so it isn't re-prefilled every turn, then send only the source text as the user message:

```json
{"role": "system", "content": "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释。输出必须全部使用英语，不要输出源语言或原文"},
{"role": "user", "content": "{TEXT}"}
```

This is the recommended format in server mode for multi-turn or repeated translation calls (e.g. batch translating subtitle lines), since the instruction's prefill cost is paid once instead of once per request.