---
layout: docs
title: Hy-MT3
nav_order: 14
parent: Models
---

## 🧩 Model Card: [hy-mt3-flash:1.8b](https://huggingface.co/tencent/Hy-MT3-1.8B)

- **Type:** Text-to-Text (Translation, 102 languages)
- **Think:** No
- **Tool Calling Support:** No
- **Base Model:** [tencent/Hy-MT3-1.8B](https://huggingface.co/tencent/Hy-MT3-1.8B)
- **Quantization:** Q4_K (importance-matrix tuned on translation text), Q8_0 output head
- **Max Context Length:** 8k tokens
- **Default Context Length:** 8k tokens

▶️ Run with FastFlowLM in PowerShell:

```shell
flm run hy-mt3-flash:1.8b
```

⚠️ **Note:** Hy-MT3 is served as a single-turn model, like [Hy-MT2](https://fastflowlm.com/docs/models/hy-mt2/): every request starts from a clean KV state. It needs FastFlowLM **v1.0.14** or later.

📖 Prompt Guide

**Prompt Format**

Hy-MT3 is a dedicated translation model, not a general-purpose chat model, and it has no default system prompt. Use the full language name (English names in English prompts, Chinese names in Chinese prompts) for `{TARGET_LANG}`.

**Default translation**

    Translate the following text into {TARGET_LANG}. Note that you should only output the translated result without any additional explanation:

    {TEXT}

or, in Chinese:

    将以下文本翻译为{TARGET_LANG}，注意只需要输出翻译后的结果，不要额外解释：

    {TEXT}

Hy-MT3 also follows constraints given in the prompt: a terminology list ("Reference the following translations: `{term}` translates to `{translation}`"), a target style, delimiters to keep, structured data (JSON, XML, ...) whose structure and placeholders must stay intact, or background information. See the [model card](https://huggingface.co/tencent/Hy-MT3-1.8B-GGUF) for the exact wording of each.

**Instruction as the system prompt**

Pin the instruction as the system prompt, then send only the source text as the user message. This is the recommended format in server mode for repeated translation calls (for example subtitle lines):

```json
{"role": "system", "content": "Translate the following text into English. Note that you should only output the translated result without any additional explanation."},
{"role": "user", "content": "{TEXT}"}
```

**Recommended sampling** (Tencent): temperature 0.7, top_p 0.6, top_k 20, repetition penalty 1.05.
