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
- **Max Context Length:** 16k tokens
- **Default Context Length:** 512 tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:

```shell
flm run hy-mt2:1.8b
```

📖 Prompt Guide

**Prompt Format**

Hy-MT2 is a dedicated translation model, not a general-purpose chat model — it has no default system prompt. Put the translation instruction and source text in a single **user message**:

    将以下文本翻译为{TARGET_LANG}，注意只需要输出翻译后的结果，不要额外解释：

    {TEXT}

or, in English:

    Translate the following segment into {TARGET_LANG}, without additional explanation.

    {TEXT}
