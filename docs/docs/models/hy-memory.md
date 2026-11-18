---
layout: docs
title: Hy-Memory
nav_order: 15
parent: Models
---

## 🧩 Model Card: [hy-memory:30b-a3b](https://huggingface.co/tencent/Hy-Memory-30B-A3B)

- **Type:** Text-to-Text (long-term memory writer, MoE, ~3B active parameters)
- **Think:** No (the template's `reasoning_effort` defaults to `no_think`)
- **Tool Calling Support:** No
- **Base Model:** [tencent/Hy-Memory-30B-A3B](https://huggingface.co/tencent/Hy-Memory-30B-A3B) (Apache-2.0)
- **Quantization:** Q4_K (experts and projections), Q8_0 (lm_head)
- **Max Context Length:** 16k tokens
- **Default Context Length:** 16k tokens
- **Host memory:** about 19 GB for the weights (pinned in NPU-visible memory, not reclaimable) plus 1–2 GB of working memory, and up to about 4 GB more for the prefix cache; 32 GB or more of system RAM is recommended.

▶️ Run with FastFlowLM in PowerShell:

```shell
flm run hy-memory:30b-a3b
```

📖 Usage

Hy-Memory is the language model behind external memory frameworks (Mem0, MemOS, Hy-Memory). It receives one extract / reconcile prompt at a time and answers with a JSON object or array, so FLM serves it with **greedy decoding** by default. Every request is a whole conversation; the prefix cache reuses the part that matches an earlier request.

```shell
flm serve hy-memory:30b-a3b
```

```json
{
  "model": "hy-memory:30b-a3b",
  "messages": [
    {"role": "user", "content": "Extract facts as a JSON array: Alice moved to Paris in 2021 and works at the Louvre."}
  ]
}
```

The chat template reads `reasoning_effort` (`no_think`, `low` or `high`; default `no_think`). Images and audio are ignored: the model is text only.

⚠️ The prompt is prefilled at about 200 tokens/s, so a full 16k-token prompt takes over a minute to its first token. Keep memory-writer prompts short, or reuse a long shared system prompt through the prefix cache.
