---
layout: docs
title: Deepseek
nav_order: 2
parent: Models
---

## 🧩 Model Card: [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)

- **Type:** Text-to-Text
- **Think:** Yes
- **Tool Calling Support:** No  
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- **Quantization:** Q4_1
- **Max Context Length:** 128k tokens  
- **Default Context Length:** 16k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run deepseek-r1:8b
```

---

## 🧩 Model Card: [DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)

- **Type:** Text-to-Text
- **Think:** Yes
- **Tool Calling Support:** No  
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [deepseek-ai/DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
- **Quantization:** Q4_1
- **Max Context Length:** 64k tokens  
- **Host memory:** the token embeddings are read straight from `model.q4nx` through a read-only memory map, one row per token, instead of being copied into a 1.2 GiB table in RAM. This saves about 1.2 GiB of RAM (the size of the table; measured on Qwen3-0.6B and Qwen3-4B, which run the same code) with the same speed and the same output. Only the rows of tokens that were actually seen stay in memory, and the OS can drop them under memory pressure; if it does, a prompt full of new tokens takes a little longer to start (under 10% in our tests, with the file evicted from the page cache). On Windows, the loaded model's `model.q4nx` cannot be replaced or deleted, so stop the server before `flm remove` or `flm pull` of this model.
- **Default Context Length:** 16k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run deepseek-r1-0528:8b
```