---
layout: docs
title: LLaMA
nav_order: 1
parent: Models
---

## 🧩 Model Card: [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** No  
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- **Quantization:** Q4_1
- **Max Context Length:** 128k tokens  
- **Default Context Length:** 128k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run llama3.2:1b
```

With an IRON-enabled build, `llama3.2-iron:1b` uses the same Instruct model in
BF16, with the tokenizer and chat template from FastFlowLM's package above.
Accept Meta's model license and provide a Hugging Face token (`HF_TOKEN` or
`hf auth login`), then run:

```shell
flm pull llama3.2-iron:1b
flm run llama3.2-iron:1b --ctx-len 2048
```

This downloads into `models/llama3.2-1b-instruct`. If you previously pulled the
base checkpoint, pull again after updating FastFlowLM; the new directory keeps
the two checkpoints separate.

---

## 🧩 Model Card: [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** No  
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **Quantization:** Q4_1
- **Max Context Length:** 128k tokens  
- **Default Context Length:** 64k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run llama3.2:3b
```

---

## 🧩 Model Card: [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** No  
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- **Quantization:** Q4_1
- **Max Context Length:** 128k tokens  
- **Default Context Length:** 16k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run llama3.1:8b
```

---