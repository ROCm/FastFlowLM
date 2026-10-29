---
layout: docs
title: LFM
nav_order: 9
parent: Models
---

## 🧩 Model Card: [LiquidAI/LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** No
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [LiquidAI/LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B)
- **Quantization:** Q4_0
- **Max Context Length:** 32k tokens 
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run lfm2:1.2b
```

---

## 🧩 Model Card: [LiquidAI/LFM2-2.6B](https://huggingface.co/LiquidAI/LFM2-2.6B)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** No
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [LiquidAI/LFM2-2.6B](https://huggingface.co/LiquidAI/LFM2-2.6B)
- **Quantization:** Q4_0
- **Max Context Length:** 32k tokens 
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run lfm2:2.6b
```

---

## 🧩 Model Card: [LiquidAI/LFM2-2.6B-Transcript](https://huggingface.co/LiquidAI/LFM2-2.6B-Transcript)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** No
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [LiquidAI/LFM2-2.6B-Transcript](https://huggingface.co/LiquidAI/LFM2-2.6B-Transcript)
- **Quantization:** Q4_0
- **Max Context Length:** 32k tokens 
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run lfm2-trans:2.6b
```

⚠️ This model is intended for single-turn conversations with a specific format, described [here](https://huggingface.co/LiquidAI/LFM2-2.6B-Transcript).

---

## 🧩 Model Card: [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** No
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
- **Quantization:** Q4_0
- **Max Context Length:** 32k tokens 
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run lfm2.5-it:1.2b
```

---

## 🧩 Model Card: [LiquidAI/LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking)

- **Type:** Text-to-Text
- **Think:** Yes
- **Tool Calling Support:** No
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [LiquidAI/LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking)
- **Quantization:** Q4_0
- **Max Context Length:** 32k tokens 
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run lfm2.5-tk:1.2b
```

---

---

## 🧩 Model Card: [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B)

- **Type:** Text-to-Text
- **Think:** Yes (on by default, toggleable)
- **Tool Calling Support:** Yes
- **Base Model:** [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B)
- **Quantization:** Q4_0 (the vendor's quantization-aware-distilled build)
- **Max Context Length:** 32k tokens 
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**
- **FastFlowLM:** 1.0.11 or later

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run lfm3:1.7b
```

ℹ️ Released under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE) (commercial use is limited by company revenue: check it before deploying).
