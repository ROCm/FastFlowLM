---
layout: docs
title: Qwen
nav_order: 3
parent: Models
---

## 🧩 Model Card: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

- **Type:** Text-to-Text
- **Think:** Toggleable  
- **Base Model:** [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://docs.fastflowlm.com/instructions/cli.html#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://docs.fastflowlm.com/instructions/cli.html#%EF%B8%8F-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3:0.6b
```

📝 **Note:**

- **CLI**: Type `/think` to toggle on/off interactively.  
- **Server Mode**: Set the `"think"` flag in the request payload.

---

## 🧩 Model Card: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

- **Type:** Text-to-Text
- **Think:** Toggleable  
- **Base Model:** [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://docs.fastflowlm.com/instructions/cli.html#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://docs.fastflowlm.com/instructions/cli.html#%EF%B8%8F-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3:0.6b
```

📝 **Note:**

- **CLI**: Type `/think` to toggle on/off interactively.  
- **Server Mode**: Set the `"think"` flag in the request payload.

---

## 🧩 Model Card: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

- **Type:** Text-to-Text
- **Think:** Toggleable  
- **Base Model:** [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://docs.fastflowlm.com/instructions/cli.html#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://docs.fastflowlm.com/instructions/cli.html#%EF%B8%8F-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3:4b
```

📝 **Note:**

- **CLI**: Type `/think` to toggle on/off interactively.  
- **Server Mode**: Set the `"think"` flag in the request payload.

---

## 🧩 Model Card: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

- **Type:** Text-to-Text
- **Think:** Toggleable  
- **Base Model:** [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 16k tokens ([change default](https://docs.fastflowlm.com/instructions/cli.html#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://docs.fastflowlm.com/instructions/cli.html#%EF%B8%8F-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3:8b
```

📝 **Note:**

- **CLI**: Type `/think` to toggle on/off interactively.  
- **Server Mode**: Set the `"think"` flag in the request payload.

---

## 🧩 Model Card: [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)

- **Type:** Text-to-Text
- **Think:** Yes  
- **Base Model:** [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://docs.fastflowlm.com/instructions/cli.html#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://docs.fastflowlm.com/instructions/cli.html#%EF%B8%8F-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3-tk:4b
```

---

## 🧩 Model Card: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

- **Type:** Text-to-Text
- **Think:** No  
- **Base Model:** [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://docs.fastflowlm.com/instructions/cli.html#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://docs.fastflowlm.com/instructions/cli.html#%EF%B8%8F-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3-it:4b
```

---

## 🧩 Model Card: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

- **Type:** Image-Text-to-Text
- **Think:** No  
- **Base Model:** [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://docs.fastflowlm.com/instructions/cli.html#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://docs.fastflowlm.com/instructions/cli.html#%EF%B8%8F-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3vl-it:4b
```

📝 **Note**

- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.
