---
layout: docs
title: Gemma
nav_order: 4
parent: Models
---

## 🧩 Model Card: [gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)

- **Type:** Text-to-Text
- **Think:** No  
- **Base Model:** [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://docs.fastflowlm.com/instructions/cli.html#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://docs.fastflowlm.com/instructions/cli.html#%EF%B8%8F-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run gemma3:1b
```

---

## 🧩 Model Card: [gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)

- **Type:** Image-Text-to-Text
- **Think:** No  
- **Base Model:** [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- **Max Context Length:** 128k tokens  
- **Default Context Length:** 64k tokens ([change default](https://docs.fastflowlm.com/instructions/cli.html#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://docs.fastflowlm.com/instructions/cli.html#%EF%B8%8F-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run gemma3:4b
```

📝 **Note:** In CLI mode, attach an image with:

```shell
/input "file/to/image.jpg" describe this image.
```

---

## 🧩 Model Card: [gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)

- **Type:** Image-Text-to-Text
- **Think:** No  
- **Base Model:** [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://docs.fastflowlm.com/instructions/cli.html#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://docs.fastflowlm.com/instructions/cli.html#%EF%B8%8F-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run gemma3:270m
```

⚠️ **Warning:** 
> – `gemma3:270m` is **Experimental** in FLM  
> – Limited accuracy; may produce errors  
> – Can loop on long outputs (quirk from Unsloth weights, also seen in LM Studio)  
> – Experimenting with different quantization + hyperparameters  

---