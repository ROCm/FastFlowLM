---
layout: docs
title: MedGemma
nav_order: 5
parent: Models
---

## 🧩 Model Card: [medgemma:4b](https://huggingface.co/google/medgemma-4b-it)

- **Type:** Image-Text-to-Text
- **Think:** No  
- **Base Model:** [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
- **Max Context Length:** 128k tokens 
- **Default Context Length:** 64k tokens ([change default](https://docs.fastflowlm.com/instructions/cli.html#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://docs.fastflowlm.com/instructions/cli.html#%EF%B8%8F-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run medgemma:4b
```

🔐 Why It Matters

- **Privacy First** — There is nothing more personal than your health!  
- **Powered by NPU** — Leverages AMD Ryzen™ AI NPU for fast, low-power inference.  
- **Healthcare Applications** — A concrete example of how local LLMs + NPUs enable privacy-preserving, research-driven healthcare workflows.

📺 Demo Video

- [MedGemma:4B (Multimodal) on AMD Ryzen™ AI NPU — Demo Video](https://www.youtube.com/watch?v=KWzXZEOcgK4&list=PLf87s9UUZrJoDdz639Yc6w1UTyJ4cFHZ1&index=5&ab_channel=FastFlowLM)  
- Prompts & images in the demo are from the [official paper (pp.12–13)](https://arxiv.org/abs/2507.05201)  
- [Model page](https://deepmind.google/models/gemma/medgemma/)  

⚠️ **Disclaimer** 

This tool (**MedGemma + FastFlowLM**) is **not** a diagnostic or clinical tool. 
Always consult a licensed medical professional for healthcare decisions.

---