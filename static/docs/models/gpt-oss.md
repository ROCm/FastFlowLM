---
layout: docs
title: gpt-oss
nav_order: 6
parent: Models
---

## 🧩 Model Card: [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)

- **Type:** Text-to-Text
- **Think:** Low / Medium / High (reasoning effort) 
- **Base Model:** [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- **Max Context Length:** 128k tokens 
- **Default Context Length:** 8192 tokens ([change default](https://docs.fastflowlm.com/instructions/cli.html#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://docs.fastflowlm.com/instructions/cli.html#%EF%B8%8F-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run gpt-oss:20b
```

Default resoning effort for both CLI and Server Modes is Medium

Set reasoning effort (CLI):

```shell
# CLI
flm run gpt-oss:20b
/set r-eff high
```

> 📝 NOTE

> - **Memory Requirements**  
   ⚠️ **Note**: Running `gpt-oss:20b` may need a system with **> 32 GB RAM**. The model itself uses ~15.1 GB of memory in FLM, and there is an internal cap (~15.6 GB) from on NPU memory allocation enforced by AMD/Microsoft, which makes only about half of the total system RAM available to the NPU. **On 32 GB machines, it sometimes works sometimes not**, so we recommend more RAM for a smooth experience.


## 🧩 Model Card: [gpt-oss-safeguard-20b](https://huggingface.co/openai/gpt-oss-safeguard-20b)

- **Type:** Text-to-Text
- **Think:** Low / Medium / High (reasoning effort) 
- **Base Model:** [openai/gpt-oss-safeguard-20b](https://huggingface.co/openai/gpt-oss-safeguard-20b)
- **Max Context Length:** 128k tokens 
- **Default Context Length:** 8192 tokens ([change default](https://docs.fastflowlm.com/instructions/cli.html#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://docs.fastflowlm.com/instructions/cli.html#%EF%B8%8F-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run gpt-oss-safeguard:20b
```

Default resoning effort for both CLI and Server Modes is Medium

Set reasoning effort (CLI):

```shell
# CLI
flm run gpt-oss:20b
/set r-eff high
```

> 📝 NOTE

> - **Memory Requirements**  
   ⚠️ **Note**: Running `gpt-oss-safeguard:20b` may need a system with **> 32 GB RAM**. The model itself uses ~15.1 GB of memory in FLM, and there is an internal cap (~15.6 GB) from on NPU memory allocation enforced by AMD/Microsoft, which makes only about half of the total system RAM available to the NPU. **On 32 GB machines, it sometimes works sometimes not**, so we recommend more RAM for a smooth experience.

---