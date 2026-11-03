---
layout: docs
title: MiniCPM
nav_order: 15
parent: Models
---

## 🧩 Model Card: [MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)

- **Type:** Image-Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes
- **Prefix cache (server mode):** Yes, prompts with pictures included (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [openbmb/MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)
- **Quantization:** Q4_K
- **Max Context Length:** 256k tokens
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run minicpm-v-4.7:1b
```

📝 **Note**

- Images are processed the same way as in the original MiniCPM-V: the model sees a downscaled view of the whole image, and large images are also split into up to 9 tiles. Because of this, the `--img-pre-resize` (`-r`) option has no effect on this model.
- Image TTFT stops growing at about 1080p: from there up, every image is split into the maximum 9 tiles, so a 4K image costs about the same as a 1080p one. Each tile is about 448 × 448 pixels, so very fine detail in large images is downscaled.
- Video and audio input are not supported.

---

## 🧩 Model Card: [MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)

- **Type:** Image-Text-to-Text (Mixture of Experts: 35B parameters, about 3B active per token)
- **Think:** Toggleable (off by default, as the model card recommends)
- **Tool Calling Support:** Yes
- **Prefix cache (server mode):** Yes, prompts with pictures included (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [openbmb/MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)
- **Quantization:** Q4_K
- **Memory:** about 22 GB
- **Max Context Length:** 256k tokens
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run minicpm-v-4.7-moe:35b-a3b
```

📝 **Note**

- Pictures work as on MiniCPM-V-4.7-1B: a downscaled view of the whole image plus up to 9 tiles, so the `--img-pre-resize` (`-r`) option has no effect on this model.
- The text side is the same engine as `qwen3.6-moe:35b-a3b` (same layer layout, same NPU kernels); only the vision side differs.
- Thinking is off unless you turn it on (`think: true` in a request, or the interactive toggle). The model card recommends instruct mode.
- Video and audio input are not supported. Speculative decoding (DSpark) is not used.

---
