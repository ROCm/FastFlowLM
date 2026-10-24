---
layout: docs
title: Nemotron Bolt
nav_order: 17
parent: Models
---

## 🧩 Model Card: [NVIDIA-Nemotron-3.5-Bolt-E4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB)

- **Type:** Any-to-Text
- **Think:** Toggleable
- **Conversational mode:** Toggleable (short, spoken-style answers for voice and device assistants)
- **Tool Calling Support:** Yes
- **Base Model:** [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16), post-trained from [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) (same architecture as `gemma4-it:e4b`)
- **Quantization:** Q4_1 (converted from NVIDIA's Q4_K_M release)
- **Max Context Length:** 64k tokens
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**
- **License:** [OpenMDW-1.1](https://openmdw.ai/license/1-1/)
- **FastFlowLM:** 1.0.11 or later

▶️ Run with FastFlowLM in PowerShell:

```shell
flm run nemotron-bolt:e4b
```

🖼️ **Note:** In CLI mode, attach an image with:

```shell
/input "file/to/image.jpg" describe this image.
```

🗣️ **Note:** In CLI mode, attach an audio with:

```shell
/input "file/to/audio.mp3" summarize this audio.
```

📝 **Note:**

- Nemotron-3.5-Bolt-E4B is NVIDIA's post-trained Gemma-4-E4B for multi-turn tool use, instruction following and
  grounded answers in fast local voice and device assistants. Its vision and audio towers are Gemma-4-E4B-it's,
  unchanged, so images and audio work as for `gemma4-it:e4b`.
- In server mode, set `"conversational_mode": true` in a chat request to get short, spoken-style answers (the chat
  template adds a system instruction that asks for brief replies). Thinking is switched with `"think": true`, as for
  the other reasoning models.
- NVIDIA recommends temperature 1.0 and top_p 0.95 in both thinking modes; these are the defaults.
- The "4GB" in NVIDIA's repository name is the size of their GGUF. The FastFlowLM package is larger because it also
  carries the vision and audio towers and keeps the embedding tables at 8 bits.
