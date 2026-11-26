---
layout: docs
title: Inkling
nav_order: 21
parent: Models
---

## 🧩 Model Card: [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)

- **Type:** Image-Text-to-Text (Vision, MoE: 26.4B parameters, 3.4B active; 12 middle layers run three times, about 5.4B of compute per token)
- **Minimum FLM version:** v1.0.14 (first release that supports this model)
- **Think:** Always (the model reasons before every answer; `reasoning_effort` sets how much, see below)
- **Tool Calling Support:** Yes
- **Base Model:** [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
- **FLM package:** [FastFlowLM/Inkling-2-Mini-NPU2](https://huggingface.co/FastFlowLM/Inkling-2-Mini-NPU2) ([ModelScope](https://modelscope.cn/models/amd/Inkling-2-Mini-NPU2))
- **Quantization:** Q4_K (importance matrix from [unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)), Q8_0 output head, BF16 vision tower
- **Max Context Length:** 1M tokens (tested up to 64k)
- **Default Context Length:** 64k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:

```shell
flm run inkling-2:mini
```

> 📝 NOTE
>
> - **Memory:** the package is 17.3 GB and the model needs about 20 GB of memory in FLM (more at long context). Use a system with **64 GB of RAM or more**: the NPU can only use about half of the system memory, so 32 GB machines cannot load it.
> - **Requires FastFlowLM v1.0.14 or later.**
> - Audio input of the base model is not supported; text and images are.

### 🧠 Thinking

Inkling-2 always thinks first: the reasoning goes to `reasoning_content` in server mode, the answer to `content`. Leave room for both in `max_tokens`.

In server mode, `reasoning_effort` sets how long it thinks: `none`, `minimal`, `low`, `medium`, `high` (default), `xhigh`, `max`, or a number from 0 to 0.99. Turning thinking off (`think: false`, `reasoning_effort: "none"`, or `/think` in the CLI) selects effort `none`, which still thinks, briefly.

```json
{"model": "inkling-2:mini", "reasoning_effort": "low", "messages": [{"role": "user", "content": "Hi"}]}
```

### 🖼️ Images

Images are not resized: every 40 × 40 pixel cell is one image token, up to 4,096 tokens (2560 × 2560 pixels) per image; larger images are scaled down to fit. A 640 × 480 photo is 192 tokens and is encoded in about 0.3 s.

Server mode, OpenAI-compatible:

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:52625/v1", api_key="flm")
img = base64.b64encode(open("photo.jpg", "rb").read()).decode()
r = client.chat.completions.create(
    model="inkling-2:mini",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Describe this image in one sentence."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}}]}],
    max_tokens=512)
print(r.choices[0].message.content)
```
