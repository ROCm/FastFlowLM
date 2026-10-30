---
layout: docs
title: Flash Models
nav_order: 19
parent: Models
---

## ⚡ Flash Models

**Flash** models trade context length for a much faster time-to-first-token. They are single-turn (every request starts from a clean KV state; a system prompt is pinned and reused), have a fixed **1k-token** context, and do not support tool calling. They are built for high-throughput, short, independent requests: classification, captioning, extraction, routing, translation.

| **Model tag** | **Task** | **Model card** |
|---------------|----------|----------------|
| `gemma4e-flash:e2b` | Any-to-Text (image, audio) | [Gemma 4 E2B — Flash](https://fastflowlm.com/docs/models/gemma/#-model-card-gemma-4-e2b-it--flash) |
| `gemma4e-flash:e4b` | Any-to-Text (image, audio) | [Gemma 4 E4B — Flash](https://fastflowlm.com/docs/models/gemma/#-model-card-gemma-4-e4b-it--flash) |
| `qwen3vl-flash:4b` | Image-Text-to-Text | [Qwen3-VL-4B — Flash](https://fastflowlm.com/docs/models/qwen/#-model-card-qwen3-vl-4b-instruct--flash) |
| `hy-mt2-flash:1.8b` | Text-to-Text (Translation) | [Hy-MT2](https://fastflowlm.com/docs/models/hy-mt2/) |

- 📊 [Flash benchmarks](https://fastflowlm.com/docs/benchmarks/flash_results/)
- 🔌 [Flash models in server mode](https://fastflowlm.com/docs/instructions/server/openapi/#-example-flash-models-single-turn-pinned-system-prompt)
