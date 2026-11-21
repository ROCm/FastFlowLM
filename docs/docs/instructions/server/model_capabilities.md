---
layout: docs
title: Model Capabilities & Bring-Your-Own-Key Clients
nav_order: 14
parent: Local Server (Server Mode)
---

# 🧩 Model Capabilities & Bring-Your-Own-Key Clients

A client that lets you add your own endpoint (VS Code's custom endpoint, the pi agent, Cline and others) asks, per model: does it call tools, does it take images, can it think, and how long is its context. `GET /v1/models` answers these for every model, so you can copy the values instead of guessing them.

> 📘 **Related pages:** [OpenAI API + Examples](https://fastflowlm.com/docs/instructions/server/openapi/) · [Tool Calling](https://fastflowlm.com/docs/instructions/server/tool_calling/) · [Reasoning, Tool Calls & Streaming](https://fastflowlm.com/docs/instructions/server/reasoning_and_tools/) · [Request Parameters & Long Prompts](https://fastflowlm.com/docs/instructions/server/request_parameters/)

---

## What the listing returns

```bash
curl http://127.0.0.1:52625/v1/models
```

Each entry keeps the OpenAI fields (`id`, `object`, `created`, `owned_by`) and adds:

```json
{
  "id": "qwen3.5:4b",
  "object": "model",
  "created": 1795270650,
  "owned_by": "FastFlowLM",
  "capabilities": ["completion", "tools", "vision", "thinking"],
  "context_length": 32768,
  "architecture": {
    "input_modalities": ["text", "image"],
    "output_modalities": ["text"]
  },
  "supported_parameters": ["tools", "tool_choice", "reasoning"]
}
```

| Field | Meaning |
|---|---|
| `capabilities` | What the model is for. `completion` comes first for every chat model. The other words: `tools` (the model is shown the tools a request offers), `vision` (takes images), `audio` (takes audio), `thinking` (can reason before it answers), `embedding`. |
| `context_length` | The context, in tokens, that the server uses for this model: the value of `--ctx-len` when the server was started with it, otherwise the model's default. Present for chat models. |
| `architecture.input_modalities` | `text`, plus `image` and `audio` when the model takes them. |
| `architecture.output_modalities` | `text` for a chat model, `embeddings` for an embedding model. |
| `supported_parameters` | The request parameters that the capabilities imply: `tools` and `tool_choice` for a model with `tools`, `reasoning` for a model with `thinking`. An empty list for a model with neither. |

The capability words are the ones Ollama uses; `context_length`, `architecture` and `supported_parameters` are the names OpenRouter uses for an OpenAI-style model list. All of them are read from one record per model, so they never disagree with each other.

FLM reports no output limit, because it enforces none. A request without `max_tokens` gets 4,096 tokens.

---

## Copy the values into a client

| You need | Read it from |
|---|---|
| Does the model call tools? | `tools` is in `capabilities` |
| Does the model take images? | `image` is in `architecture.input_modalities` |
| Can the model think? | `thinking` is in `capabilities` |
| Context length | `context_length` |

### VS Code: custom OpenAI-compatible endpoint

| VS Code property | Value |
|---|---|
| `toolCalling` | `true` when `capabilities` has `tools` |
| `vision` | `true` when `input_modalities` has `image` |
| `thinking` | `true` when `capabilities` has `thinking` |
| `maxInputTokens` | `context_length` minus the room you leave for the answer |
| `maxOutputTokens` | The largest answer you want, for example `4096`. Keep `maxInputTokens` + `maxOutputTokens` at or below `context_length`. |

VS Code does not show a model in its model picker for agent use unless `toolCalling` is `true`, so pick a model that lists `tools`.

### pi: `models.json`

| pi property | Value |
|---|---|
| `input` | `["text", "image"]` when `input_modalities` has `image`, otherwise `["text"]` |
| `contextWindow` | `context_length` |
| `maxTokens` | The largest answer you want, for example `4096` |

---

## Notes

- Do not use a model that does not list `tools` for an agent: FLM lists `tools` only for models whose prompt shows them the tools a request offers.
- `context_length` changes with `--ctx-len`. If you restart the server with another value, read the listing again.
- The listing describes the models this build and this machine can run. A model that needs a backend your build does not have is not listed.
