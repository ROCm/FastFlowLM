---
layout: docs
title: EmbeddingGemma
nav_order: 8
parent: Models
---

## 🧩 Model Card: [embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)

- **Type:** Embedding (Sentence Similarity)
- **Think:** No
- **Tool Calling Support:** No
- **Base Model:** [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
- **Quantization:** Q4_1
- **Max Chunk Size:** 2048
- **Default Context Length:** NA

▶️ Run with FastFlowLM in PowerShell:  

> Embedding models run in Server Mode, alone or alongside an LLM.

### Server Mode 

Start with embedding model enabled:

```shell
flm serve gemma3:4b --embed embed-gemma:300m
```

As a standalone embedding server (no LLM):

```shell
flm serve embed-gemma:300m
```

`--embed` / `-e` take an embedding model tag (see `flm list`); `1` selects the default (embed-gemma:300m) and `0` turns it off. An unknown tag, or the tag of a different kind of model, stops flm with an error.

Send text to `POST /v1/embeddings` via an OpenAI client or Open WebUI. Set
`model` to `embed-gemma:300m`; the server downloads and loads it when needed.

### Request options

`input` is a string or a non-empty array of strings, with one vector returned
per string. Each vector has 768 dimensions; omit `dimensions` or set it to `768`.

`task` is optional and case-sensitive. Omission or an empty string selects `query`:

| Task | Prompt prefix |
|---|---|
| Default, `query` | `task: search result \| query: ` |
| `document` | `title: <title> \| text: ` (`none` when omitted) |
| `clustering` | `task: clustering \| query: ` |
| `classification` | `task: classification \| query: ` |
| `code_retrieval` | `task: code retrieval \| query: ` |
| `sentence_similarity` | `task: sentence similarity \| query: ` |
| `summarization` | `task: summarization \| query: ` |

The optional `title` string is available with `task: "document"`:

```json
{"model":"embed-gemma:300m","input":"Paris is in France.","task":"document","title":"France","dimensions":768}
```

These options describe the built-in `embed-gemma:300m` model. For extension
models, consult the provider's documentation.

> see more API details here → [/v1/embeddings/](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create)

**Example 1**: OpenAI Client

```python

from openai import OpenAI

client = OpenAI(
   base_url="http://127.0.0.1:52625/v1", # FastFlowLM's local API endpoint
   api_key="flm", # Dummy key (FastFlowLM doesn't require authentication)
)

resp = client.embeddings.create(
   model="embed-gemma",
   input="Hi, everyone!"
)

print(resp.data[0].embedding)

```

**Example 2**: Open WebUI  

1. Follow **Open WebUI** setup [guide](https://fastflowlm.com/docs/instructions/server/webui/).  
2. In the **bottom-left corner**, click **`User`** icon, then select **`Settings`**.  
3. In the **bottom panel**, open **`Admin Settings`**.  
4. In the **left sidebar**, navigate to **Documents**.  
5. Set **Embedding Model Engine** to **OpenAI**.  
6. Enter:  
> API Base URL: `http://127.0.0.1:52625/v1` (Open WebUI Desktop) or `http://host.docker.internal:52625/v1` (Open WebUI in Docker)  
> API KEY: `flm` (any value works)    
> Embedding Model: `embed-gemma:300m`     
7. **Save** the setting.   
8. Follow the [RAG + FastFlowLM example](https://fastflowlm.com/docs/instructions/server/webui/#️-example-local-private-database-with-rag--fastflowlm) to launch your **Local Private Database with RAG** all powered by FLM.


---