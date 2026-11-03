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
- **Host memory:** the token embeddings are read straight from `model.q4nx` through a read-only memory map, one row per token, instead of being copied into a 0.4 GiB table in RAM. This saves about 0.4 GiB of RAM (measured together with its host LLM: 1.37 GiB before, 0.43 GiB now; 0.4 GiB of the 0.9 GiB saved is this model) with the same speed and the same output. Only the rows of tokens that were actually seen stay in memory, and the OS can drop them under memory pressure; if it does, a prompt full of new tokens takes a little longer to start (under 10% in our tests, with the file evicted from the page cache). On Windows, the loaded model's `model.q4nx` cannot be replaced or deleted, so stop the server before `flm remove` or `flm pull` of this model.
- **Max Chunk Size:** 2048
- **Default Context Length:** NA

▶️ Run with FastFlowLM in PowerShell:  

> The embedding model must be used with an LLM (loaded concurrently) in Server Mode.
> The embedding model does not work in CLI Mode.

### Server Mode 

Start with embedding model enabled:

```shell
flm serve gemma3:4b --embed 1 # Load the embedding model (embed-gemma:300m) in the background, with concurrent LLM loading (gemma3:4b).
```

Send file(s) to `POST /v1/embeddings` via any OpenAI Client or Open WebUI.

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