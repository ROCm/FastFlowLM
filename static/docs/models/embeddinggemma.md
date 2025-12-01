---
layout: docs
title: EmbeddingGemma
nav_order: 8
parent: Models
---

## 🧩 Model Card: [embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)

- **Type:** Embedding (Sentence Similarity)
- **Think:** No
- **Base Model:** [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
- **Max Chunk Size:** 2048
- **Default Context Length:** NA

▶️ Run with FastFlowLM in PowerShell:  

> Embedding model requires to use with an LLM (load concurrently) for Server Mode.
> Embedding model does not work under CLI Mode.

### Server Mode 

Start with embedding model enabled:

```shell
flm serve gemma3:4b --embed 1 # Load the embedding model (embed-gemma:300m) in the background, with concurrent LLM loading (gemma3:4b).
```

Send file(s) to `POST /v1/embeddings` via any OpenAI Client or Open WebUI.

> see more API details here → [/v1/embeddings/](https://platform.openai.com/docs/api-reference/embeddings)

**Example 1**: OpenAI Client

```python

from openai import OpenAI

client = OpenAI(
   base_url="http://localhost:52625/v1", # FastFlowLM's local API endpoint
   api_key="flm", # Dummy key (FastFlowLM doesn't require authentication)
)

resp = client.embeddings.create(
   model="embed-gemma",
   input="Hi, everyone!"
)

print(resp.data[0].embedding)

```

**Example 2**: Open WebUI  

1. Follow **Open WebUI** setup [guide](https://docs.fastflowlm.com/instructions/server/webui.html).  
2. In the **bottom-left corner**, click **`User`** icon, then select **`Settings`**.  
3. In the **bottom panel**, open **`Admin Settings`**.  
4. In the **left sidebar**, navigate to **Documents**.  
5. Set **Embedding Model Engine** to **OpenAI**.  
6. Enter:  
> API Base URL: `http://localhost:52625/v1` (Open WebUI Desktop) or `http://host.docker.internal:52625/v1` (Open WebUI in Docker)  
> API KEY: `flm` (any value works)    
> Embedding Model: `embed-gemma:300m`     
7. **Save** the setting.   
8. Follow the [RAG + FastFlowLM example](https://docs.fastflowlm.com/instructions/server/webui.html#️-example-local-private-database-with-rag--fastflowlm) to launch your **Local Private Database with RAG** all powered by FLM.


---