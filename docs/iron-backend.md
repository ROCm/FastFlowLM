<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# IRON backend

The `iron` backend runs a complete IRON graph in a persistent Python process.
FastFlowLM handles tokenization, chat templates, sampling, and HTTP requests.
The worker retains the IRON model, compiled images, weights, and device state.

## Build

Configure FastFlowLM with `FLM_ENABLE_IRON=ON`. Set these variables when you run
the source build:

```bash
source /opt/xilinx/xrt/setup.sh
export FLM_IRON_PYTHON=/scratch/roesti/IRON/.venv/bin/python
export PYTHONPATH=/scratch/roesti/FastFlowLM/python:${PYTHONPATH:-}
```

An installed build places `flm_iron` under `share/flm/python`. Add that directory
to `PYTHONPATH`.

## Llama 3.2 1B

The FastFlowLM package supplies `config.json`, tokenizer files, and chat
metadata. IRON loads a separate checkpoint directory that contains
`model.safetensors` and `tokenizer.model`:

```bash
export FLM_IRON_MODEL_PATH=/path/to/llama3.2-1b
flm serve llama3.2:1b --backend iron --ctx-len 2048

curl -H 'Content-Type: application/json' \
	-d '{"model":"llama3.2:1b","messages":[{"role":"user","content":"Hello"}],"max_tokens":8}' \
	http://127.0.0.1:52625/v1/chat/completions
```

The context length must be a multiple of 2048 and cannot exceed 32768.
XRT locks the model arena. Increase the process memlock limit before using a
context whose arena exceeds the current limit. Preemption is unavailable.

The FastFlowLM tokenizer and chat metadata must name the same model variant as
the IRON checkpoint. In particular, use Llama 3.2 1B Instruct metadata with an
Instruct checkpoint, and base-model metadata with the base checkpoint.

## EmbeddingGemma

IRON loads a directory that contains `model.safetensors` and `tokenizer.json`:

```bash
export FLM_IRON_EMBEDDING_PATH=/path/to/embeddinggemma-2
flm serve llama3.2:1b --embed 1 --backend iron
```

The embedding endpoint passes text and its query or document task to the IRON
encoder. The worker returns the normalized 768-element float32 embedding.

## Protocol

FastFlowLM and the worker exchange length-prefixed JSON headers and raw binary
payloads over a Unix socket pair. Llama requests contain the complete token
history and return bfloat16 logits. Embedding requests contain UTF-8 text and
return float32 values. Worker errors poison the backend, so FastFlowLM requires
a model reload after a process or protocol failure.
