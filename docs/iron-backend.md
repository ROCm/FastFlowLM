<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# IRON backend

The `iron` backend runs a complete IRON graph through IRON's persistent Python
server.
FastFlowLM handles tokenization, chat templates, sampling, and HTTP requests.
The worker retains the IRON model, compiled images, weights, and device state.

## Build

Configure FastFlowLM with `FLM_ENABLE_IRON=ON`. Set these variables when you run
the source build:

```bash
source /opt/xilinx/xrt/setup.sh
export FLM_IRON_PYTHON=/scratch/roesti/IRON/.venv/bin/python
export PYTHONPATH=/scratch/roesti/IRON:${PYTHONPATH:-}
```

`FLM_IRON_PYTHON` must name an environment with IRON installed. `PYTHONPATH` is
needed only when that environment uses an IRON source checkout.

For a user install, activate the IRON environment and let the installer record
its Python executable:

```bash
source /path/to/iron/.venv/bin/activate
./src/home_install.sh --iron
source ~/flm_exe/flm_env.sh
```

The generated environment script configures FastFlowLM, XRT, and the IRON
worker. An editable IRON installation also removes the need for `PYTHONPATH`.

## Llama 3.2 1B

Accept the license for `meta-llama/Llama-3.2-1B`, then provide a Hugging Face
token through `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` or `HF_TOKEN_PATH`. Pull and
serve the BF16 model registered for IRON:

```bash
flm pull llama3.2-iron:1b
flm serve llama3.2-iron:1b --ctx-len 2048

curl -H 'Content-Type: application/json' \
	-d '{"model":"llama3.2-iron:1b","messages":[{"role":"user","content":"Hello"}],"max_tokens":8}' \
	http://127.0.0.1:52625/v1/chat/completions
```

The context length must be positive and cannot exceed the checkpoint's
`max_position_embeddings`. XRT locks the model arena. Increase the process
memlock limit before using a context whose arena exceeds the current limit.
Preemption is unavailable.

The FastFlowLM tokenizer and chat metadata must name the same model variant as
the IRON checkpoint. In particular, use Llama 3.2 1B Instruct metadata with an
Instruct checkpoint, and base-model metadata with the base checkpoint.

## EmbeddingGemma 2

EmbeddingGemma 2 is available only in builds that enable the IRON backend. Pull
the model from its public repository and serve it as an embedding-only model:

```bash
flm pull embeddinggemma-2
flm serve embeddinggemma-2 --backend iron

curl -H 'Content-Type: application/json' \
	-d '{"model":"embeddinggemma-2","input":"Hello"}' \
	http://127.0.0.1:52625/v1/embeddings
```

The embedding endpoint passes text and its query or document task to the IRON
encoder. The worker returns the normalized 768-element float32 embedding.

## Protocol

FastFlowLM and the worker exchange length-prefixed JSON headers and raw binary
payloads through the worker's standard input and output pipes. The load request
names a model in `iron.applications.MODELS`. Llama requests contain the complete
token history and return bfloat16 logits. EmbeddingGemma 2 requests contain UTF-8
text and return float32 values. Worker errors poison the backend, so FastFlowLM
requires a model reload after a process or protocol failure.
