---
layout: docs
title: whisper
nav_order: 7
parent: Models
---

## 🧩 Model Card: [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)

- **Type:** Speech-to-Text (ASR: Automatic Speech Recognition)
- **Think:** No
- **Tool Calling Support:** No
- **Base Model:** [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- **Max Context Length:** NA
- **Default Context Length:** NA

### ⚡ Performance and memory

Whisper processes audio in 30 s windows: the encoder runs once per window, then the decoder writes the text
token by token. Measured on Ryzen AI MAX+ 395 (Strix Halo):

| | before | now |
|---|---|---|
| encoder, per 30 s window | ~1.9 s | **~0.6 s** (Windows: ~0.7 s) |
| 5-minute English recording, wall time | ~44 s | **~25 s** |
| short clip (6 s), wall time | ~2.5 s | **~1.2 s** |

Accuracy is unchanged (FLEURS, 7 languages, 84 files: pooled error 7.50 % before, 7.53 % now; the per-language
differences are single words).

The encoder keeps its dequantised weights in memory: a Whisper-only server uses about **2.3 GB** instead of
~1.0 GB. To trade that memory for speed (~0.8 s per 30 s window instead of ~0.6 s), set
`WHISPER_CACHE_DEQUANT=0` before starting FLM:

```shell
# Linux
WHISPER_CACHE_DEQUANT=0 flm serve --asr whisper-v3:turbo
```
```powershell
# Windows (PowerShell)
$env:WHISPER_CACHE_DEQUANT = "0"; flm serve --asr whisper-v3:turbo
```

`WHISPER_PROFILE=1` prints one line per encoder window with the time spent in each stage (stderr).

Full speed, accuracy and memory tables: [Whisper benchmarks](/docs/benchmarks/whisper_results/).

▶️ Run with FastFlowLM in PowerShell:  

> The ASR model must be used with an LLM (loaded concurrently) in CLI Mode.
> The ASR model can be used as an independent ASR model in Server Mode (flm v0.9.21 and after).

### CLI Mode   

Start with ASR enabled: 

Load the ASR model (whisper-v3:turbo) in the background, with concurrent LLM loading (gemma3:4b).
```shell
flm run gemma3:4b --asr whisper-v3:turbo
```
or
```shell
flm run gemma3:4b -a whisper-v3:turbo
```

`--asr 1` / `-a 1` is the same thing: `1` selects the default ASR model (whisper-v3:turbo).

Then, type (replace `filename.mp3` with your audio file path):
```shell
/input "path\to\audio_sample.mp3" summarize it
```

### Server Mode 

Start with ASR enabled: 

- Load the ASR model (whisper-v3:turbo) in the background, with concurrent LLM loading (gemma3:4b).
```shell
flm serve gemma3:4b --asr whisper-v3:turbo
```
or
```shell
flm serve gemma3:4b -a whisper-v3:turbo
```

- Load the ASR model (whisper-v3:turbo) as a standalone ASR model.
```shell
flm serve --asr whisper-v3:turbo
```
or
```shell
flm serve -a whisper-v3:turbo
```

`--asr` / `-a` take an ASR model tag (see `flm list`); `1` selects the default (whisper-v3:turbo) and `0` turns it off. An unknown tag, or the tag of a different kind of model, stops flm with an error.

Send audio to `POST /v1/audio/transcriptions` via any OpenAI Client or Open WebUI.

> see more API details here → [/v1/audio/](https://platform.openai.com/docs/api-reference/audio)

**Example 1**: OpenAI Client

```python
# Import the official OpenAI Python SDK (FastFlowLM mirrors the OpenAI API schema)
from openai import OpenAI

# Initialize the client to point at your local FastFlowLM server
# - base_url: FastFlowLM's local OpenAI-compatible REST endpoint
# - api_key: Dummy token; FastFlowLM typically doesn't enforce auth, but the client requires a string
client = OpenAI(
    base_url="http://127.0.0.1:52625/v1",  # FastFlowLM local API endpoint
    api_key="flm",                         # Placeholder key
)

# Open the audio file in binary mode and create a transcription request
# - model: name of the speech-to-text model exposed by FLM (e.g., "whisper-v3")
# - file: file-like object pointing to your audio
with open("audio.mp3", "rb") as f:
    resp = client.audio.transcriptions.create(
        model="whisper-v3",
        file=f,
    )

# Print the transcribed text returned by the server
print(resp.text)
```

**Example 2**: Open WebUI

- Follow Open WebUI setup [guide](https://fastflowlm.com/docs/instructions/server/webui/).
- In the bottom-left corner, click User icon, then select Settings.
- In the bottom panel, open Admin Settings.
- In the left sidebar, navigate to Audio.
- Set Speech-to-Text Engine to OpenAI.
- Enter:
> API Base URL: `http://127.0.0.1:52625/v1` (Open WebUI Desktop) or `http://host.docker.internal:52625/v1` (Open WebUI in Docker)   
> API KEY: flm (any value works)    
> STT Model: whisper-large-v3-turbo (type in the model name; can be different)    
- Save the setting.
- You're ready to upload audio files! (Choose an LLM to load and use concurrently)

---