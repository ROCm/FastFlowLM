---
layout: docs
title: Qwen
nav_order: 3
parent: Models
---

## 🧩 Model Card: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

- **Type:** Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** No  
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Quantization:** Q4_1
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3:0.6b
```

📝 **Note:**

- **CLI**: Type `/think` to toggle on/off interactively.  
- **Server Mode**: Set the `"think"` flag in the request payload.

---

## 🧩 Model Card: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

- **Type:** Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** No  
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- **Quantization:** Q4_1
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3:1.7b
```

📝 **Note:**

- **CLI**: Type `/think` to toggle on/off interactively.  
- **Server Mode**: Set the `"think"` flag in the request payload.

---

## 🧩 Model Card: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

- **Type:** Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- **Quantization:** Q4_1
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3:4b
```

📝 **Note:**

- **CLI**: Type `/think` to toggle on/off interactively.  
- **Server Mode**: Set the `"think"` flag in the request payload.

---

## 🧩 Model Card: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

- **Type:** Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- **Quantization:** Q4_1
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 16k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3:8b
```

📝 **Note:**

- **CLI**: Type `/think` to toggle on/off interactively.  
- **Server Mode**: Set the `"think"` flag in the request payload.

---

## 🧪 Qwen3 0.6B / 1.7B / 4B / 8B on the core backend (developer preview)

- **Tags:** `qwen3-core:0.6b`, `qwen3-core:1.7b`, `qwen3-core:4b`, `qwen3-core:8b` — their own tags, beside the NPU2 `qwen3:*` entries. `supported_platforms` is `dev04`, and the `-core` suffix is what asks for corelib's kernels.
- **Backend:** `core` — one engine for all four sizes. The GGUF's own metadata selects the size and is cross-checked against it.
- **Source format:** GGUF, read directly by corelib. With a Hugging Face token that can read AMD's private repositories (`HF_TOKEN`, or `hf auth login`), `flm pull` fetches AMD's 4-bit file, which packs exactly as it is; without one it falls back to the public GGUF. A public `Q8_0` is requantized to 4-bit groups on every load, which is lossy.
- **Context:** each tag's limit is the most corelib runs that model at; the default is lower where corelib or our own runs found answers going wrong past it. Over-capacity requests are rejected with HTTP 400 before any work reaches the device.
- **Availability:** developer build against corelib 0.53.0 — see [Phi-4 on the core backend](phi.md).

The same corelib path runs every `-core` tag:

| Tag | Default file | Public fallback | Context limit | Default context |
|---|---|---|---|---|
| `qwen3-core:0.6b` | unsloth `Q8_0`, refit to group 64 (lm_head 32) | — | 32768 | 32768 |
| `qwen3-core:1.7b` / `4b` / `8b` | AMD `Q4_G128` | unsloth `Q8_0` | 32768 | 32768 |
| `qwen3.5-core:4b` / `9b` | AMD `Q4_G128` | unsloth `Q8_0` | 262144 | 32768 |
| `qwen3.6-moe-core:35b-a3b` | AMD `Q4_G32` | unsloth `Q8_0` | 262144 | 32768 |
| `qwen3.8-core:27b` | AMD `Q4_G128`, on the NPU | unsloth `Q4_0`, on the GPU | 262144 | 32768 |
| `gemma4-it-core:e2b` / `e4b` | AMD `Q4_G32` | unsloth `Q8_0` | 131072 | 65536 |
| `phi4-mini-it-core:4b` | AMD `Q4_G128` | unsloth `Q8_0` | 131072 | 131072 |
| `llama3.2-core:1b` / `3b`, `llama3.1-core:8b` | `Q8_0`, refit to group 64 | — | 131072 | 131072 |

Every model's attention is paged: corelib keeps the KV cache in 4096-token pages and grows it a page at a time, so a large context costs memory only as a conversation fills it. Raise the context with `--ctx-len` (`-c`), up to the limit.

FastFlowLM sends a long prompt in chunks of `max_prefill_len` (4096) tokens that end on multiples of 4096 counted from the start of the conversation, not from the start of the turn, so every chunk after a turn's first starts on a page boundary, where corelib runs a whole page as one pass. Llama, Qwen3, Phi-4-mini and Gemma 4 run a later turn together with the earlier tokens of the page it starts in. Qwen3.5, Qwen3.6 and Qwen3.8 carry recurrent state and continue from where they are; going back to the last checkpoint restores a snapshot, which copies the filled KV pages into host memory, and going back anywhere else replays the conversation.

Known limits, each corelib's (its `docs/known-problems.md` names them):

- **Qwen3.6 past 32768.** Paged attention can return garbage: a passkey asked for after about 48k tokens came back as `!` repeated (on an AIE4 test machine). A context past 32768 still loads, and says so.
- **Qwen3.8 on long prompts.** A long prompt can come back as `!` repeated, run to run and from corelib's own runner too: on an AIE4 test machine a passkey was answered at 6k and 14k tokens and garbled at 10k and 28k. Prefill passes shorter than a page past position 0 run their attention on the CPU (`PAGED_PREFILL_SHORT_CHUNK`).
- **Qwen3.5 and Qwen3.8 decode** drift from the reference the more pages it reads (`PAGED_DECODE_PAGE_DRIFT`).
- **Gemma 4 past position 65535** (`PAGED_ATTENTION_POSITION_WRAP`).
- **Llama on a very long, repetitive prompt.** A passkey in about 96k tokens of one repeated sentence came back as `!` repeated, from corelib's own runner too; Phi-4-mini answered the same prompt.
- **A 4096-token prefill pass can fail to complete** (`PAGED_PREFILL_PREEMPTION_TIMEOUT`), more often while another process uses the NPU. corelib fails only that call, so FastFlowLM resets the session and runs the conversation again from the start, once; a second failure needs an unload and reload.
Every core model runs as corelib's own model, which packs its weights on every load: there is no packed-weight cache beside the GGUF. A `<gguf name>.core-weights.bin` / `.json` pair left there by an earlier build is no longer read and can be deleted.

```powershell
flm pull qwen3-core:4b
flm run  qwen3-core:4b
```

---

## 🧩 Model Card: [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)

- **Type:** Text-to-Text
- **Think:** Yes
- **Tool Calling Support:** Yes  
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
- **Quantization:** Q4_1
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3-tk:4b
```

---

## 🧩 Model Card: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** Yes  
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **Quantization:** Q4_1
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3-it:4b
```

---

## 🧩 Model Card: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

- **Type:** Image-Text-to-Text
- **Think:** No
- **Tool Calling Support:** Yes  
- **Prefix cache (server mode):** Yes, prompts with pictures included (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- **Quantization:** Q4_1
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3vl-it:4b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen3vl-it:3b -r 1
```

```shell
flm serve qwen3vl-it:3b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px
- 4: height = 1440 px 
- 5: height = 2160 px 
- 6: height = 2880 px 
- 7: height = 3240 px 
- 8: height = 4320 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

📝 **Note**

- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.

---

## 🧩 Model Card: [Qwen3-VL-4B-Instruct — Flash](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

- **Type:** Image-Text-to-Text
- **Think:** No
- **Tool Calling Support:** No  
- **Base Model:** [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- **Quantization:** Q4_1
- **Max Context Length:** 1k tokens  
- **Default Context Length:** 1k tokens (fixed, see note below)

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3vl-flash:4b
```

⚡ **Note — what "Flash" changes:**

- Same checkpoint as `qwen3vl-it:4b`, served by a different engine. Prefill runs on a single fused NPU overlay instead of swapping between the matmul and attention overlays every layer, which cuts time-to-first-token on short prompts.
- **Prefill speed: ~350 tokens/s** at a 128-token prompt — the short-prompt case Flash is tuned for.
- **Decode speed is unchanged.** Flash only changes how prefill runs, so generation tokens/s matches `qwen3vl-it:4b` — see the [Qwen3 benchmarks](https://fastflowlm.com/docs/benchmarks/qwen3_results/).
- **Single-turn.** Every request starts from a clean KV state — earlier turns are not carried over. A system prompt is the one exception: it is pinned on first use and reused on later requests, so it is not re-prefilled on every call.
- **Context length is fixed at 1k tokens.** Context-length overrides are ignored for this model.
- **Tool calling is not supported.** Any tools passed with the request are dropped.
- **Images are always downscaled to a 256 px longer side** to fit the short context. The `--img-pre-resize` / `-r` option is ignored. Images already smaller than 256 px keep their original resolution.
- **Example usage:** [Flash models in server mode](https://fastflowlm.com/docs/instructions/server/openapi/#-example-flash-models-single-turn-pinned-system-prompt)

---

## 🧩 Model Card: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** No  
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- **Quantization:** Q4_1
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen2.5-it:3b
```

---

## 🧩 Model Card: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

- **Type:** Image-Text-to-Text
- **Think:** No
- **Tool Calling Support:** No  
- **Prefix cache (server mode):** Yes, prompts with pictures included (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Base Model:** [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- **Quantization:** Q4_1
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen2.5vl-it:3b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen2.5vl-it:3b -r 1
```

```shell
flm serve qwen2.5vl-it:3b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px
- 4: height = 1440 px 
- 5: height = 2160 px 
- 6: height = 2880 px 
- 7: height = 3240 px 
- 8: height = 4320 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

📝 **Note**

- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.

---

## 🧩 Model Card: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)

- **Type:** Image-Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** No  
- **Base Model:** [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- **Quantization:** Q4_K
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3.5:0.8b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen3.5:0.8b -r 1
```

```shell
flm serve qwen3.5:0.8b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px
- 4: height = 1440 px 
- 5: height = 2160 px 
- 6: height = 2880 px 
- 7: height = 3240 px 
- 8: height = 4320 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

📝 **Note**

- Optimal sampling parameters for generation vary depending on the task. Check the [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B#using-qwen35-via-the-chat-completions-api) for details.
- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.
- In server mode this model reuses the prompts it has already read (system prompt, tools, earlier turns), so agents such as Hermes wait for new text only. See [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/).

---

## 🧩 Model Card: [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)

- **Type:** Image-Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Base Model:** [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- **Quantization:** Q4_K
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3.5:2b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen3.5:2b -r 1
```

```shell
flm serve qwen3.5:2b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px
- 4: height = 1440 px 
- 5: height = 2160 px 
- 6: height = 2880 px 
- 7: height = 3240 px 
- 8: height = 4320 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

📝 **Note**

- Optimal sampling parameters for generation vary depending on the task. Check the [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B#using-qwen35-via-the-chat-completions-api) for details.
- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.
- In server mode this model reuses the prompts it has already read (system prompt, tools, earlier turns), so agents such as Hermes wait for new text only. See [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/).

---

## 🧩 Model Card: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

- **Type:** Image-Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Base Model:** [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- **Quantization:** Q4_K
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3.5:4b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen3.5:4b -r 1
```

```shell
flm serve qwen3.5:4b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px
- 4: height = 1440 px 
- 5: height = 2160 px 
- 6: height = 2880 px 
- 7: height = 3240 px 
- 8: height = 4320 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

📝 **Note**

- Optimal sampling parameters for generation vary depending on the task. Check the [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B#using-qwen35-via-the-chat-completions-api) for details.
- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.
- In server mode this model reuses the prompts it has already read (system prompt, tools, earlier turns), so agents such as Hermes wait for new text only. See [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/).

---

## 🧩 Model Card: [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)

- **Type:** Image-Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Base Model:** [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Quantization:** Q4_K
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3.5:9b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen3.5:9b -r 1
```

```shell
flm serve qwen3.5:9b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px 
- 4: height = 1440 px 
- 5: height = 2160 px 
- 6: height = 2880 px 
- 7: height = 3240 px 
- 8: height = 4320 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

📝 **Note**

- Optimal sampling parameters for generation vary depending on the task. Check the [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B#using-qwen35-via-the-chat-completions-api) for details.
- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.
- In server mode this model reuses the prompts it has already read (system prompt, tools, earlier turns), so agents such as Hermes wait for new text only. See [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/).

---

## 🧩 Model Card: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

- **Type:** Image-Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Base Model:** [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- **Quantization:** Q4_K
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3.6-moe:35b-a3b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen3.6-moe:35b-a3b -r 1
```

```shell
flm run qwen3.6-moe:35b-a3b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px 
- 4: height = 1440 px 
- 5: height = 2160 px 
- 6: height = 2880 px 
- 7: height = 3240 px 
- 8: height = 4320 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

📝 **Note**

- Optimal sampling parameters for generation vary depending on the task. Check the [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B#using-qwen36-via-the-chat-completions-api) for details.
- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.
- In server mode this model reuses the prompts it has already read (system prompt, tools, earlier turns), so agents such as Hermes wait for new text only. See [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/).

---

## 🧩 Model Card: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

- **Type:** Image-Text-to-Text
- **Think:** Yes
- **Tool Calling Support:** Yes  
- **Prefix cache (server mode):** Partly: the shared system prompt of text conversations is saved; a conversation that keeps going uses the older single-checkpoint reuse (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
- **Speculative Decoding:** Yes — built-in MTP draft head (see note below)
- **Base Model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Quantization:** Q4_K
- **Max Context Length:** 8k tokens  
- **Default Context Length:** 4k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**


▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3.8-mtp:27b
```

▶️ Serve with FastFlowLM in PowerShell:  

```shell
flm serve qwen3.8-mtp:27b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen3.8-mtp:27b -r 1
```

```shell
flm serve qwen3.8-mtp:27b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px 
- 4: height = 1440 px 
- 5: height = 2160 px 
- 6: height = 2880 px 
- 7: height = 3240 px 
- 8: height = 4320 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

⚡ **Note — MTP speculative decoding:**

Qwen3.8 ships with a **multi-token prediction (MTP) head** — a small draft model trained alongside the main network. FastFlowLM runs it on the NPU:

1. The MTP head **drafts** several candidate tokens in one shot.
2. The full model **verifies** them in a single pass.
3. Accepted drafts are kept; the first rejected one is replaced by the base model's own choice, and drafting restarts from there.

What that means in practice:

- **Faster decoding, same output.** Verification is an exact compare against what the base model would have emitted, so every token you receive is a token the base model would have produced on its own. Speculation changes throughput, not quality.
- **Gains are prompt-dependent.** Predictable, low-entropy stretches — code, structured output, long reasoning chains — see the highest draft acceptance. Highly creative or surprising text accepts fewer drafts and converges toward ordinary decode speed.
- **Nothing to configure.** Speculation is driven by the engine; there is no flag to set and no API change.

📝 **Note:**

- Optimal sampling parameters for generation vary depending on the task. Check the [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B#api-usage) for details.
- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.
