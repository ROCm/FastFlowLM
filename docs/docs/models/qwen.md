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

## 🧪 Qwen3 0.6B / 1.7B / 4B / 8B on the rai backend (developer preview)

- **Tags:** `qwen3-rai:0.6b`, `qwen3-rai:1.7b`, `qwen3-rai:4b`, `qwen3-rai:8b` — their own tags, beside the NPU2 `qwen3:*` entries. `supported_platforms` is `aie_next`, and the `-rai` suffix is what asks for corelib's kernels.
- **Backend:** `rai` — one engine for all four sizes. The GGUF's own metadata selects the size and is cross-checked against it.
- **Source format:** GGUF `Q8_0`, read directly. Every weight is requantized to **group 64** while it is packed, except the 0.6B lm_head, which packs at **group 32**.
- **Usable generation window:** 4095 tokens, prompt plus output. Over-capacity requests are rejected with HTTP 400 before any work reaches the device.
- **Availability:** developer build against corelib 0.26.0 — see [Phi-4 on the rai backend](phi.md).

The same corelib path also runs `gemma4-it-rai:e2b` and `gemma4-it-rai:e4b` (text only), `qwen3.5-rai:9b` and `qwen3.6-moe-rai:35b-a3b` (text only). Qwen3 and Qwen3.5 stay at 4096; the other two go further:

- **`qwen3.6-moe-rai:35b-a3b` — up to 262144 tokens.** The default is 4096; raise it with `--ctx-len` (`-c`). Long contexts have three costs:
    - **Memory, up front.** The KV cache is allocated when the model is created, at the full capacity you ask for (rounded up to whole 4096-token pages), whether or not a conversation ever fills it: about 20 KiB per token (10 attention layers × 2 KV heads × 256 × BF16, for K and V), so 262144 tokens is about 5 GiB on top of the weights.
    - **Long prompts run at decode speed.** A prompt is one forward call, but only its first 4096 tokens from position 0 run as a single pass; every token after them, and every later turn's prompt, is stepped one row at a time, at about decode speed. corelib measures decode at about 26 tokens/s at a shallow context, slowing by about 2.4 ms per token for each further 4096-token page (about 18 tokens/s at 32k). So a 100k-token prompt takes on the order of an hour: about 64 minutes even at the shallow rate, and nearer two hours once the per-page slowdown is counted (an extrapolation; corelib's measurements stop at 32k). FastFlowLM also still sends a prompt in chunks of at most 4096 tokens (`max_prefill_len`), so a long prompt is a one-pass first chunk followed by stepped ones.
    - **Checkpoints take host memory.** A checkpoint (what lets a conversation rewind without replaying) copies the KV cache into host memory, in whole 4096-token pages up to the current position: about 80 MiB per filled page, so up to about 5 GiB at a full 262144-token context.
- **`gemma4-it-rai:e2b` / `gemma4-it-rai:e4b` — up to 16384 tokens**, the length corelib has validated paged Gemma 4 to, and never past the GGUF's own `gemma4.context_length`. The default is 16384. Up to 4096 any length works; above 4096 the context must be a multiple of 4096. The KV cache for the whole context is allocated up front, when the model loads.

Qwen3, Qwen3.5, Qwen3.6 and Gemma 4 run as corelib's own models, which pack their weights on every load: there is no packed-weight cache beside the GGUF. A `<gguf name>.rai-weights.bin` / `.json` pair left there by an earlier build is no longer read and can be deleted.

```powershell
flm pull qwen3-rai:4b
flm run  qwen3-rai:4b
```

---

## 🧩 Model Card: [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)

- **Type:** Text-to-Text
- **Think:** Yes
- **Tool Calling Support:** Yes  
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

---

## 🧩 Model Card: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

- **Type:** Image-Text-to-Text
- **Think:** Yes
- **Tool Calling Support:** Yes  
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
