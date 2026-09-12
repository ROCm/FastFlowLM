---
layout: docs
title: Phi
nav_order: 10
parent: Models
---

## 🧩 Model Card: [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** No
- **Base Model:** [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- **Quantization:** Q4_1
- **Max Context Length:** 128k tokens 
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run phi4-mini-it:4b
```

---

## 🧪 Model Card: Phi-4-mini-instruct on AIE4 (developer preview)

- **Tag:** `phi4-mini-it-aie4:4b`
- **Backend:** `corelib_aie4_gguf` — runs on AIE4 through AMD's `ryzenai_corelib.dll`
- **Source format:** GGUF, read directly. No ONNX model, no tensor manifest, and no converted or packed weight file is produced or shipped.
- **Quantization:** GGML `Q8_0` in the file, requantized to **group 64** while the weights are packed for the device, through corelib's explicit `*_create_gguf_requantized` entry points. This is a **lossy** second quantization step and it is not reversible; output will differ from the Q8_0 source.
- **Usable generation window:** 4095 tokens — the rendered prompt plus the requested output together, so the largest admissible prompt is 4094. An over-capacity request is rejected with HTTP 400 *before* any work is submitted to the device. Note this is far below the model's 128k context; see below for why.
- **Availability:** Windows only, and this is a **developer build**. The AIE4 runtime is not packaged by the MSI or Inno installer; you supply the DLL yourself.

This tag pulls from two pinned repositories, because the GGUF publisher does not ship the tokenizer files FastFlowLM's tokenizer frontend consumes:

| File | Repository | Revision |
|---|---|---|
| `Phi-4-mini-instruct.Q8_0.gguf` | [`unsloth/Phi-4-mini-instruct-GGUF`](https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF) | `78eb92a46fc37e6b524df991ed9aca9bc6aa7b80` |
| `tokenizer.json` | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| `tokenizer_config.json` | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| `config.json` | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) | `cfbefacb99257ffa30c83adab238a50856ac3083` |

All four are SHA-256 verified before the download is promoted, and the pulled directory contains exactly these four files.

### Building

The AIE4 path is compiled only when you ask for it. With the option off, the binary contains no reference to corelib at all.

From `FastFlowLM/src`, in a Visual Studio developer command prompt:

```powershell
$env:RYZENAI_CORELIB_INCLUDE_DIR = 'C:/path/to/ryzenai-corelib/install/include'
cmake --preset windows-aie4          # sets FLM_ENABLE_CORELIB_AIE4=ON, builds into src/build-aie4
cmake --build --preset windows-aie4
```

The `windows-aie4` preset reads `RYZENAI_CORELIB_INCLUDE_DIR` from the environment, so set it before configuring. The configure step also locates a Boost include directory, and hard-errors if the option is enabled on a non-Windows host. Everything else — XRT, FFmpeg, curl, FFTW — is the ordinary FastFlowLM dependency set; the AIE4 option does not relax any of it.

### Pointing FastFlowLM at the runtime

`flm.exe` never links `ryzenai_corelib.lib`; the DLL is resolved and loaded at run time, by absolute path:

1. `FLM_AIE4_CORELIB_PATH`, if set. It must be an **absolute path to a `.dll` file** — a relative path or a directory is rejected outright.
2. Otherwise `<directory containing flm.exe>\aie4\ryzenai_corelib.dll`.

The corelib ABI is still pre-1.0, so FastFlowLM requires an **exact `0.3.0`** match on major, minor and patch. The version is queried before any other entry point, so a mismatched runtime reports a version error rather than a missing symbol. The DLL's own dependency directory must be reachable on `PATH`.

```powershell
$env:FLM_AIE4_CORELIB_PATH = 'C:/path/to/ryzenai_corelib.dll'
flm pull phi4-mini-it-aie4:4b
flm run  phi4-mini-it-aie4:4b
```

### Why the context is 4096, and why the usable window is one less

Phi-4-mini itself supports 128k, and the existing `phi4-mini-it:4b` tag defaults to 32k. This backend gives you 4095. That is a real functional regression and it has two separate causes, which are worth keeping apart.

**The 4096 ceiling is a correctness boundary, not a buffer size.** 4096 is exactly Phi-4-mini's `rope.scaling.original_context_length`. LongRoPE selects its factors by *sequence length*, not per position: at or below the original length the short factors apply, above it the long ones do. This implementation derives only the short branch, so 4096 is the point past which the rope tables would silently be wrong. It is enforced rather than assumed — loading fails with `invalid Phi-4 RoPE metadata` unless the GGUF reports `rope.scaling.original_context_length` of exactly 4096. Raising this ceiling means deriving the long factors, not enlarging an array.

**The extra −1 is this frontend's own conservatism.** `kMaxDecodeWindow` is 4095, one below the attention window, so that any request the server admits is guaranteed to have room to finish rather than failing partway. It costs exactly one token and it is not imposed by corelib.

### No fallback

Backend selection is explicit: it comes from `execution_backend` in the model catalog and is never inferred from hardware, filename, or quantization level. If corelib is missing, unloadable, or the wrong version, this tag **fails to load with a diagnostic** — it will not quietly fall back to CPU or to the NPU2/Q4NX backend. A build without AIE4 support, and an AIE4 build with no DLL present, both still start and run every other model, including `phi4-mini-it:4b`.

---