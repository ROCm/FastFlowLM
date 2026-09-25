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

## 🧪 Model Card: Phi-4-mini-instruct on the rai backend (developer preview)

- **Tag:** `phi4-mini-it-rai:4b` — its own tag, beside `phi4-mini-it:4b`, which stays the NPU2/Q4NX build. They are two entries in the one catalog, `model_list.json`, describing two different packages, and nothing about one is a patch on the other. Two things decide whether an install offers this one, and **the first is the machine**: `supported_platforms` says `aie_next` here and `aie2p` there, checked against the generation FastFlowLM reads from the device, so on shipping silicon this tag is not listed even by a build that has corelib compiled in. Only then does the second apply — whether corelib's kernels were linked at all, which the `-rai` suffix on the family name is what asks for. `utils::get_device()` is still a stand-in and currently answers `aie_next` for every build, so a corelib developer build offers this tag and not the NPU2 one; a build without corelib on the same setting offers no models at all, and says so.
- **Backend:** `rai` — the backend names the kernel provider; FastFlowLM reaches these kernels through AMD's `ryzenai_corelib`
- **Source format:** GGUF, read directly. No ONNX model, no tensor manifest, and no converted or packed weight file is produced or shipped.
- **Quantization:** GGML `Q8_0` in the file, requantized to **group 64** while the weights are packed for the device, through corelib's explicit `*_create_gguf_requantized` entry points. This is a **lossy** second quantization step and it is not reversible; output will differ from the Q8_0 source.
- **Usable generation window:** 4095 tokens — the rendered prompt plus the requested output together, so the largest admissible prompt is 4094. An over-capacity request is rejected with HTTP 400 *before* any work is submitted to the device. Note this is far below the model's 128k context; see below for why.
- **Availability:** Windows and Linux, and this is a **developer build**. The rai runtime is not packaged by the MSI, Inno or snap installer; you build against corelib yourself.

On rai this tag pulls from two pinned repositories, because the GGUF publisher does not ship the tokenizer files FastFlowLM's tokenizer frontend consumes:

| File | Repository | Revision |
|---|---|---|
| `Phi-4-mini-instruct.Q8_0.gguf` | [`unsloth/Phi-4-mini-instruct-GGUF`](https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF) | `78eb92a46fc37e6b524df991ed9aca9bc6aa7b80` |
| `tokenizer.json` | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| `tokenizer_config.json` | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| `config.json` | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) | `cfbefacb99257ffa30c83adab238a50856ac3083` |

All four are SHA-256 verified before the download is promoted, and the pulled directory contains exactly these four files.

### Building

The rai path is compiled only when you ask for it. With the option off, the binary contains no reference to corelib at all.

From `FastFlowLM/src`, in a Visual Studio developer command prompt:

```powershell
$env:RYZENAI_CORELIB_INCLUDE_DIR = 'C:/path/to/ryzenai-corelib/install/include'
$env:RYZENAI_CORELIB_LIBRARY = 'C:/path/to/ryzenai-corelib/install/lib/ryzenai_corelib.lib'
cmake --preset windows-rai -DFLM_ENABLE_RAI=ON   # builds into src/build-rai
cmake --build --preset windows-rai
```

or, on Linux:

```shell
export RYZENAI_CORELIB_INCLUDE_DIR=/path/to/ryzenai-corelib/install/include
export RYZENAI_CORELIB_LIBRARY=/path/to/ryzenai-corelib/install/lib/libryzenai_corelib.so
cmake --preset linux-rai -DFLM_ENABLE_RAI=ON     # builds into src/build-rai
cmake --build --preset linux-rai
```

Both presets read `RYZENAI_CORELIB_INCLUDE_DIR` and `RYZENAI_CORELIB_LIBRARY` from the environment, so set both before configuring. They ship with `FLM_ENABLE_RAI` off so that the preset still configures on a machine without corelib, which is why the option is passed on the command line above. The configure step additionally locates a Boost include directory on Windows only — XRT's `xrt/detail/any.h` reaches for `boost::any` when `__cplusplus` reads below 201703L, which MSVC does unless it is handed `/Zc:__cplusplus`; GCC and Clang report C++20 honestly, so nothing there needs Boost. Everything else — XRT, FFmpeg, curl, FFTW — is the ordinary FastFlowLM dependency set; the rai option does not relax any of it.

The `src/test/phi4_rai` suite is still Windows-only and is not configured on Linux; a Linux build is a compile-and-link path, not a tested one.

### Pointing FastFlowLM at the runtime

A rai build (`-DFLM_ENABLE_RAI=ON`) **links corelib in**, because the NPU device the whole process shares comes from corelib's `ryzenai::corelib::GetDevice()` rather than from a device FastFlowLM opens itself. Point the build at the library with `RYZENAI_CORELIB_INCLUDE_DIR` and `RYZENAI_CORELIB_LIBRARY`. `FLM_RAI_CORELIB_PATH` selects a shared library (`.dll` on Windows, `.so` elsewhere) only in the older dynamically loading configuration; in a statically linked rai build it is ignored, and `flm` says so if it is set.

The corelib ABI is still pre-1.0, so FastFlowLM requires an **exact `0.3.0`** match on major, minor and patch. The version is queried before any other entry point, so a mismatched runtime reports a version error rather than a missing symbol. Corelib's own dependency directory must be reachable on `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux).

```powershell
flm pull phi4-mini-it:4b
flm run  phi4-mini-it:4b
```

### Why the context is 4096, and why the usable window is one less

Phi-4-mini itself supports 128k, and the existing `phi4-mini-it:4b` tag defaults to 32k. This backend gives you 4095. That is a real functional regression and it has two separate causes, which are worth keeping apart.

**The 4096 ceiling is a correctness boundary, not a buffer size.** 4096 is exactly Phi-4-mini's `rope.scaling.original_context_length`. LongRoPE selects its factors by *sequence length*, not per position: at or below the original length the short factors apply, above it the long ones do. This implementation derives only the short branch, so 4096 is the point past which the rope tables would silently be wrong. It is enforced rather than assumed — loading fails with `invalid Phi-4 RoPE metadata` unless the GGUF reports `rope.scaling.original_context_length` of exactly 4096. Raising this ceiling means deriving the long factors, not enlarging an array.

**The extra −1 is this frontend's own conservatism.** `kMaxDecodeWindow` is 4095, one below the attention window, so that any request the server admits is guaranteed to have room to finish rather than failing partway. It costs exactly one token and it is not imposed by corelib.

### No fallback

Backend selection follows the tag, never a filename or a quantization level. The `-rai` suffix *is* the request for corelib's kernels, and the entry's `supported_platforms` decides whether this machine is offered it at all. There is nothing to fall back to and nothing to resolve between: `phi4-mini-it:4b` and `phi4-mini-it-rai:4b` are separate tags, so asking for one never gets you the other. Once this tag is selected, there is no fallback: if corelib is missing, unloadable, or the wrong version, the tag **fails to load with a diagnostic** rather than quietly running on CPU or on the NPU2/Q4NX backend.

### Naming the backend yourself

The two engines are registered under the kernel provider they use, `flm` and `rai`, and you can name one with `--backend`, with `FLM_BACKEND`, or with a `"backend"` field on an `/api/chat` or `/api/generate` request. The [CLI reference](../instructions/cli.md) has the full precedence table.

This does not widen what the hardware can run. A given release is built for one NPU generation, so only that generation's backend is compiled in; asking for the other one fails immediately, naming what this build actually has, instead of failing deep inside an engine that was never going to work. If what you meant was the other package, ask for its tag — and if the tag is not listed, that is a different build of FLM, not a different flag.

---