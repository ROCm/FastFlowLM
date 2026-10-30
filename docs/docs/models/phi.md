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
- **Prefix cache (server mode):** Yes, saved prompts are reused (see [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/))
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

## 🧪 Model Card: Phi-4-mini-instruct on the core backend (developer preview)

- **Tag:** `phi4-mini-it-core:4b` — its own tag, beside `phi4-mini-it:4b`, which stays the NPU2/Q4NX build. They are two entries in the one catalog, `model_list.json`, describing two different packages, and nothing about one is a patch on the other. Two things decide whether an install offers this one, and **the first is the machine**: `supported_platforms` says `dev04` here and `aie2p` there, checked against the generation FastFlowLM reads from the device, so on shipping silicon this tag is not listed even by a build that has corelib compiled in. Only then does the second apply — whether corelib's kernels were linked at all, which the `-core` suffix on the family name is what asks for. A corelib build asks corelib which NPU the machine has, so on a dev04 NPU it offers this tag and not the NPU2 one, and on an aie2p NPU the reverse.
- **Backend:** `core` — the backend names the kernel provider; FastFlowLM reaches these kernels through AMD's `ryzenai_corelib`
- **Source format:** GGUF, read directly by corelib's own `phi4mini` model, the way Qwen3, Qwen3.5, Qwen3.6 and Gemma 4 run. No ONNX model, no tensor manifest, and no converted or packed weight file is produced or shipped.
- **Quantization:** whichever GGUF `flm pull` fetched. AMD's **`Q4_G128`** file, from Phi-4-mini's ONNX export, is packed exactly as it is, apart from its FP16 scales narrowing to BF16. The fallback, unsloth's **`Q8_0`**, is requantized to **group 64** on every load, which is a **lossy** second quantization step, so its output differs from the Q8_0 source.
- **Context:** up to 131072 tokens, and 131072 by default, the most corelib runs Phi-4-mini at. The usable generation window is one less than the context: the rendered prompt plus the requested output together. An over-capacity request is rejected with HTTP 400 *before* any work is submitted to the device.
- **Availability:** Windows and Linux, and this is a **developer build**. The core runtime is not packaged by the MSI, Inno or snap installer; you build against corelib yourself.

On core this tag pulls from pinned repositories, because the GGUF publishers do not ship the tokenizer files FastFlowLM's tokenizer frontend consumes. The GGUF comes from AMD's repository when `flm pull` can reach it: it is private for now, so that takes a Hugging Face token with access to it (`HF_TOKEN`, or `hf auth login`). Otherwise the pull falls back to unsloth's:

| File | Repository | Revision |
|---|---|---|
| `Phi-4-mini-instruct-Q4_G128.gguf` | [`amd/Phi-4-mini-instruct-Q4_G128-GGUF`](https://huggingface.co/amd/Phi-4-mini-instruct-Q4_G128-GGUF) | `812448440ca39fa1a18e1f0f874c18dfdc5be075` |
| `Phi-4-mini-instruct.Q8_0.gguf` (fallback) | [`unsloth/Phi-4-mini-instruct-GGUF`](https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF) | `78eb92a46fc37e6b524df991ed9aca9bc6aa7b80` |
| `tokenizer.json` | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| `tokenizer_config.json` | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| `config.json` | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) | `cfbefacb99257ffa30c83adab238a50856ac3083` |

Each file is SHA-256 verified before the download is promoted, and a pulled directory holds one GGUF and the three tokenizer and config files.

corelib packs the weights on every load: there is no packed-weight cache beside the GGUF. A `phi4-core-weights.bin` / `.json` pair left there by an earlier build is no longer read and can be deleted.

### Building

The core path is compiled only when you ask for it. With the option off, the binary contains no reference to corelib at all.

From `FastFlowLM/src`, in a Visual Studio developer command prompt:

```powershell
$env:RYZENAI_CORELIB_INCLUDE_DIR = 'C:/path/to/ryzenai-corelib/install/include'
$env:RYZENAI_CORELIB_LIBRARY = 'C:/path/to/ryzenai-corelib/install/lib/ryzenai_corelib.lib'
cmake --preset windows-core -DFLM_ENABLE_CORE=ON   # builds into src/build-core
cmake --build --preset windows-core
```

or, on Linux:

```shell
export RYZENAI_CORELIB_INCLUDE_DIR=/path/to/ryzenai-corelib/install/include
export RYZENAI_CORELIB_LIBRARY=/path/to/ryzenai-corelib/install/lib/libryzenai_corelib.so
cmake --preset linux-core -DFLM_ENABLE_CORE=ON     # builds into src/build-core
cmake --build --preset linux-core
```

Both presets read `RYZENAI_CORELIB_INCLUDE_DIR` and `RYZENAI_CORELIB_LIBRARY` from the environment, so set both before configuring. They ship with `FLM_ENABLE_CORE` off so that the preset still configures on a machine without corelib, which is why the option is passed on the command line above. The configure step additionally locates a Boost include directory on Windows only — XRT's `xrt/detail/any.h` reaches for `boost::any` when `__cplusplus` reads below 201703L, which MSVC does unless it is handed `/Zc:__cplusplus`; GCC and Clang report C++20 honestly, so nothing there needs Boost. Everything else — XRT, curl — is the ordinary FastFlowLM dependency set; the core option does not relax any of it.

The `src/test/phi4_core` suite is still Windows-only and is not configured on Linux; a Linux build is a compile-and-link path, not a tested one.

### Pointing FastFlowLM at the runtime

A core build (`-DFLM_ENABLE_CORE=ON`) **links corelib in**. corelib keeps its NPU device to itself and opens a hardware context per stream on it; FastFlowLM opens its own device for every other family, and a core model shares no buffer with those. The core backend is offered only where corelib reports a usable NPU (`ryzenai_corelib_npu_usable`): elsewhere corelib would run a session on the GPU or the host. Point the build at the library with `RYZENAI_CORELIB_INCLUDE_DIR` and `RYZENAI_CORELIB_LIBRARY`. `FLM_CORE_CORELIB_PATH` selects a shared library (`.dll` on Windows, `.so` elsewhere) only in the older dynamically loading configuration; in a statically linked core build it is ignored, and `flm` says so if it is set.

The corelib ABI is still pre-1.0, so FastFlowLM requires an **exact `0.53.0`** match on major, minor and patch. The version is queried before any other entry point, so a mismatched runtime reports a version error rather than a missing symbol. corelib's default build installs `ryzenai_corelib` plus one `ryzenai_corelib_model_<name>` library per model, each carrying that model's kernels; corelib loads them only from its own directory, so they must sit beside it (a Windows core build copies them next to `flm.exe`). Each engine enables its model's kernel bundle on the stream it opens; no kernel store or environment variable is involved. Corelib's own dependency directory must be reachable on `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux).

```powershell
flm pull phi4-mini-it-core:4b
flm run  phi4-mini-it-core:4b
```

### Context, and why the usable window is one less

corelib runs Phi-4-mini to its own 131072 tokens on paged attention, growing the KV cache a page at a time. LongRoPE selects its factors by *sequence length*, not per position: corelib ropes every position with the short factors at a capacity of one 4096-token page and with the long factors past it, as llama.cpp does, so a context set to 4096 or less takes the short ones and anything larger the long ones. A GGUF without `rope_factors_long` runs 4096 only.

**The extra −1 is the core backend's.** It decodes at most one token fewer than its context, so that any request the server admits has room to finish rather than failing partway. It costs exactly one token.
### No fallback

Backend selection follows the tag, never a filename or a quantization level. The `-core` suffix *is* the request for corelib's kernels, and the entry's `supported_platforms` decides whether this machine is offered it at all. There is nothing to fall back to and nothing to resolve between: `phi4-mini-it:4b` and `phi4-mini-it-core:4b` are separate tags, so asking for one never gets you the other. Once this tag is selected, there is no fallback: if corelib is missing, unloadable, or the wrong version, the tag **fails to load with a diagnostic** rather than quietly running on CPU or on the NPU2/Q4NX backend.

### Naming the backend yourself

The two engines are registered under the kernel provider they use, `flm` and `core`, and you can name one with `--backend`, with `FLM_BACKEND`, or with a `"backend"` field on a `/v1/chat/completions` or `/v1/completions` request. The [CLI reference](../instructions/cli.md) has the full precedence table.

This does not widen what the hardware can run. A given release is built for one NPU generation, so only that generation's backend is compiled in; asking for the other one fails immediately, naming what this build actually has, instead of failing deep inside an engine that was never going to work. If what you meant was the other package, ask for its tag — and if the tag is not listed, that is a different build of FLM, not a different flag.

---