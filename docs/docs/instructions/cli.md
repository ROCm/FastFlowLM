---
layout: docs
title: System Command and CLI Mode
nav_order: 1
parent: Instructions
---

# ⚡ CLI Mode

FLM CLI mode offers a familiar terminal-based interactive experience, fully offline and accelerated exclusively on AMD NPUs. Here are detailed descriptions of commands and setup for CLI mode usage. It includes:

- **[🔧 Pre-Run PowerShell Commands (System)](#-pre-run-powershell-commands)**
- **[💻 Commands Inside CLI Mode](#-commands-inside-cli-mode)**
- **[🗂️ Others](#️-others)**

---

## 🔧 Pre-Run PowerShell Commands (System)

### 🖥️ System Compatibility Check

Verify that your hardware meets the necessary requirements to run FastFlowLM:

```shell
flm validate
```

Output the validation results as a JSON object:

```shell
flm validate --json
```

---

### 🆘 Show Help

```shell
flm help
```

---

### 🚀 Run a Model

Run a model interactively from the terminal:

```shell
flm run llama3.2:1b
```

> `flm` is short for FastFlowLM. If the model isn't available locally, it will be downloaded automatically. This launches FastFlowLM in CLI mode.

> **Linux note:** `flm validate` checks the kernel DRM device, while `flm run` opens the NPU through XRT. If validation succeeds but `flm run` fails with `No such device with index '0'`, confirm XRT can see the NPU:
> ```shell
> xrt-smi examine
> ```
> On Arch Linux, install `xrt-plugin-amdxdna` in addition to `xrt` and the `amdxdna` driver. See the [Linux install guide](/docs/install_lin/) for the full driver and firmware checklist.

---

### ⬇️ Pull a Model (Download Only)

Download a model from HuggingFace without launching it:

```shell
flm pull llama3.2:3b
```

This code forces a re-download of the model, overwriting the current version.

```shell
flm pull llama3.2:3b --force
```

> ⚠️ Use `--force` **only if the model file is corrupted** (e.g., incomplete download). Proceed with caution.

> 📦 Some tags are built on another one: `nemotron3.5-bolt-dflash:e4b` adds a small draft package to the `nemotron3.5-bolt:e4b` weights.
> `flm pull` downloads the base package first (from the same source, `--modelscope` included), `flm list` marks such a tag
> "(requires ...)", and it is ready only when both packages are.

#### 📁 Default Model Storage Location

| Platform | Default Path |
|----------|-------------|
| Windows  | `C:\Users\<USER>\.flm\models` |
| Linux    | `~/.config/flm/models` |

#### 🔧 Changing the Model Storage Path

You can override the default location by setting the `FLM_MODEL_PATH` environment variable.

**Windows** — Update the existing system environment variable:
1. Open **Start** and search for **"Edit the system environment variables"**.
2. Click **Environment Variables…**.
3. Under **System variables**, find `FLM_MODEL_PATH`, select it, and click **Edit…**.
4. Update the value to your desired path (e.g., `D:\models\flm`).
5. Click **OK** and restart any open terminals for the change to take effect.

**Linux** — Set temporarily for the current shell session:
```shell
export FLM_MODEL_PATH="/your/custom/path"
```

To make the change permanent, add the line above to your `~/.bashrc`, then reload it:
```shell
echo 'export FLM_MODEL_PATH="/your/custom/path"' >> ~/.bashrc
source ~/.bashrc
```

---

### 📦 List Supported and Downloaded Models

Display all available models and locally downloaded models:

```shell
flm list
```

Output as JSON:

```shell
flm list --json
```


Filters flag:

```shell
# Show everything
flm list --filter all

# Only models already installed
flm list --filter installed

# Only models not yet installed
flm list --filter not-installed
```

Quiet mode:

```shell
# Default view (pretty, with icons)
flm list

# Quiet view (no emoji / minimal)
flm list --quiet

# Show everything
flm list --filter all --quiet

# Only models already installed
flm list --filter installed --quiet

# Only models not yet installed
flm list --filter not-installed --quiet
```

---

### ❌ Remove a Downloaded Model

Delete a model from local storage:

```shell
flm remove llama3.2:3b
```

For a tag built on another one (for example `nemotron3.5-bolt-dflash:e4b`), `flm remove` deletes only its own package; the base model stays.
Removing a base model that other installed tags need prints a warning naming them.

### ✅ Check a Downloaded Model

Verify the file hashes for a downloaded model:

```shell
flm check llama3.2:3b
```

---

### 🚀 Start Server Mode (Local)

Launch FastFlowLM as a local REST API server (also supports the OpenAI API):

```shell
flm serve llama3.2:1b
```

---

### 🔌 Show Server Port 

Show current FLM port (default) in PowerShell:  
  
```shell
flm port
```

---

### ⚡ NPU Power Mode

By default, **FLM runs in `performance` NPU power mode**. You can switch to other NPU power modes (`powersaver`, `balanced`, or `turbo`) using the `--pmode` flag:

For **CLI mode**:
```shell
flm run gemma3:4b --pmode balanced
```

For **Server mode**:
```shell
flm serve gemma3:4b --pmode balanced
```

---

### 📏 Set Context Length at Launch

The default context length for each model can be found [here](https://fastflowlm.com/docs/models/).   

Set the context length with `--ctx-len` (or `-c`).  

In PowerShell, run:

For **CLI mode**:
```shell
flm run llama3.2:1b --ctx-len 8192
```

For **Server mode**:
```shell
flm serve llama3.2:1b --ctx-len 8192
```

> - Internally, FLM enforces a minimum context length of 512. If you specify a smaller value, it will automatically be adjusted up to 512.  
> - If you enter a context length that is not a power of 2, FLM automatically rounds it up to the nearest power of 2. For example: input `8000` → adjusted to `8192`.

---

### 🔀 Choose an Execution Backend

**A backend names where the kernels come from.** There are three:

| id | kernels | ships on |
|---|---|---|
| `flm` | FastFlowLM's own NPU kernels | Strix / Krackan Point |
| `core` | AMD's ryzenai-corelib | the next NPU generation |
| `oga-hipep` | onnxruntime-genai with a hip-ep runtime installed on demand | the GPU, on a host that also has an NPU (see below) |

You normally never set this. The model tag already names the flow — a family ending in `-core` wants corelib's kernels, one ending in `-oga-hipep` wants the GPU through hip-ep, everything else FastFlowLM's — and `flm list` only ever offers you tags this build and this machine can actually run. `flm run` separately prints the silicon as `NPU platform: dev04`. A model family has at most one engine per backend, so there is nothing to choose between.

The flag exists for overriding what the tag says, and for the targets that will join this list later:

```shell
flm run   phi4-mini-it:4b     --backend flm
flm serve phi4-mini-it-core:4b --backend core
```

**Precedence**, highest first:

| # | Source | |
|---|---|---|
| 1 | `--backend <id>` | the flag above, or a `"backend"` field on a `/v1/chat/completions` or `/v1/completions` request |
| 2 | `FLM_BACKEND=<id>` | environment variable, for a whole shell session |
| 3 | the model's catalog entry | `core` for a tag whose family ends in `-core`, `oga-hipep` for one ending in `-oga-hipep`, `flm` otherwise |

A per-request `"backend"` overrides `--backend` for that request, and reloads the model if it differs from the one already loaded — exactly as asking for a different model does.

A model that comes from a `--model-extension` catalog always runs on the backend its entry names. `--backend`, `FLM_BACKEND` and the request field can only repeat that backend, not switch it.

Asking for a backend this build does not ship fails at load with a message listing what is actually available. There is no silent fallback to another engine or to the CPU. Note that a release is built for one NPU generation: on hardware it was not built for, `run`, `serve` and `bench` refuse up front, and `flm validate` still reports what it found.

> A build carries the engines for one generation only, so there is no flag that moves it to the other catalog entry — that is a different build of FLM.

#### oga-hipep: OGA models on the GPU

Models whose family ends in `-oga-hipep` run on the GPU through a hip-ep package. FLM reuses an existing package when it finds one, and on the standard Windows XRT build it can install the pinned runtime into the current user's profile the first time `run` or `serve` needs it.

```shell
flm pull qwen3.5-oga-hipep:9b
flm run  qwen3.5-oga-hipep:9b
```

- **Build it in.** The standard Windows release ZIP and MSI include this backend. Source builds must configure FLM with `-DFLM_ENABLE_OGA_HIPEP=ON` on Windows with the XRT runtime (`FLM_USE_HRX=OFF`). It is unsupported on HRX and Linux builds. Without it, `flm list` does not offer these models.
- **Runtime discovery and install.** `flm run qwen3.5-oga-hipep:9b` and `flm serve qwen3.5-oga-hipep:9b` first look for a usable package under `%LOCALAPPDATA%\hip-ep\<version>` and `%ProgramFiles%\hip-ep\<version>`, newest version first. A package needs onnxruntime-genai 0.19 or later, so the hip-ep 0.5.2 an earlier FLM installed (onnxruntime-genai 0.17) is passed over; it stays on disk until `flm uninstall hipep` removes it. When none qualifies and `HIPEP_PKG_ROOT` is unset, FLM downloads `amdgpu-0.5.3.zip` (about 137 MiB, 330 MiB unpacked) from the `hip-ep` v0.5.3 release, verifies its built-in SHA-256, and installs it under `%LOCALAPPDATA%\hip-ep\flm-0.5.3`. Whichever package is used, FLM registers its `hipgpu.dll` with onnxruntime-genai as the `AMDGPUExecutionProvider`, once per process. `flm pull`, `flm list`, help output, and non-OGA models do not trigger this install path.
- **Profile paths.** Non-ASCII profile paths work when Windows provides a usable short-path alias, which both the shared downloader and the provider registration need. If short names are disabled, FLM reports the limitation; use `HIPEP_PKG_ROOT` with a package extracted under an ASCII path.
- **Manual and offline use.** You can still unzip a package yourself into a folder under `%LOCALAPPDATA%\hip-ep` or `%ProgramFiles%\hip-ep`. The folder must hold `onnxruntime-genai.dll` (0.19 or later), `onnxruntime.dll` and `hipgpu.dll`, at its root or in a `bin` directory. `amdgpu-0.5.3.zip` already holds them in an `amdgpu-0.5.3` folder, so extract it to `%LOCALAPPDATA%\hip-ep` itself: Windows' Extract All suggests a destination named after the zip, which nests that folder one level too deep. The version may sit anywhere in the folder name, such as `amdgpu-0.5.3` or `hip-ep-0.5.3`; a folder without one, such as `amdgpu`, is tried after the versioned ones. Or set `HIPEP_PKG_ROOT` to the package root. When `HIPEP_PKG_ROOT` is set it is the only location used, and a package it points to that is rejected is an error, not a fallback to the default locations or to an automatic download.
- **Uninstall.** `flm uninstall hipep` removes what the automatic install added: `%LOCALAPPDATA%\hip-ep\flm-<version>` and the installer's files, including the cached zip, under `%LOCALAPPDATA%\hip-ep-install`. A package you unzipped yourself, any package under `%ProgramFiles%`, the package `HIPEP_PKG_ROOT` names, and every model stay; `flm remove <model_tag>` removes a model. While a running `flm` has the runtime loaded for an oga-hipep model, the command removes nothing and says so: stop that process first. The next `run` or `serve` of an oga-hipep model installs hip-ep again when it finds no other package.
- **Models.** `qwen3.5-oga-hipep` (`9b`, `35b-a3b`, `122b-a10b`), `qwen3.6-oga-hipep` (`27b`, `35b-a3b`), `qwen3.8-oga-hipep:27b` and `qwen3.8-flash-next-oga-hipep:125b-a6b` take images as well as text. `qwen2.5-it-oga-hipep:14b`, `qwen2.5-coder-it-oga-hipep:14b`, `gemma3-oga-hipep:4b`, `gemma4-it-oga-hipep` (`12b`, `26b-a4b`) and `gpt-oss-oga-hipep:120b` take text only: the Gemma packages ship a vision model, and `gemma4-it-oga-hipep:12b` an audio model too, but FLM does not run them, so a request with an image or audio gets HTTP 400. The packages are 4 to 124 GB each.
- **Images.** The images of one prompt share a budget of 8100 patches (one 1440x1440 image, 2025 image tokens), so a prompt with several images resizes each one smaller. The first image after the model loads takes a few seconds longer: FLM first runs the vision model once at that full size, so that every size after it gets correct image features (hip-ep 0.5.2 gave wrong ones without it).
- **The host must have an NPU.** FLM still opens the NPU for this backend, so a host without an NPU is not supported yet.
- **Conversation state.** Exact token prefixes continue the existing generator, so a request does not run its tokens twice. For Qwen3.5, Qwen3.6, Qwen3.8 and gpt-oss, FLM also keeps a checkpoint at the end of each prompt, as on the NPU: a follow-up turn whose prompt starts with the previous prompt rewinds the generator to that checkpoint and runs only the rest, even when the chat template leaves the reasoning of earlier replies out. onnxruntime-genai 0.19 cannot rewind the Gemma 4 packages, so with thinking on a Gemma 4 follow-up replays the conversation; FLM warns at the first refused rewind and does not try again. Any other history, and every new or changed image, replays the whole conversation (OGA takes images only when it starts a generator); `usage.prompt_tokens_details.cached_tokens` counts only the tokens that were not run again. NPU prefix slots are not used.

---

### 🖧 Set Server Port at Launch

Set a custom port at launch:

  ```shell
  flm serve llama3.2:1b --port 8000
  flm serve llama3.2:1b -p 8000
  ```

> ⚠️ `--port` (`-p`) only affects the **current run**; it won’t change the default port.

---

### 🛠️ Set Host at Launch

Specify a custom host address when starting the server:

```powershell
flm serve llama3.2:1b --host 127.0.0.1
```

⚠️ Note: --host applies only to the current session. It does not modify the default host configuration (default: `127.0.0.1`).

---

### 🌐 Cross-Origin Resource Sharing (CORS)

Browser pages can call the FLM server only from allowed origins. Without any option these are:

- pages served from this machine: `http(s)://localhost`, `127.0.0.1` or `[::1]`, on any port
- desktop-app and extension pages: `app://`, `tauri://`, `vscode-webview://`, `vscode-file://`, `file://`, `chrome-extension://`, `moz-extension://`, `safari-web-extension://`

A request from any other origin gets `403`. Clients that send no `Origin` header (curl, the OpenAI SDKs, a web UI's backend such as Open WebUI's) are not affected.

- Allow more origins, e.g. a web UI on another machine

```shell
flm serve --cors-origins "http://nas:3000,https://webui.example.com"
```
- Allow every origin

```shell
flm serve --cors-origins "*"
```
- Disable CORS headers: no `Access-Control-*` header is sent, so pages on another origin cannot read the replies. The origin check above still applies.

```shell
flm serve --cors 0
```

> ⚠️ **Default:** CORS headers are **enabled** (`--cors 1`) for the allowed origins.  
> 🔒 **Host check:** when the server listens on a loopback address (the default, `127.0.0.1`), it also refuses a request whose `Host` header names another machine, which blocks DNS rebinding. Local names, private-network addresses, `*.local`, `*.internal` (such as `host.docker.internal`) and this machine's name are accepted. Behind a tunnel or reverse proxy that keeps its own host name, add its origin, e.g. `--cors-origins "https://abc.example.app"`. `"*"` turns this check off too.  
> 📄 **Local files:** a page opened from a file in a browser sends `Origin: null`, which is refused; allow it with `--cors-origins "null"`.  
> 🔒 **Security tip:** keep the allowed origins short, and restrict access at your proxy if your server is exposed beyond localhost (127.0.0.1).

---

### ⏸️ Preemption

Preemption allows high-priority tasks to interrupt ongoing NPU jobs, improving responsiveness for critical workloads. To enable preemption:

For **CLI mode**:
```shell
flm run llama3.2:1b --preemption 1
```

For **Server mode**:
```shell
flm serve llama3.2:1b --preemption 1
```

> ⚠️ Note: Preemption is for **engineering testing/optimization** only. It requires a special driver + toolkit and is **not for public use**.


---

### 🧩 Change Prefill Chunk Size at Launch

The `--prefill-chunk-len` flag controls how many tokens are processed per chunk during the prefill phase of inference. The default is the model's own `max_prefill_len`: 4096 for most models, 1024 for the flash models.

For **CLI mode**:

```shell
flm run llama3.2:1b --prefill-chunk-len 8192
```

For **Server mode**:

```shell
flm serve llama3.2:1b --prefill-chunk-len 8192
```

> - Values below 512 are raised to 512, and the value is rounded up to a power of 2 (`5000` → `8192`).
> - The chunk length is not a limit on the prompt: a longer prompt is read in more chunks. The prompt has to fit in the context length (`--ctx-len`). See [Long Prompts: Prefill Chunking](https://fastflowlm.com/docs/instructions/server/request_parameters/#-long-prompts-prefill-chunking).

---

### 🗃️ Prefix Cache (Server Mode)

In **Server mode**, Qwen3.5 (`0.8b`, `2b`, `4b`, `9b`) and `qwen3.6-moe:35b-a3b` remember the system prompt, tools and earlier turns they have already read, so an agent that sends the same long prompt again only waits for what is new. It is on by default.

```shell
flm serve qwen3.5:4b --prefix-cache-mb 8192 --prefix-slots 16
```

| Option | Default | Meaning |
|---|---|---|
| `--prefix-cache-mb N` | `4096` | Most host memory (MiB) the saved prompts may use; a limit, not a reservation |
| `--prefix-slots N` | `8` (at most `32`) | How many saved prompts to keep; `0` turns the prefix cache off |
| `--prefix-min-gain N` | `128` | A new prompt is saved only if it is at least `N` tokens longer than a saved prompt that already covers it. Larger: fewer saves and evictions, but up to `N` more tokens re-read on a resume. `0`: save every prompt. See [Tuning `--prefix-min-gain`](https://fastflowlm.com/docs/instructions/server/prefix_cache/#tuning---prefix-min-gain) |

`flm run` (CLI mode) does not use it. See [Prefix Cache (Prompt Reuse)](https://fastflowlm.com/docs/instructions/server/prefix_cache/) for memory per model, how to check that it works, and its limits.

---

### 🎞️ Raw RGB Images (Server Mode)

`minicpm-v-4.7:1b` accepts decoded 8-bit RGB frames in `/v1/chat/completions` (`image_raw`), without JPEG or PNG encoding. These options cap what one request may carry:

| Option | Default | Meaning |
|---|---|---|
| `--max-raw-frames N` | `128` | Most raw frames in one request |
| `--max-raw-mb N` | `128` | Most MiB of raw pixels in one request |
| `--allow-raw-path DIR` | off | Let `image_raw` read frame files (`"path"`) from `DIR`; nothing outside it is read |

See [Raw RGB Images](https://fastflowlm.com/docs/instructions/server/raw_rgb_images/).

### 🎙️ ASR (Automatic Speech Recognition)

**Requirement:** The ASR model (e.g., `whisper-large-v3-turbo`) must run **with an LLM loaded concurrently**. Enabling `--asr <model tag>` (or `--asr 1` for the default) starts Whisper in the background **while** your chosen LLM loads.

#### CLI mode
```shell
flm run gemma3:4b --asr whisper-v3:turbo  # Load Whisper (whisper-large-v3-turbo) in the background and load the LLM (gemma3:4b) concurrently.
```

#### Server mode
```shell
flm serve gemma3:4b --asr whisper-v3:turbo  # Background-load Whisper and initialize the LLM (gemma3:4b) concurrently.
```

> **Note:** ASR alone isn’t supported—an LLM must be present for end-to-end voice→text→LLM workflows.

See the ASR guide [here](https://fastflowlm.com/docs/models/whisper/)

---

### 🧭 Decision Model (System One)

`--decision <tag>` loads a decision model (`d3:lite`) next to the chat model, or alone, and serves `POST /v1/systemone`
(text and images, up to 32,768 input tokens per question; FastFlowLM 1.0.11 or later).
Decision models run in **Server Mode only** (`flm run d3:lite` prints how to serve it).

```shell
flm serve --decision d3:lite                 # decision model only
flm serve qwen3.5:4b --decision d3:lite      # next to a chat model
```

See the d3 guide [here](https://fastflowlm.com/docs/models/d3/)

---

## 💻 Commands Inside CLI Mode

Once inside the CLI, use the following commands. System commands always start with `/` (e.g., `/help`).

---

### 🆘 Help

```text
/?
```

> Displays all available interactive system commands. Highly recommended for first-time users.

---

### 🪪 Model Info

```text
/show
```

> View model architecture, size, **max context size (Adjustable – see bottom)** and more.

---

### 🔄 Change Model

```text
/load [model_name]
```

> Unload the current model and load a new one. KV cache will be cleared.

---

### 💾 Save Conversation

```text
/save
```

> Save the current conversation history to disk.

---

### 🧹 Clear Memory

```text
/clear
```

> Clear the KV cache (model memory) for a fresh start.

---

### 📊 Show Runtime Stats

```text
/status
```

> Display runtime statistics like token count, throughput, etc.

---

### 🕰️ Show History

```text
/history
```

> Review the current session's conversation history.

---

### 🔍 Toggle Verbose Mode

```text
/verbose
```

> Enable detailed performance metrics per turn. Run again to disable.

---

### 📦 List Models

Display all available models and locally downloaded models:

```text
/list
```

---

### 👋 Quit CLI Mode

```text
/bye
```

> Exit the CLI.

---

### 🧠 Think Mode Toggle

Type `/think` to toggle Think Mode on or off interactively in the CLI.

> 💡 **Note**: This feature is only supported on certain models, such as **Qwen3**.

---

### 📂 Load a Local Text File in CLI Mode

Use any file that can be opened in Notepad (like `.txt`, `.json`, `.csv`, etc.).

Format (in CLI mode):

```shell
/input "<file_path>" prompt
```

Example:

```shell
/input "C:\Users\Public\Desktop\alice_in_wonderland.txt" Summarize it into 200 words
```

> Notes:

* Use quotes **only around the file path**
* **No quotes** around the prompt
* File must be plain text (readable in Notepad)

👉 [Download a sample prompt (around 40k tokens)](https://github.com/ROCm/FastFlowLM/blob/main/assets/alice_in_wonderland.txt)  

> ⚠️ **Caution:** a model’s supported context length is limited by available DRAM capacity. For example, with **32 GB** of DRAM, **LLaMA 3.1:8B** cannot run beyond a **32K** context length. For the full **128K** context, we recommend a larger memory system.

If DRAM is heavily used by other programs while running **FastFlowLM**, you may encounter errors due to insufficient memory, such as:

```error
[XRT] ERROR: Failed to submit the command to the hw queue (0xc01e0200):
Even after the video memory manager split the DMA buffer, the video memory manager
could not page-in all of the required allocations into video memory at the same time.
The device is unable to continue.
```

> 🤔 Interested in checking the DRAM usage?

<!-- **Method 1 – Task Manager (Quick View)**   -->
1. Press **Ctrl + Shift + Esc** (or **Ctrl + Alt + Del** and select **Task Manager**).  
2. Go to the **Performance** tab.  
3. Click **Memory** to see total, used, and available DRAM, as well as usage percentage.  

<!-- **Method 2 – Resource Monitor (Detailed View)**  
1. Press **Windows + R**.  
2. Type:  ```resmon```
3. Press **Enter**.  
4. Go to the **Memory** tab to view detailed DRAM usage and a per-process breakdown. -->

---

### 🌄 Loading Images in CLI Mode (for VLMs only, e.g. gemma3:4b)

Supports **.png** and **.jpg** formats.  

```shell
/input "<image_path>" prompt
```

Example:

```shell
/input "C:\Users\Public\Desktop\cat.jpg" describe this image
```

> Notes:

* Make sure the model you are using is a **vision model (VLM)** (e.g., gemma3:4b) 
* Put quotes **only around the file path**  
* Do **not** use quotes around the prompt  
* Image must be in **.jpg** or **.png** format  

---

### ⚙️ Set Variables

```text
/set
```

> Customize decoding parameters like `top_k`, `top_p`, `temperature`, `context length (max)`, `generate limit`, etc.

> ⚠️ **Note:** Providing invalid or extreme hyperparameter values may cause inference errors.
> `generate limit` sets an upper limit on the number of tokens that can be generated for each response. Example:

```text
/set gen-lim 128
```

---

## 📊 Benchmarking Tool

Use the FLM benchmarking tool to measure a model's performance across different context lengths.

Each benchmark tests context lengths from `1k` to `32k`, running `2` iterations at each length.

```shell
flm bench llama3.2:1b
```

If the model is not downloaded yet, `flm bench` pulls it first, like `flm run` and `flm serve` (add `--modelscope` to download from ModelScope):

```shell
flm bench llama3.2:1b --modelscope
```

Change the iteration times by `bench-iterations`:

```shell
flm bench llama3.2:1b --bench-iterations 4
```

Cap the longest context tested with `-c` / `--ctx-len`:

```shell
flm bench llama3.2:1b --ctx-len 8192
```

The value is rounded down to a power-of-two number of `k` (e.g. `-c 5000` tests `1k`, `2k` and `4k`), and model-specific limits still apply. Without it, the benchmark runs up to `32k`.

FLM prints the results in your terminal and also saves them as a CSV file in the current folder for later reference.

```text
[FLM]  === Benchmark Results ===

 Context Length |              TTFT (s) |      Prefill Speed (tok/s) |     Decoding Speed (tok/s)
----------------------------------------------------------------------------------------------------
             1k |       0.815 ±   0.012 |        1233.57 ±     17.91 |       61.39 ±      0.46
             2k |       1.144 ±   0.035 |        1728.16 ±     55.68 |       59.04 ±      0.36
             4k |       1.955 ±   0.009 |        2001.55 ±      9.22 |       54.32 ±      0.47
             8k |       3.821 ±   0.010 |        2037.62 ±      5.40 |       46.89 ±      0.19
            16k |       9.144 ±   0.011 |        1698.20 ±      2.08 |       37.20 ±      0.14
            32k |      26.346 ±   0.037 |        1177.16 ±      1.53 |       26.38 ±      0.07
----------------------------------------------------------------------------------------------------
```

---

## 🗂️ Others

### 🛠 Change Default Context Length (max)

You can find more information about available models here:  

```
C:\Program Files\flm\model_list.json
```

You can also change the `default_context_length` setting.

> ⚠️ **Note:** Be cautious! The system reserves DRAM space based on the context length you set.  
> Setting a longer default context length may cause errors on systems with smaller DRAM.
> Also, each model has its own context length limit (examples below).  
> - **qwen3-tk:4b** → up to **256k** tokens  
> - **gemma3:4b** → up to **128k** tokens
> - **gemma3:1b** → up to **32k** tokens
> - **llama3.x** → up to **128k** tokens

