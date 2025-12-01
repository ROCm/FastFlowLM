---
layout: docs
title: CLI Mode
nav_order: 1
parent: Instructions
---

# ⚡ CLI Mode

FLM CLI mode offers a terminal-based interactive experience, similar to Ollama, but fully offline and accelerated exclusively on AMD NPUs. Here are detailed descriptions of commands and setup for CLI mode usage. It includes:

- **[🔧 Pre-Run PowerShell Commands](#-pre-run-powershell-commands)**
- **[💻 Commands Inside CLI Mode](#-commands-inside-cli-mode)**
- **[🗂️ Others](#️-others)**

---

## 🔧 Pre-Run PowerShell Commands

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

---

### ⬇️ Pull a Model (Download Only)

Download a model from Hugging Face without launching it:

```shell
flm pull llama3.2:3b
```

This code forces a re-download of the model, overwriting the current version.

```shell
flm pull llama3.2:3b --force
```

> ⚠️ Use `--force` **only if the model file is corrupted** (e.g., incomplete download). Proceed with caution.

---

### 📦 List Supported and Downloaded Models

Display all available models and locally downloaded models:

```shell
flm list
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

---

### 🖧 Start Server Mode (Local)

Launch FastFlowLM as a local REST API server (also support OpenAI API):

```shell
flm serve llama3.2:1b
```

---

### 🖧 Show Server Port 

Show current FLM port (default) in PowerShell:  
  
```shell
flm port
```

---

### ⚡ NPU Power Mode NPU Power Mode

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

### 🎛️ Set Context Length at Launch

The default context length for each model can be found [here](https://docs.fastflowlm.com/models/).   

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

### 🖧 Set Server Port at Launch

Set a custom port at launch:

  ```shell
  flm serve llama3.2:1b --port 8000
  flm serve llama3.2:1b -p 8000
  ```

> ⚠️ `--port` (`-p`) only affects the **current run**; it won’t change the default port.

---

### 🌐 Cross-Origin Resource Sharing (CORS)

CORS lets browser apps hosted on a different origin call your FLM server safely.

- Enable CORS

```shell
flm serve --cors 1
```
- Disable CORS

```shell
flm serve --cors 0
```

> ⚠️ **Default:** CORS is **enabled**.  
> 🔒 **Security tip:** Disable CORS (or restrict at your proxy) if your server is exposed beyond localhost.

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

### 🎙️ ASR (Automatic Speech Recognition)

**Requirement:** The ASR model (e.g., `whisper-large-v3-turbo`) must run **with an LLM loaded concurrently**. Enabling `--asr 1` starts Whisper in the background **while** your chosen LLM loads.

#### CLI mode
```shell
flm run gemma3:4b --asr 1  # Load Whisper (whisper-large-v3-turbo) in the background and load the LLM (gemma3:4b) concurrently.
```

#### Server mode
```shell
flm serve gemma3:4b --asr 1  # Background-load Whisper and initialize the LLM (gemma3:4b) concurrently.
```

> **Note:** ASR alone isn’t supported—an LLM must be present for end-to-end voice→text→LLM workflows.

See the ASR guide [here](https://docs.fastflowlm.com/models/whisper.html)

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

👉 [Download a sample prompt (around 40k tokens)](https://github.com/FastFlowLM/FastFlowLM/blob/main/assets/alice_in_wonderland.txt)  

> ⚠️ **Caution:** a model’s supported context length is limited by available DRAM capacity. For example, with **32 GB** of DRAM, **LLaMA 3.1:8B** cannot run beyond a **32K** context length. For the full **128K** context, we recommend larger memory system.

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