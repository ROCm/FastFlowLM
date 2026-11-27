---
layout: docs
title: Local LLM Server Basics
nav_order: 0
parent: Local Server (Server Mode)
---

# 🧠 Understanding Local LLM Servers

This page explains the key concepts behind **Local LLM Servers**, including FastFlowLM (FLM) Server and others.  

## Keeping two chat models loaded

With `flm serve --max-loaded-models 2`, explicitly name the NPU or GPU chat
model in each request. The server can retain one model per device. It still
runs one request at a time. `GET /api/ps` shows what is currently loaded;
`GET /v1/models` is the catalog, not the resident set.

On Windows the reference pair is `qwen3.5:4b` (NPU) and `qwen3.5-oga-hipep:9b`
(GPU), using their default cache and prefill settings and at most the package's
default context length. An uncalibrated package, a larger context, explicit
cache/prefill overrides, an unknown memory reading or insufficient shared host
memory makes the server switch one model at a time instead. A new model on the
same device replaces its predecessor. Eviction releases only the engine and its
in-memory conversation state: a later request cold-loads an installed, compatible
package without downloading it. There is no idle timeout or SSD snapshot.

---

The Windows check uses the smaller of available physical memory and available
commit. On a 64 GB Ryzen AI Max+ 395, the standalone default-context loads
reduced available memory by about 5.6 GiB (NPU) and 9.3 GiB (GPU). The 10 and
16 GiB incoming estimates include room for future conversation slots; the check
also leaves 4 GiB for the resident's slots and 2 GiB for the host. These are
estimates, not allocator guarantees. Without a supported host-memory reading
(including on Linux), the server logs the fallback and switches one model at a
time. If that cold load fails after memory was freed, the evicted model remains
installed, not resident. An unrelated failure during an admitted two-model load
does not cause the server to discard the peer or retry.

## 🔌 What is a Local Server?

The word “server” can be confusing. It can mean:

- **Server hardware**: a physical machine in a data center.
- **Server software**: a program that waits for requests (from another program) and responds to them.

A **local server** is simply **server software** that runs **on your own device** (like a laptop, desktop, or smartphone). It does **not** run in the cloud or on remote machines.

In short:

> **Local server = a background program on your device that handles requests.**

---

## 🧠 What is a Local LLM Server?

A **Local LLM Server** runs a **large language model (LLM)** entirely on your device.

It loads the model into memory and exposes it to apps through an API (usually the OpenAI-style API).

### 🔧 Real Examples of Local LLM Servers:

- **Ollama**
- **llama-cpp-server**

### ✅ Why Use a Local LLM Server?

Instead of directly adding the model to your app via C++ or Python, it’s often better to use a local server. Why?

| Benefit | Why It Matters |
|--------|----------------|
| **Easy integration** | No need to worry about device-specific code (CPU, GPU, NPU). Just send simple API calls. |
| **Saves memory** | The server loads the model **once** and shares it across apps. No need for every app to load its own copy. |
| **Cleaner architecture** | Keeps model logic (streaming, tool use, error handling) separate from your app logic. |
| **Cloud-to-local transition** | You can prototype your app using OpenAI cloud models, then later switch to a local model — without changing your code much. |

> In short: **Local LLM servers let your apps talk to big models running directly on your machine — cleanly and efficiently.**

---

## 🌐 What is the OpenAI API Standard?

Every LLM server — whether local or cloud — needs a way to receive prompts and return completions. That’s where **APIs** come in.

### ✅ The OpenAI API is the most common standard.

Why?

1. **Widely supported** by many local LLM servers.
2. **Used by many apps** (so it’s easy to plug in).
3. **Works for both local and cloud models**.

> Even though OpenAI runs their own cloud-based LLMs, their **API design is public** and **free to adopt**.

That means local servers like Ollama and FastFlowLM — and your own custom servers — can all **pretend to be OpenAI** to your app.

### 🔁 Why does this matter?

It makes switching between cloud and local effortless:

- You can build your app using OpenAI’s cloud models.
- Later, switch to a local LLM (for privacy, cost, or speed).
- Your app won’t need to change — it keeps using the same API.

---