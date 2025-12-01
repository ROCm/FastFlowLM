---
layout: docs
title: Overview
nav_order: 0
has_children: false
---

# ⚡ FastFlowLM (FLM)

**FLM** is the only **NPU-first runtime** built for AMD Ryzen™ AI.  

Run **LLMs — now with Vision support — in minutes**: **no GPU required**, **over 10× more power-efficient**, and with **context lengths up to 256k tokens**.  

Think **Ollama — but laser-optimized for NPUs**.  

From *idle silicon* to *instant power* — **FastFlowLM makes Ryzen™ AI shine.**

---

## 🧪 Test Drive (Remote Demo)

✨ Don’t have a Ryzen™ AI PC? Instantly try FastFlowLM on a live AMD Ryzen™ AI 5 340 NPU with 96 GB memory ([spec](https://www.amazon.com/4X4-BOX-AI340-Display-Support-Copilot/dp/B0F2GFLF67/ref=sr_1_5?crid=1X16RDUCQ2497&dib=eyJ2IjoiMSJ9.C5GS4xMl_kkJ7Yr6dNFi6g.Dfj_l9Dk1yuIBjppqmKSqNAAPQc1F4Mu3zJ9-MDlszw&dib_tag=se&keywords=4x4+box+ai340&qid=1752010554&sprefix=www.amazon.com%2F4X4-BOX-AI340%2Caps%2C176&sr=8-5)) — no setup needed.  

🚀 **Launch Now**: [https://open-webui.testdrive-fastflowlm.com/](https://open-webui.testdrive-fastflowlm.com/)  
🔐 **Login**: `guest@flm.npu`  
🔑 **Password**: `0000`

> **Note:**
> - The context length per chat is limited to **4,096 tokens** for test drive.
> - Only **a small number of models** are offered for test drive (less confusing).
> - Alternatively, **sign up** with your own credentials instead of using the shared guest account.
> - Real-time demo powered by **FastFlowLM + Open WebUI** — no downloads, no installs.
> - **Occasional downtime**: Windows updates, power or connectivity issues. Try again later if happens.

**Also Try:**
- 🖼️ **Gemma3:4B — the first NPU-only VLM!**  
Choose `gemma3:4b`, click `+` → Upload files, and add your PNG/JPG images.  

- 🌐 **Web Search — local agentic AI–powered search**  
Open *Integrations* (below the chatbox), toggle on `Web Search`, and start searching instantly.  

- 🗂️ **RAG (Retrieval-Augmented Generation)** — your secure, local knowledge system  
Select the `FLM-RAG` model (powered by Qwen3-Thinking-2507-4B) with a knowledge base pre-built from the FLM GitHub repo, and ask anything about **FastFlowLM**! 

📺 [Watch this short video](https://www.youtube.com/watch?v=0AhkX2ZLu7Y&list=PLf87s9UUZrJp4r3JM4NliPEsYuJNNqFAJ&index=2) to see how to try the remote demo in just a few clicks.

> ⚠️ **Please note**:
> - Some universities or companies may **block access** to the test drive site. If it doesn't load over Wi-Fi, try switching to a **cellular network**.  
> - FastFlowLM is designed for **single-user local use**. This remote demo machine may experience short wait times when **multiple users** access it concurrently — please be patient.
> - When switching models, it may take longer time to replace the model in memory.
> - Large prompts and VLM (gemma3:4b and gwen3vl:4b) may take longer — but it works! 🙂

---

## 📚 Sections

### 🚀 [Get Started](install/)
Quick 5‑minute setup guide for Windows.

### 🛠️ [Instructions](instructions/)
Run FastFlowLM using the CLI mode or server mode.

### 📊 [Benchmarks](benchmarks/)
Real-time performance comparisons vs AMD's official stack and other tools.

### 🧩 [Models](models/)
Supported models, quantization formats, and compatibility details.
