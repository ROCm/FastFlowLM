---
layout: page
title: "FastFlowLM"
permalink: /
description: "Ollama-style runtime that unlocks AMD Ryzen AI NPUs with instant installs, rich models, and matching docs."
sections:
  - type: hero
    id: top
    kicker: "NPU-first runtime"
    title: "The fastest, most efficient LLM inference on NPUs"
    body: |
      FastFlowLM delivers an Ollama-style developer experience optimized for tile-structured NPU accelerators. Install in seconds, stream tokens instantly, and run context windows up to 256k — all with dramatically better efficiency than GPU-first stacks. Our GA release for AMD Ryzen™ AI NPUs is available today, with betas for Qualcomm Snapdragon and Intel Core Ultra coming soon.
    ctas:
      - label: "Download FastFlowLM (Windows)"
        href: "https://github.com/FastFlowLM/FastFlowLM/releases/latest/download/flm-setup.exe"
        style: primary
        external: true
      - label: "GitHub"
        href: "https://github.com/FastFlowLM/FastFlowLM"
        style: ghost
        external: true
    stats:
      - label: "Runtime size"
        value: "~16 MB"
      - label: "Context"
        value: "Up to 256k tokens"
      - label: "Supported chips"
        value: "Ryzen™ AI (Strix, Halo, Kraken)"
    right:
      carousel:
        slides:
          - title: "NPU‑First Architecture"
            description: "Built exclusively for AMD Ryzen™ AI NPUs with optimized kernels for maximum efficiency."
            visual:
              type: "npu_orbit"
              title: "Ryzen™ AI"
              subtitle: "FastFlowLM Runtime"
              metrics:
                - "10× power efficiency"
                - "256k ctx"
                - "Vision · Audio · Text"
          - title: "Ollama‑Style Developer Flow"
            description: "Same simple CLI & API patterns you already know — deeply optimized for NPU performance."
            visual:
              type: "code"
              title: "PowerShell"
              content: |
                flm run llama3.2:1b
                flm list
                flm serve llama3.2:1b
          - title: "Unmatched Performance"
            description: "Ultra‑lightweight runtime with exceptional power efficiency and massive context windows."
            visual:
              type: "metrics"
              metrics:
                - value: "10×"
                  label: "Power Efficiency"
                - value: "256k"
                  label: "Context Tokens"
                - value: "~16MB"
                  label: "Runtime Size"
          - title: "Multi‑Modal AI Support"
            description: "Run language models, vision models, audio processing, embeddings, and Mixture‑of‑Experts architectures."
            visual:
              type: "pills"
              pills:
                - "LLMs"
                - "VLMs"
                - "Audio"
                - "Embeddings"
                - "MoE"

  - type: media
    variant: alt
    kicker: "GPT-OSS on NPU"
    title: "GPT-OSS-20B streaming fully on the Ryzen™ AI NPU"
    media:
      src: "/assets/gpt-oss-demo.gif"
      alt: "GPT-OSS 20B running locally on the Ryzen AI NPU"
      href: "https://youtu.be/sZt1WyNoL2U?si=7U3z6u6E9KF6G_Dd"
      kicker: "GPT-OSS on NPU"
      body: |
        Runs GPT-OSS-20B at 19 TPS (token per second) with 10× GPU efficiency — the fastest MoE on any NPU.


  - type: media
    variant: alt
    kicker: "Whisper on-device"
    title: "Transcribe and summarize long-form audio locally"
    media:
      src: "/assets/cramer2.gif"
      alt: "Whisper transcription and summarization demo"
      href: "https://youtu.be/0t8ijUPg4A0?si=ETKdvig6lYiZb1Q_"
      kicker: "Whisper on-device"
      body: |
        Transcribe hours of audio locally — FLM runs OpenAI Whisper fully on the NPU — fast, private, and efficient.

  - type: media
    variant: alt
    kicker: "Llama 3.2 on WebUI"
    title: "Interact with Llama 3.2-3B through the FastFlowLM WebUI"
    media:
      src: "/assets/llama-demo.gif"
      alt: "Llama 3.2 chat demo running in the FastFlowLM WebUI"
      href: "https://youtu.be/mPrr9FLd8ps?si=vsyHkmtrBjP4s-dq"
      kicker: "Llama 3.2 on WebUI"
      body: |
        Runs Meta Llama 3.2-3B at 28 TPS with over 10× GPU efficiency — the fastest on any NPU.

  - type: two_column
    id: install
    left:
      kicker: "Install"
      title: "From download to first token in under a minute"
      body: |
        FastFlowLM ships as a 16 MB runtime with an Ollama-compatible CLI.
        No CUDA, no drivers, no guesswork—just run the installer, pull a model, and start chatting.
      items:
        - heading: "Zero-conf installer"
          body: "Signed FastFlowLM installers cover every Ryzen™ AI laptop—just download and run."
        - heading: "Drop-in APIs"
          body: "Compatible with Ollama, OpenAI, and Open WebUI endpoints for existing tooling."
        - heading: "Secure by default"
          body: "Local auth tokens, TLS, and offline mode keep your data on-device."
    right:
      kicker: "Quickstart"
      code_blocks:
        - title: "CLI"
          content: |
            Invoke-WebRequest https://github.com/FastFlowLM/FastFlowLM/releases/latest/download/flm-setup.exe `
              -OutFile flm-setup.exe
            Start-Process .\flm-setup.exe -Wait
            flm pull llama3.2:3b
            flm run llama3.2:3b --ctx 256k
        - title: "APIs"
          content: |
            POST /v1/chat/completions
            Authorization: Bearer $FLM_TOKEN

            curl -s localhost:11434/api/generate \
              -d '{"model":"gemma3:4b","prompt":"hello"}'

  - type: cards
    id: models
    kicker: "Models"
    title: "One CLI, every Ryzen-ready model"
    body: |
      Pull curated FastFlowLM recipes.
      The runtime streams tokens via HTTP, WebSocket, or the Ollama-compatible API, so existing apps work without rewrites.
    cards:
      - label: "Flagship reasoning"
        title: "Llama 3.2 · DeepSeek · Qwen 3"
        body: "Optimized kernels for 70B down to 1B, with automatic quantization and smart context reuse."
      - label: "Vision & speech"
        title: "Gemma 3 VLM · Whisper · Gemma Audio"
        body: "VLM and audio pipelines run on the NPU, enabling private multimodal assistants."
      - label: "Edge fine-tuning"
        title: "FLM MoE + Embedding suites"
        body: "Use built-in adapters, LoRA checkpoints, and embedding endpoints for retrieval workflows."
    ctas:
      - label: "Browse models"
        href: "/models/"
        style: ghost
      - label: "Model docs"
        href: "/docs/models/"
        style: ghost

  - type: two_column
    id: benchmarks
    variant: alt
    left:
      kicker: "Benchmarks"
      title: "Proof on silicon, not slides"
      body: |
        FastFlowLM is tuned on real Ryzen™ AI hardware with synthetic and application-level workloads.
        Expect steady 40–80 tok/s on 7B models at < 10 W, plus deterministic latency for agentic chains.
      items:
        - heading: "Full-stack telemetry"
          body: "Counters for NPU, CPU, and memory let you see exactly where cycles go."
        - heading: "Scenario-driven suites"
          body: "Instruction tuning, RAG, chat, and multimodal tests mirror real workloads."
      ctas:
        - label: "Benchmark details"
          href: "/benchmarks/"
          style: ghost
        - label: "View raw results"
          href: "/docs/benchmarks/"
          style: ghost
    right:
      metric_cards:
        - label: "Llama3.2 3B @ 4-bit"
          value: "72 tok/s"
          desc: "Ryzen™ AI 9 HX 370 · 8 ms median latency"
        - label: "Gemma 3 4B Vision"
          value: "18 fps"
          desc: "Vision + text pipeline on XDNA2 with shared memory"
        - label: "Power draw"
          value: "9.6 W"
          desc: "Full assistant stack vs ~45 W GPU baseline"

  - type: two_column
    id: docs
    right:
      ctas:
        - label: "Dive into docs"
          href: "/docs/"
          style: primary
        - label: "Installation guide"
          href: "/docs/install/"
          style: ghost
    left:
      title: "Remote test drive"
      body: |
        No Ryzen™ AI hardware yet? Launch the hosted FastFlowLM + Open WebUI sandbox and stream from a live
        AMD Ryzen™ AI box with 96 GB RAM.
      items:
        - heading: "Live hardware"
          body: "Same builds we use internally, refreshed with every release."
        - heading: "Guest access"
          body: "Instant login with rotating demo credentials."
        - heading: "Bring your apps"
          body: "Point your HTTP client at the public endpoint to try agent flows."
      ctas:
        - label: "Launch test drive"
          href: "/test-drive/"
          style: ghost
---

