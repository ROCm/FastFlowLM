---
layout: page
title: "Test Drive"
permalink: /test-drive/
description: "Remote sandbox for trying FastFlowLM on real AMD Ryzen AI hardware."
sections:
  - type: hero
    kicker: "Remote access"
    title: "Try FastFlowLM without a Ryzen™ AI laptop"
    body: |
      Launch our hosted FastFlowLM + Open WebUI environment backed by an AMD Ryzen™ AI box with 96&nbsp;GB RAM.
      Stream tokens, upload images, or run agentic workflows exactly as you would locally.
    ctas:
      - label: "Open test drive"
        href: "https://open-webui.testdrive-fastflowlm.com/"
        style: "primary"
        external: true
      - label: "Read the docs"
        href: "/docs/"
        style: "ghost"
    right:
      title: "Credentials"
      body: |
        Username: `guest@flm.npu`  
        Password: `0000`
        
        Prefer isolation? Create your own account inside Open WebUI. Sessions reset nightly.

  - type: two_column
    variant: "alt"
    left:
      kicker: "What's included"
      title: "Preloaded models & tooling"
      items:
        - heading: "Models"
          body: |
            Llama 3.2 3B/8B, Gemma 3 4B Vision, DeepSeek-R1 7B, Whisper, EmbeddingGemma.
        - heading: "Integrations"
          body: |
            Open WebUI, REST playground, LangChain quickstart scripts.
        - heading: "Monitoring"
          body: |
            Live NPU telemetry so you can watch throughput and power draw.
    right:
      title: "Limits"
      items:
        - heading: "Context"
          body: |
            4,096 tokens per chat for fairness.
        - heading: "Availability"
          body: |
            Server may reboot for OS updates. Try again in a few minutes if offline.

  - type: two_column
    left:
      kicker: "Bring your team"
      title: "Need a guided walkthrough?"
      body: |
        Schedule a live session with the FastFlowLM team.
        We'll tailor the demo to your use case (assistants, RAG, multimodal) and share logs afterward.
    right:
      ctas:
        - label: "Book a session"
          href: "mailto:info@fastflowlm.com"
          style: "primary"
        - label: "Chat on Discord"
          href: "https://discord.gg/z24t23HsHF?utm_source=site"
          style: "ghost"
          external: true
---

