---
layout: page
title: "Models"
permalink: /models/
description: "Curated Ryzen AI-ready model catalog with first-class FastFlowLM recipes."
sections:
  - type: hero
    kicker: "Model catalog"
    title: "Choose a recipe"
    body: |
      FastFlowLM curates the most requested families and publishes tuned manifests under `flm pull <model>`.
      We validate every build on Ryzen™ AI laptops and provide matching markdown cards in `docs/models`.
    ctas:
      - label: "View model docs"
        href: "/docs/models/"
        style: primary
    right:
      title: "Supported families"
      body: |
        Each manifest describes quantization, context window, tokenizer, and recommended memory,
        so there are no surprises after download.
      pills:
        - "Llama 3.2"
        - "Gemma 3"
        - "DeepSeek"
        - "Qwen 3"
        - "GPT-OSS"
        - "FLM MoE"
        - "Whisper"
        - "EmbeddingGemma"

  - type: two_column
    left:
      kicker: "Roadmap"
      title: "Upcoming drops"
      body: |
        Follow the GitHub project for rolling releases. We publish nightly builds for community testing and
        stable channels every few weeks.
      items:
        - heading: "Fill this in"
          body: "Coming soon!"
    right:
      title: "Stay in the loop"
      ctas:
        - label: "Release notes"
          href: "https://github.com/FastFlowLM/FastFlowLM/releases"
          style: primary
          external: true
        - label: "Discord"
          href: "https://discord.gg/z24t23HsHF?utm_source=site"
          style: ghost
          external: true
---

