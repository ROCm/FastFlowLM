---
layout: page
title: "Benchmarks"
permalink: /benchmarks/
description: "Transparent Ryzen AI telemetry for chat, multimodal, and agent workloads."
sections:
  - type: hero
    kicker: "Benchmarks"
    title: "Measured on real laptops."
    body: |
      Every FastFlowLM release is validated on Strix Point and Halo reference designs.
      We publish the results in `docs/benchmarks` so teams can compare apples-to-apples.
    ctas:
      - label: "View benchmark docs"
        href: "/docs/benchmarks/"
        style: primary
    right:
      metrics:
        - label: "Llama3.2 3B @ 4-bit"
          value: "72 tok/s"
          desc: "Ryzen™ AI 9 HX 370 · 8 ms p50 latency"
        - label: "DeepSeek-R1 7B"
          value: "41 tok/s"
          desc: "Reasoning traces enabled · < 11 W"
        - label: "Gemma3 4B Vision"
          value: "18 fps"
          desc: "Image + text streaming pipeline"

---

