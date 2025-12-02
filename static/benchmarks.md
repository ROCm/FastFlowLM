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
      
      The runtime extends AMD’s native 2K context limit to 256K tokens for long-context LLMs and VLMs and, in power
      efficiency tests, consumes 67.2× less power than the integrated GPU and 222.9× less power than the CPU on the same
      chip while holding higher throughput.
    ctas:
      - label: "View benchmark docs"
        href: "/docs/benchmarks/"
        style: primary
    right:
      metrics:
        - label: "Llama3.2 1B"
          value: "62.7 tps"
          desc: "AMD Ryzen™ AI 7 350 with 32 GB DRAM"
        - label: "Qwen 3 0.6B"
          value: "53.7 tps"
          desc: "Prefill speed: 1,280 tps"
        - label: "Gemma3 1B"
          value: "40 tps"
          desc: "Prefill speed: 1,004 tps"

  - type: media
    variant: alt
    kicker: "Llama 3.2 on Ryzen™ AI"
    title: "Prefill + decoding throughput across 256K tokens"
    gallery:
      - src: "/assets/bench/llama3-2-3b.png"
        alt: "Llama 3.2 3B prefill throughput on the Ryzen AI NPU"
      - src: "/assets/bench/llama3-2-3b-decoding.png"
        alt: "Llama 3.2 3B decoding throughput chart"
    items:
      - heading: "Long-context ready"
        body: "Screenshots are taken straight from the long-context regression run in `docs/benchmarks`."
      - heading: "Laptop-verified"
        body: "Instrumentation overlays remain visible from the Ryzen™ AI 9 HX 370 Halo reference design capture."

  - type: media
    variant: alt
    kicker: "Gemma3 4B Vision"
    title: "Image + text throughput on-device"
    media:
      src: "/assets/bench/gemma3-4b.png"
      alt: "Gemma3 4B benchmark overview for throughput and TTFT"
      title: "Vision + text TTFT and TPS"


  - type: media
    variant: alt
    kicker: "Qualcomm reference telemetry"
    # title: "Documenting partner power + thermal sweeps"
    media:
      # title: "Power, temp, and workload overlays"
      body: |
    gallery:
      - src: "/assets/bench/qualcomm15.png"
        alt: "Qualcomm benchmark instrumentation on a reference laptop"
      - src: "/assets/bench/qualcomm14-1.png"
        alt: "Qualcomm bench capture 1"
      - src: "/assets/bench/qualcomm14-2.png"
        alt: "Qualcomm bench capture 2"
      - src: "/assets/bench/qualcomm14-3.png"
        alt: "Qualcomm bench capture 3"
      - src: "/assets/bench/qualcomm14-4.png"
        alt: "Qualcomm bench capture 4"
    items:
      - heading: "On-device telemetry"
        body: "Photos include wattage, temperature, and throughput overlays straight from the Qualcomm reference UI."
      - heading: "Replayable sessions"
        body: "Visual artifacts are stored next to CSV logs so OEM teams can validate every published datapoint."

---

