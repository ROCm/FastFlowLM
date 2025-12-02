---
layout: page
title: "How It Works"
permalink: /how-it-works/
description: "How FastFlowLM squeezes every watt and token out of AMD Ryzen™ AI NPUs."
sections:
  - type: hero
    kicker: "Technology deep dive"
    title: "FastFlowLM is built NPU-first"
    body: |
      FastFlowLM retools the entire inference stack for AMD's XDNA-based Ryzen™ AI silicon.
      Instead of mirroring GPU kernels, we partition prefill and decoding into tile-sized workloads,
      keep KV state resident on-chip, and stream attention math through the NPU's 12 TFLOPS bf16 fabric.

      The result: up to **5.2× faster prefill** and **4.8× faster decoding** than the iGPU, **33.5× / 2.2×**
      gains versus the CPU, while drawing **67× / 223× less power**. We also lift the context ceiling from **2K to
      256K tokens** and halve Gemma 3 image TTFT from **8 s down to 4 s** on the same laptop-class part.
    ctas:
      - label: "See benchmarks"
        href: "/benchmarks/"
        style: primary
    right:
      metrics:
        - label: "Prefill acceleration"
          value: "5.2× vs iGPU"
          desc: "Ryzen AI NPU prefill throughput."
        - label: "Decoding acceleration"
          value: "4.8× vs iGPU"
          desc: "Tile-aware token streaming."
        - label: "Context window"
          value: "256K tokens"
          desc: "Up from 2K in stock stacks."
        - label: "Power draw"
          value: "67× / 223× less"
          desc: "NPU vs iGPU / CPU."
        - label: "Image TTFT"
          value: "4 s"
          desc: "Gemma 3 vision, down from 8 s."

  - type: two_column
    left:
      kicker: "Parallel-by-design"
      title: "Fine-grained orchestration on XDNA"
      body: |
        The Ryzen AI chip blends an iGPU and an always-on NPU, both rated at 12 TFLOPS bf16 compute.
        While the GPU enjoys 125 GB/s of memory bandwidth, the NPU sits at 60 GB/s—so FastFlowLM
        had to attack the problem with software-led tiling, compression, and scheduling.
      items:
        - heading: "AIE tile partitioning"
          body: |
            We map transformer blocks to configurable tiles, fuse matmuls + activation, and ensure the
            compute fabric never waits on host memory.
        - heading: "Streaming KV residency"
          body: |
            Attention state stays inside NPU SRAM, enabling 256K-token prompts without bouncing to LPDDR.
        - heading: "Dynamic power envelopes"
          body: |
            Always-on inference taps the NPU's low-leakage island, yielding the 67×/223× power savings cited above.
    right:
      panel: true
      title: "Execution phases"
      body: |
        FastFlowLM dissects inference into deterministic phases so the runtime can pipeline work across CPU, GPU, and NPU.
      items:
        - heading: "Prefill turbo"
          body: |
            Token embedding, rotary math, and large matmuls are staged across contiguous tiles for the 5.2× prefill gains.
        - heading: "Token streaming"
          body: |
            Lightweight decode kernels reuse on-chip KV blocks, hold steady at 4.8× faster than the iGPU, and avoid cache thrash.
        - heading: "Vision + multimodal"
          body: |
            Image TTFT drops from 8 s to 4 s by overlapping patch projection with text prefill on separate compute islands.

  - type: cards
    kicker: "Edge to rack"
    title: "Scaling the FastFlowLM approach"
    body: |
      Ryzen AI proves the concept locally, but the same architecture is already moving toward rack-scale NPU deployments.
    cards:
      - label: "Edge laptops"
        title: "Always-on, efficient AI"
        body: |
          NPUs stay cool handling assistants, copilots, and background perception without burning the iGPU/CPU.
          Users avoid the thermal throttling that plagues CPU/iGPU-only stacks.
      - label: "Rack roadmap"
        title: "Qualcomm AI200/AI250 & AMD discrete NPU"
        body: |
          Qualcomm is redefining rack inference with Hexagon NPUs, while AMD is building a discrete XDNA accelerator.
          FastFlowLM's close-to-metal runtime is ready for both paths.
      - label: "Projected scaling"
        title: "10× GPU-class throughput"
        body: |
          With equal compute and bandwidth, our rack NPU plan models >10.4× faster prefill, >9.6× faster decoding,
          and >114× better TPS/W than GPU baselines.

  - type: media
    kicker: "Rack architecture"
    title: "From chips to racks, all tiled structure"
    body: ""
    media:
      src: "/assets/architecture.png"
      alt: "Diagram of racks feeding an AIE and memory tile layout"
      title: "Racks of AIE + memory tile array"
      items:
        - heading: "Racks of NPU Chips"
          body: |
            The images on the left are photos of multiple server trays and chassis that host NPUs in a rack.
        - heading: "AIE compute tiles"
          body: |
            In the schematic, the upper blocks labeled "AIE tile" represent an array of programmable compute tiles.
        - heading: "Memory tiles"
          body: |
            The darker blocks at the bottom labeled "Memory tile" show the on-chip memory region that sits alongside the AIE tiles.
        - heading: "FastFlowLM Technology"
          body: |
            Our FastFlowLM software scales up and scales out naturally to rack level inferences.    
---

