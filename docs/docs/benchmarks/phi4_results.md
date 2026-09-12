---
layout: docs
title: Phi4
parent: Benchmarks
nav_order: 6
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v0.9.30.
> - Under FLM's default NPU power mode (Performance)   
> - Newer versions may deliver improved performance.
> - Fine-tuned models show performance comparable to their base models. 

---

### **Test System 1:** 

AMD Ryzen™ AI 7 350 (Kraken Point) with 32 GB DRAM; performance is comparable to other Kraken Point systems.

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/bench/phi4_mini_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/bench/phi4_mini_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Phi-4-mini-instruct**  | NPU (FLM)    | 21.8	| 21.2	| 19.9	| 18.1	| 14.9	| 11.2|

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Phi-4-mini-instruct**  | NPU (FLM)    | 643	| 787	| 857	| 809	| 644	| 447 | 

---

## 🧪 Phi-4-mini-instruct Q8_0 GGUF on AIE4 (`phi4-mini-it-aie4:4b`)

These are **descriptive measurements from a single acceptance run**, not a benchmark sweep and not a pass threshold. They are not comparable to the tables above: the prompts here are 4–10 tokens, whereas those tables sweep 1k–32k, so the per-token rates are dominated by fixed overhead rather than by context length.

### Provenance

| | |
|---|---|
| Machine | `XCOMEDUSAD-43` |
| CPU | `AMD Eng Sample: 100-000001713-33_N` |
| NPU | `AMD XDNA(TM) NPU` |
| OS | Microsoft Windows 11 Enterprise 10.0.26100 build 26100 |
| Windows power scheme | Balanced (`381b4222-f694-41f0-9685-ff5bb260df2e`). The NPU power mode is separately set to `performance` by FLM at startup. |
| FastFlowLM commit | `87721089097396579ec4529f50616a6c0e1c7b74` |
| corelib commit / ABI | `3c35aebdefa3f0c2255668bab1be5648ece320f8` / `0.3.0` |
| corelib DLL SHA-256 | `f404da219a3cc84d3334c265e09ba7987f0c4bcc1b1cedeac7c3c45a7be2c9ae` |
| GGUF revision | `78eb92a46fc37e6b524df991ed9aca9bc6aa7b80` |
| Tokenizer/config revision | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| Run | 2026-09-12 00:34:19 → 00:52:02, `passed: true`, 0 failures |

### Measurements

| Metric | Value | Conditions |
|---|---|---|
| Model load to serving | **5.1 / 5.3 s** | fresh `flm serve` processes, timed from launch to the first successful `/api/version`. Was 44–49 s at the accepted commit; see below. |
| Cold TTFT | **4.21 s** | first prompt in a fresh process; includes one-time kernel and ELF setup |
| Warm TTFT | **65.0 ms** | subsequent prompts in the same process |
| Decode, REST | **21.3 tok/s** | `/api/chat`, 16 generated tokens |
| Decode, warm CLI session | **35.8 tok/s** | 10 prompts in one loaded process |

### Startup: 45 s → 5 s

The acceptance run measured 44–49 s to serving. Profiling it with `FLM_AIE4_PROFILE_LOAD=1` found two independent costs, both since fixed:

| Phase | Before | After |
|---|---|---|
| Startup integrity check — SHA-256 over the 4 GB GGUF | ~28 s (62%) | **0 s** — not run |
| Weight requantization — 161 objects from Q8_0 | 15.3 / 14.9 s | **2.5 / 3.0 s** |
| Shape plan | 0.05 s | 0.05 s |
| GGUF resolve, host prep, device tensors | < 0.2 s | < 0.2 s |
| **Process launch to serving** | **44.9 / 46.3 s** | **5.1 / 5.3 s** |

The integrity check was re-hashing every pinned file on every launch — a pull-time concern on the startup path. `flm pull` and `flm check` still verify in full; only the run and serve paths were changed to ask for status alone.

The packer was being given a threads hint of 0, which corelib treats as ONE deliberately. This requantizing path is compute-bound and scales with the hint, so 8 brings it to 2.5–3.0 s — within range of the 2.2 s that `python/phi4_driver.py` reports for the same 161 weights, and reached **without** the 8-concurrent-creates configuration whose failure mode is documented in corelib's header (2 of 10 loads producing all-zero output, attribution open). The creates remain serialized.

Output was re-verified after the change: `2+2` → `4`, `capital of France` → `Paris`, `primary color` → `Red.`, and a correct one-sentence description of AMD. No degeneration, no all-zero output.

Separately, and **not** fixed: `calculate_file_sha256` uses a portable pure-C++ SHA-256 with no hardware acceleration, and takes ~28 s over 4 GB where `Get-FileHash` on the same machine takes **3.67 s**. That ~8× gap is not specific to this model or backend — it is still paid by `flm pull` and `flm check` for every model.

**Do not read the per-process cold cycles as throughput.** Ten fresh-process cycles generating 8 tokens each reported 3.70–20.26 tok/s decode and 1.09–3.65 tok/s prefill. Every one of those pays the one-time setup inside its own measurement window, so the average describes start-up cost, not steady-state speed.

The **5.4×** spread between warm TTFT (65 ms) and cold TTFT (4.21 s), and the **1.7×** spread between the REST and warm-CLI decode figures, are both unexplained by anything measured here. Treat single-run differences below roughly 2× as noise.

### Functional results

All from the same run:

- `flm pull` / `flm check` — four pinned files, all SHA-256 verified; the model directory contains exactly those four.
- CLI — 10/10 fresh-process load-and-generate cycles exited 0; `Backend: corelib_aie4_gguf` and the loaded DLL path reported in every one.
- REST — `/api/chat` and `/v1/chat/completions` both 200, streaming and non-streaming.
- Cancellation — an in-flight stream cancelled cleanly; the next request returned 200 on the same server.
- Capacity boundary — a request totalling 4096 tokens is rejected with **HTTP 400** before submission (`rendered prompt has 4 tokens and requested output has 4092 tokens`); a 4095-token request is admitted.
- No CPU or NPU2 fallback appears in the server log at any point.

### Known issue

One `/api/chat` reply to `What is 2+2?` came back as a truncated markdown image URL (`![](https://media.giphy.com/media/kZl76FZgu`, `done_reason: length`) instead of an answer. The identical prompt answered correctly on three other occasions in the same session, including the recovery request in the same run, so this looks like sampling nondeterminism rather than a routing fault — but it is a single-observation defect, it is not understood, and it is recorded rather than smoothed over.
