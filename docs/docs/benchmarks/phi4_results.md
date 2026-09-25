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

## 🧪 Phi-4-mini-instruct Q8_0 GGUF on the rai backend (`phi4-mini-it:4b`, resolved for `aie_next`)

These are **descriptive measurements from individual acceptance runs**, not a benchmark sweep and not a pass threshold. Each figure below comes from one run, not from an average over many. They are not comparable to the tables above: the prompts here are 4–10 tokens, whereas those tables sweep 1k–32k, so the per-token rates are dominated by fixed overhead rather than by context length.

## Machine A

Two runs from this machine are reported. The **current** one is the full
acceptance matrix at the restructured tip, with the concurrent packer and the
on-disk weight cache both in play. The **earlier** one predates both; it is kept
because its load profiling is what the 45 s → 5 s section explains, and because
it is still the only run that measured the serial packer.

### Current run

Full acceptance matrix, `passed: true`, 0 failures.

#### Provenance

| | |
|---|---|
| Machine | aie_next development machine A |
| CPU | AMD Ryzen AI engineering sample |
| NPU | `AMD XDNA(TM) NPU` |
| OS | Microsoft Windows 11 Enterprise 10.0.26100 build 26100 |
| FastFlowLM commit | `b5a05ac2c798518a0969858bebe2ef5c7f1638eb` (plus two test-harness fixes that do not touch shipping code) |
| corelib commit / ABI | `3c35aebdefa3f0c2255668bab1be5648ece320f8` / `0.3.0` |
| corelib DLL SHA-256 | `f404da219a3cc84d3334c265e09ba7987f0c4bcc1b1cedeac7c3c45a7be2c9ae` |
| corelib actually loaded | confirmed from the live server's loaded-module list: exactly one `ryzenai_corelib.dll`, and it is the file above |
| GGUF revision | `78eb92a46fc37e6b524df991ed9aca9bc6aa7b80` |
| Tokenizer/config revision | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| Run | 2026-09-17 14:44:00 → 14:53:14 |

#### Measurements

The weight cache was already populated when this run started, so the load
figures are **cache-hit** loads: the packed weights are mapped from disk and
nothing is requantized.

| Metric | Value | Conditions |
|---|---|---|
| Model load, cache warm | **1.52 s** median (1.46 – 1.69 s, 11 loads) | fresh CLI processes, `Model loaded in` as reported by FLM |
| Cold TTFT | **3.92 s** | first prompt in a fresh process |
| Warm TTFT | **43.3 ms** | subsequent prompts in the same process |
| Decode, REST | **17.3 tok/s** | `/api/chat`, 16 generated tokens |

Cache-warm load is both faster and far tighter than the packing path machine B
measured (3.50 – 13.51 s, a ~4× spread): 11 loads inside a 0.23 s band. That is
the point of the cache — it replaces a variable cost with a fixed one.

The decode and TTFT figures are single observations on 4–10 token prompts and
carry the same caveats as the earlier run below. The REST decode figure moved
from 21.3 to 17.3 tok/s between the two runs; nothing measured here explains
that, and it is within the noise this document already warns about.

#### Functional results

- CLI — 10/10 fresh-process load-and-generate cycles exited 0, `Backend: rai` in every one.
- REST — `/api/chat` and `/v1/chat/completions`, streaming and non-streaming, all 200.
- Cancellation — an in-flight stream cancelled cleanly; the next request returned 200 on the same server.
- Capacity boundary — 4095 admitted, 4096 rejected with **HTTP 400** before submission.
- The four pinned model files are byte-identical to the earlier run. The two weight-cache files sit alongside them in the model directory and are excluded from that check by name.

### Earlier run

Measured at the commit named below, which predates the backend restructure, the
concurrent packer and the weight cache. Its load figures describe the **serial**
packer.

#### Provenance

| | |
|---|---|
| Machine | aie_next development machine A |
| CPU | AMD Ryzen AI engineering sample |
| NPU | `AMD XDNA(TM) NPU` |
| OS | Microsoft Windows 11 Enterprise 10.0.26100 build 26100 |
| Windows power scheme | Balanced (`381b4222-f694-41f0-9685-ff5bb260df2e`). The NPU power mode is separately set to `performance` by FLM at startup. |
| FastFlowLM commit | `87721089097396579ec4529f50616a6c0e1c7b74` |
| corelib commit / ABI | `3c35aebdefa3f0c2255668bab1be5648ece320f8` / `0.3.0` |
| corelib DLL SHA-256 | `f404da219a3cc84d3334c265e09ba7987f0c4bcc1b1cedeac7c3c45a7be2c9ae` |
| GGUF revision | `78eb92a46fc37e6b524df991ed9aca9bc6aa7b80` |
| Tokenizer/config revision | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| Run | 2026-09-12 00:34:19 → 00:52:02, `passed: true`, 0 failures |

#### Measurements

| Metric | Value | Conditions |
|---|---|---|
| Model load to serving | **5.1 / 5.3 s** | fresh `flm serve` processes, timed from launch to the first successful `/api/version`. Was 44–49 s at the accepted commit; see below. |
| Cold TTFT | **4.21 s** | first prompt in a fresh process; includes one-time kernel and ELF setup |
| Warm TTFT | **65.0 ms** | subsequent prompts in the same process |
| Decode, REST | **21.3 tok/s** | `/api/chat`, 16 generated tokens |
| Decode, warm CLI session | **35.8 tok/s** | 10 prompts in one loaded process |

#### Startup: 45 s → 5 s

The acceptance run measured 44–49 s to serving. Profiling it with `FLM_RAI_PROFILE_LOAD=1` found two independent costs, both since fixed:

| Phase | Before | After |
|---|---|---|
| Startup integrity check — SHA-256 over the 4 GB GGUF | ~28 s (62%) | **0 s** — not run |
| Weight requantization — 161 objects from Q8_0 | 15.3 / 14.9 s | **2.5 / 3.0 s** |
| Shape plan | 0.05 s | 0.05 s |
| GGUF resolve, host prep, device tensors | < 0.2 s | < 0.2 s |
| **Process launch to serving** | **44.9 / 46.3 s** | **5.1 / 5.3 s** |

The integrity check was re-hashing every pinned file on every launch — a pull-time concern on the startup path. `flm pull` and `flm check` still verify in full; only the run and serve paths were changed to ask for status alone.

The packer was being given a threads hint of 0, which corelib treats as ONE deliberately, so a single create packed on a single thread. Raising the hint brought requantization to 2.5–3.0 s here, within range of the 2.2 s `python/phi4_driver.py` reports for the same 161 weights. The packer has since moved to concurrent creates instead; machine B carries those figures.

Output was re-verified after the change: `2+2` → `4`, `capital of France` → `Paris`, `primary color` → `Red.`, and a correct one-sentence description of AMD.

Separately, and **not** fixed: `calculate_file_sha256` uses a portable pure-C++ SHA-256 with no hardware acceleration, and takes ~28 s over 4 GB where `Get-FileHash` on the same machine takes **3.67 s**. That ~8× gap is not specific to this model or backend — it is still paid by `flm pull` and `flm check` for every model.

**Do not read the per-process cold cycles as throughput.** Ten fresh-process cycles generating 8 tokens each reported 3.70–20.26 tok/s decode and 1.09–3.65 tok/s prefill. Every one of those pays the one-time setup inside its own measurement window, so the average describes start-up cost, not steady-state speed.

The **5.4×** spread between warm TTFT (65 ms) and cold TTFT (4.21 s), and the **1.7×** spread between the REST and warm-CLI decode figures, are both unexplained by anything measured here. Treat single-run differences below roughly 2× as noise.

#### Functional results

All from the same run:

- `flm pull` / `flm check` — four pinned files, all SHA-256 verified; the model directory contains exactly those four.
- CLI — 10/10 fresh-process load-and-generate cycles exited 0; the backend id and the loaded DLL path were reported in every one. The id that run printed was an earlier name for what is now `rai`, so it is described rather than quoted here.
- REST — `/api/chat` and `/v1/chat/completions` both 200, streaming and non-streaming.
- Cancellation — an in-flight stream cancelled cleanly; the next request returned 200 on the same server.
- Capacity boundary — a request totalling 4096 tokens is rejected with **HTTP 400** before submission (`rendered prompt has 4 tokens and requested output has 4092 tokens`); a 4095-token request is admitted.
- No CPU or NPU2 fallback appears in the server log at any point.

#### Known issue

One `/api/chat` reply to `What is 2+2?` came back as a truncated markdown image URL (`![](https://media.giphy.com/media/kZl76FZgu`, `done_reason: length`) instead of an answer. The identical prompt answered correctly on three other occasions in the same session, including the recovery request in the same run, so this looks like sampling nondeterminism rather than a routing fault — but it is a single-observation defect, it is not understood, and it is recorded rather than smoothed over.

---

## Machine B

Measured with the concurrent packer, after the backend restructure. **Load only**: TTFT, decode throughput and the acceptance matrix have not been re-run on this machine, so machine A remains the only source for those.

### Provenance

| | |
|---|---|
| Machine | aie_next development machine B |
| NPU | architecture `aie_next` |
| CPU | AMD Ryzen AI engineering sample, 20 cores |
| Memory | 32 GB |
| OS | Windows 11 Enterprise 10.0.26100.4652 |
| XRT / NPU driver / NPU firmware | 2.25.0 / 32.0.20214.4161 / 2.6.1.219 |
| corelib | `3c35aebd`, ABI 0.3.0, built on this machine |

### Model load

Weight requantization is effectively the whole of load; everything else — shape planning, GGUF resolution, host preparation, device allocation — stays under a quarter of a second combined.

| Packer | Requantization | Notes |
|---|---|---|
| Serial, one create at a time | **~30 s** | 29.99 / 30.19 / 25.14 s across runs, consistently slow |
| Concurrent, 8 creates in flight | **3.50 – 13.51 s**, median 8.76 s over 6 runs | one interactive run measured 7.37 s |

Two things are worth separating here.

The **consistent** 30 s came from the packer running effectively single-threaded in that session while the same binary was several times faster elsewhere. Taking the parallelism as threads FastFlowLM owns, rather than as a hint passed to the packer, removes that dependence on the surrounding environment.

What remains is **variance, not a fixed cost**: 3.50–13.51 s on an otherwise idle machine, a ~4× spread, and a separate ten-load run saw 6.74–19.41 s. Single measurements of this phase are not meaningful; quote a range. The variance is not explained by anything measured here.

### Not measured here

Load is where this machine was exercised. Cold and warm TTFT, decode throughput, the REST and cancellation matrix, and the capacity boundary were all measured on machine A at an earlier commit and have **not** been reconfirmed here.
