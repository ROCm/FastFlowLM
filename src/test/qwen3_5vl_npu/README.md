# Qwen3.5 speed harness (`test_qwen3_5vl_npu`)

A small tool that loads one Qwen3.5 model on the NPU and measures **time to first token (TTFT)**,
**prefill speed** and **decode speed** for a list of prompt lengths. It also prints the first answer at
each length so you can see the output is still correct after a change.

## Build and run

```bash
cd /scratch/michyu/FastFlowLM/src/test/qwen3_5vl_npu
make -j 12
make test                                   # qwen3.5-flash:2b; defaults: 128,1024,4096,10240, focus 10240
make test MODEL=qwen3.5:2b                  # the original 2B engine and kernels
make test ARGS="--csv /tmp/run.csv"         # extra options go in ARGS
make test FLM_MODEL_PATH=/other/models_root # another model root
```

- **Always run through `make test`.** It copies `model_list.json` and `bench_text.txt` next to the binary
  and runs it from the build folder, `src/build/test/qwen3_5vl_npu`.
- **Model folder:** `make test` sets `FLM_MODEL_PATH` (default `/scratch/michyu`), the folder that
  **contains** `models/`. The model lives in `/scratch/michyu/models/Qwen3.5-2B-NPU2`. Without `FLM_MODEL_PATH`
  the code falls back to `~/.config/flm/models` and fails with "Cannot open .../tokenizer.json". The folder
  name must match the `name` in `model_list.json` exactly (`Qwen3.5-2B-NPU2`, case-sensitive). Both 2B tags read
  it. The xclbin subfolder is the entry's `xclbin_folder` when it has one (`Qwen3.5-2B-Flash-NPU2` for
  `qwen3.5-flash:2b`), else the same name.
- **Xclbins:** the harness is built with `-DDEV_BUILD` (`../common.mk`). In that mode `LM_Config` uses
  `../../../xclbins/<model folder name>/` relative to the **current folder** and ignores `FLM_XCLBIN_PATH`.
  From the build folder that is the repo's `src/xclbins/Qwen3.5-2B-Flash-NPU2/` (flash) or `src/xclbins/Qwen3.5-2B-NPU2/`. Running the binary from
  anywhere else finds no xclbins.
- The tool prints the model and xclbin folders at start and stops with an `ERROR` line if `model.q4nx`,
  `config.json`, `tokenizer.json`, `layer.xclbin` or the prompt file is missing.
- The engine `.so` comes from `../../lib/xrt` (rpath): `libqwen3_5vl_flash.so` for `qwen3.5-flash:2b`,
  `libqwen3_5vl_npu.so` for the others. To try your own build, put only that file in a folder and add it in front
  with `LD_LIBRARY_PATH`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `-m, --model` | `qwen3.5:2b` | tag in `model_list.json` |
| `-l, --len` | `128,1024,4096,10240` | prompt lengths in tokens, comma separated |
| `--focus` | `10240` | length printed as the one-line `FOCUS` result at the end |
| `-g, --gen` | `128` | tokens to decode per run |
| `-i, --iters` | `3` | measured runs per length |
| `-w, --warmup` | `1` | unmeasured runs before measuring, at every length |
| `-f, --prompt-file` | `bench_text.txt` next to the binary | base text; relative paths are made absolute at start-up |
| `-c, --csv` | none | one row per run |
| `-s, --show` | `1` | print the first answer at each length |
| `-t, --think` | `0` | thinking on in the chat template |
| `-e, --engine-log` | `0` | let the engine print to stdout during runs |

## How it measures (and why)

- **Prompt of an exact length.** `bench_text.txt` is the same story `flm bench` uses
  (`src/src/benchmarking.hpp`). The harness loads its own `Tokenizer` from the model folder (the engine's
  tokenizer is `protected` in `AutoModel`), repeats the story's tokens until there are N, and decodes
  them back to text. The chat template adds about 8 tokens, so the "prompt" column shows N + 8.
- **No cache hits.** The prefix slot cache is switched off (`set_prefix_limits(-1, 0)`, 0 slots = off)
  and `clear_context()` runs before every run. If `cached_prompt_tokens` is ever above 0 a `WARN` line is
  printed, because then the prefill number would not be a real prefill.
- **Per-call numbers.** Speeds come from `chat_meta_info_t` of that one call:
  prefill = (prompt_tokens - cached) / prefill_duration, decode = generated_tokens / decoding_duration.
  The `show_profile()` totals are not used, because the profiler only adds up across calls.
- **TTFT** is wall time around `insert()` (prefill plus the first sampled token), timed with
  `std::chrono::steady_clock`.
- **Warmup at every length.** The first prefill at a new padded length sets up its buffers and
  instruction sequences again, so the first run at each length was 3-6% slower (10K: 7.69 s vs 7.24 s).
  Warming up at each length keeps that one-time cost out of the numbers (10K TTFT std 0.208 -> 0.012 s).
- **Greedy decoding** (`top_k 1`), so answers are the same run to run and easy to compare after a change.
- **Context length** for `load_model` is the smallest power of two (at least 8192) that fits
  the longest prompt + `--gen` + 256.
- **Quiet engine.** `modeling_qwen3_5vl.cpp` prints a `Model RAW Output:` block after every generate, with
  no switch. Instead of changing product code, the harness points `std::cout` at nothing while the engine
  runs (`cout_muter_t`: swap `rdbuf` to `nullptr`, put it back in the destructor) and prints its own lines
  in between. `--engine-log 1` turns this off for debugging.
- **CPU load** (`/proc/loadavg`) is printed at start, because host-side waiting means other CPU jobs move
  the numbers.

## Fixes needed to make it link again (2026-10-30)

`automodel.cpp` now needs code the old Makefile did not list:

- `common/AutoModel/model_backend.cpp` and `common/prefix_slot_cache.cpp` added to `SOURCES`;
- `-L../../build/libwebp/lib -lwebpdecoder` (image reader; built by the main CMake build);
- a `register_builtin_backends()` in `test.cpp` that registers only `qwen3.5`, like the minicpm harness.
  The real `builtin_backends.cpp` names every engine and would link every engine library.

Also: `model_list`'s constructor takes non-const `std::string&`, so pass named strings, not temporaries.

## Test focus: 10K prompt

The number we track after every change is the `FOCUS` line: a **10,240-token prompt** (10,248 with the chat
template), 128 decode tokens. It is printed last, for example:

```
FOCUS 10240 (prompt 10248): TTFT 7.213 s | prefill 1426.0 tok/s | decode 27.23 tok/s
```

## Baseline (2026-10-30, xsjnuckrck04, Krackan NPU, load < 0.6)

Weights: `model.q4nx` converted from `Qwen3.5-2B-q4_k.gguf` (all projections Q4_K, 4736 B chunks).
Default run (`--len 128,1024,4096,10240 --gen 128 --iters 3 --warmup 1`). Check the mode with
`xrt-smi examine -r platform | grep -i "power mode"` and quote it with every number.

**Performance power mode (current baseline, second of two runs; the first had a one-off slow 1K prefill):**

| prompt tokens | TTFT s | prefill tok/s | decode tok/s |
|---|---|---|---|
| 136 | 0.461 | 300 | 32.44 |
| 1032 | 1.091 | 952 | 31.86 |
| 4104 | 3.069 | 1342 | 29.68 |
| **10248 (focus)** | **7.182** | **1430** | **26.44** |

Default power mode, for comparison (decode was 2-3 % faster there, prefill about the same):

| prompt tokens | TTFT s | prefill tok/s | decode tok/s |
|---|---|---|---|
| 136 | 0.453 | 306 | 33.20 |
| 1032 | 1.044 | 997 | 32.58 |
| 4104 | 3.038 | 1358 | 30.58 |
| 10248 | 7.213 | 1426 | 27.23 |

## 2B layer.xclbin with lm_head inside (2026-11-02)

`src/xclbins/Qwen3.5-2B-NPU2/layer.xclbin` is now the build from
`FastFlowLM_IRON/FLM_Xclbin/Qwen3_5_2B/qwen3_5_decoding` (md5 e86e2b22): lm_head runs inside it, so it only
works with an engine that writes its mode RTPs. At the time `make test` put a private engine copy first on
`LD_LIBRARY_PATH`. Replaced on 2026-11-03 by the `qwen3.5-flash:2b` split (last section).

| `make test`, NPU otherwise idle | TTFT | prefill tok/s | decode tok/s |
|---|---|---|---|
| 128 | 0.459 s | 301.0 | 37.88 |
| 1024 | 1.070 s | 971.9 | 37.07 |
| 4096 | 3.068 s | 1342.8 | 34.95 |
| **10240 (FOCUS)** | **7.233 s** | **1420.8** | **31.59** (was 26.44) |

With the newest prefill as well (2026-11-02): the engine copy now also carries the uncommitted prefill work of
`FLM_DLL/detail/qwen3_5vl_npu` (patched in; loads `mm_v2`, `conv_v2`, `GateDeltaNet_prefill_v2`, `attn_v2`):

| `make test`, NPU otherwise idle | TTFT | prefill tok/s | decode tok/s |
|---|---|---|---|
| 128 | 0.446 s | 310.3 | 37.83 |
| 1024 | 0.701 s | 1487.2 | 37.12 |
| 4096 | 1.763 s | 2341.4 | 34.72 |
| **10240 (FOCUS)** | **4.085 s** | **2519.5** | **31.14** |

## Two 2B models (2026-11-03)

`qwen3.5:2b` is the original again (origin/main engine and xclbins). Everything above went into a separate tag,
`qwen3.5-flash:2b`: engine `libqwen3_5vl_flash.so` (`FastFlowLM_IRON/FLM_DLL/detail/qwen3_5vl_flash`, notes in its
`PREFILL_SPEEDUPS.md`), xclbins in `src/xclbins/Qwen3.5-2B-Flash-NPU2`, the same weights folder. `make test` runs
the flash tag; `make test MODEL=qwen3.5:2b` runs the original. `nll_test` takes `-m` the same way.

| `make test ARGS="--iters 2"`, NPU otherwise idle | `qwen3.5:2b` TTFT / prefill / decode | `qwen3.5-flash:2b` TTFT / prefill / decode |
|---|---|---|
| 128 | 0.425 s / 325 / 34.04 | 0.357 s / 387 / 37.73 |
| 1024 | 1.025 s / 1013 / 33.45 | 0.560 s / 1864 / 37.10 |
| 4096 | 2.988 s / 1378 / 31.48 | 1.306 s / 3167 / 34.66 |
| **10240 (FOCUS)** | **7.056 s / 1456 / 28.82** | **2.939 s / 3507 / 31.16** |

Kernels moved to Peano (2026-11-03): decode, conv1d, dequant_mm and GDN on Peano 22 (attention and vision overlay stay
Chess). `qwen3.5-flash:2b`, `make test ARGS="--iters 2"`: 128 0.342 s / 406 / 37.93, 1024 0.610 s / 1711 / 37.21,
4096 1.440 s / 2872 / 34.91, **10240 3.242 s / 3178 / 31.44** (TTFT / prefill tok/s / decode tok/s). dequant_mm (+24 %)
and GDN (+32 %) are slower under Peano; their READMEs have the details.

Then attention and the vision overlay moved to Peano too (bit-identical to Chess; attention needed the `.bss` fix and
Peano-only softmax vectorizing, the overlay the upstream Peano MM / attention commits). Every 2B flash kernel is now
built with Peano. `make test ARGS="--iters 2"`: 128 0.332 s / 418 / 37.81, 1024 0.586 s / 1780 / 37.12, 4096 1.444 s /
2861 / 34.89, **10240 3.245 s / 3176 / 31.47**.

