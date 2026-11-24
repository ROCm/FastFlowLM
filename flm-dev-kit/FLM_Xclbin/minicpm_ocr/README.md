<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# minicpm_ocr kernels — starting point for the GDN/conv1d prefill optimization

Unmodified copies, byte-identical to their sources as of 2026-10-28:

| here | copied from |
|---|---|
| `conv1d_prefill/` | `Qwen3_5/conv1d_prefill` |
| `gate_delta_net_prefill/` | `Qwen3_5/gate_delta_net_prefill/hostDeltaNetPrefill` |

Those are the sources `FLM_Xclbin/kernel_assignment.md` names for `minicpm-v-4.7:1b`, whose xclbins
MiniCPM-OCR reuses. **Note they are Qwen3_5's, not Qwen3_6's** — a similar GDN prefill kernel also
lives under `Qwen3_6/`, and it is the wrong one.

Nothing here is modified yet, and no bitstream has been built from it.

## The problem

`PREFILL_GDN` and `PREFILL_CONV` default to the **host** in `FLM_DLL/detail/minicpm_ocr_npu/`,
because the array is slower. Measured on MiniCPM-OCR, time in the 18 GatedDeltaNet layers,
interleaved A/B:

| prompt tokens | host | array | ratio |
|---|---|---|---|
| 1,011 | 282 ms | 513 ms | 1.82x |
| 8,015 | 2,288 ms | 2,695 ms | 1.18x |
| 24,031 | 6,827 ms | 7,497 ms | 1.10x |

The array's handicap is mostly fixed overhead and it does amortise — 82% at 1k down to 10% at 24k —
but the curves never cross. Fitting 8k → 24k: **0.283 ms/token host vs 0.300 array**, so the array
stays ~6% more expensive *per token*. Full table in `FLM_DLL/detail/minicpm_ocr_npu/README.md`.

## Why, from reading the kernel

`gate_delta_net_prefill_128_128_128_32_16_128.cc` has **zero** uses of `aie::mmul` and five of
`aie::broadcast`. That follows from the maths, not from sloppiness: the delta rule is applied one
token at a time,

```
S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T
o_t = S_t q_t
```

so `S@k` is a matrix-**vector** product and the state update is a **rank-1** outer product. Neither
reaches the 8x8x8 bf16 matmul datapath, and both run at an arithmetic intensity near **1 MAC/byte**
across the whole 128x128 state. That is precisely the shape of work the host does well with the
state resident in L2.

`CT_L_size` (16) and `MT_L_size` (128) are **DMA granularity, not maths chunks** — the recurrence
underneath is still strictly token-serial. Do not mistake them for a chunked formulation.

## The intended fix

The chunked (WY transform) delta rule: process C tokens at once.

| | today | chunked, C=64 |
|---|---|---|
| `S@K` | matvec (broadcast+MAC) | **matmul** → `mmul` |
| state update | rank-1 | **rank-C** → `mmul` |
| arithmetic intensity | ~1 | **~C** |

This is what `chunk_gated_delta_rule` does in flash-linear-attention, and what the PyTorch reference
already uses. **The reference exists for free:** the PyTorch reference (not shipped) cross-checks
the chunked formulation against token-serial per layer and currently passes at 1e-7.

A second, independent cost worth measuring before assuming the maths is everything: `conv.xclbin`
and `GateDeltaNet_prefill.xclbin` are two separate xclbins, so each of the 18 layers may pay
bitstream switches the host path never does. Fusing conv into the GDN design, or hoisting the
switch out of the layer loop, is a smaller change than the chunked rewrite and may be worth doing
first. On NPU2 a runtime op switch costs ~0 us and a context switch ~820 us.

## Suggested order

1. **Get `make host && ./run.exe` working here first.** It gives a seconds-long iteration loop
   instead of rebuilding the engine, and the guide requires a NumPy reference with edge cases
   (lengths that are not tile multiples, the padding value, max length) regardless.
2. **Profile compute vs DMA/lock/switch** before rewriting, so the work targets what dominates
   rather than the hypothesis above.
3. Then the chunked rewrite.

Build: `source FLM_Xclbin/setup.sh` (set `MLIR_AIE_DIR` and `PEANO_INSTALL_DIR` first), then
`make link` to fail fast before `make bitstream` (5–30 min).

If a variant is added, it must also go in `FLM_Xclbin/kernel_assignment.md`.

## Peano (2026-11-04)

Both kernels now build with Peano by default (`ENABLE_CHESSCC ?= 0`); `ENABLE_CHESSCC=1` still
builds chess, and that build is unchanged (same output, same speed). Peano output is
**bit-identical** to chess for both kernels, at every length tested.

Environment (Peano `llvm-aie 22.0.0.2026090201+a36c62b9`):

```shell
export MLIR_AIE_DIR=/path/to/mlir-aie
export PEANO_INSTALL_DIR=/path/to/llvm-aie-22.0.0.2026090201/llvm-aie   # the pinned Peano
# Chess only (ENABLE_CHESSCC=1): export AIETOOLS_ROOT=<Vitis AIE Essentials> LM_LICENSE_FILE=<license>
source ../../setup.sh   # FLM_Xclbin/setup.sh
make bitstream          # ~10 s here
```

Standalone, interleaved with chess, 3 rounds of 10 runs (16 heads for GDN, D 6144 for conv8):

| L | GDN chess | GDN Peano | conv8 chess | conv8 Peano |
|---|---|---|---|---|
| 1,024 | 1.95-2.40 ms | 1.85-2.06 ms | 0.69-0.70 ms | 0.70 ms |
| 2,048 | 3.60-3.64 ms | 3.15-3.23 ms | 1.28-1.30 ms | 1.29-1.32 ms |
| 8,192 | 13.45-13.48 ms | 12.02-12.22 ms | 4.92-5.03 ms | 4.86-5.01 ms |
| 24,000 | 38.90-38.96 ms | 34.90-35.01 ms | 14.19-14.28 ms | 14.16-14.24 ms |

Bit-exactness was also checked at partial-chunk lengths (GDN: 1, 17, 125, 1535, 4097, with
`GDN_ALPHA_MIN` 0.1 and 0.001 so the fold path runs; conv8: 1, 3, 17, 255, 1023, 4097, with
1-3 runs back to back). `GDN_SAVE=<file>` / `CONV_SAVE=<file>` make `run.exe` write the raw
outputs, which is how two builds are compared bit for bit.

Program memory per core: GDN 11,200 B (chess ~9,000 B), conv8 5,968 B (chess 7,012 B).

### What the GDN needed

The first Peano build was correct but 33% slower (17.9 vs 13.4 ms at 8k). All of it was the
inner loop of `update_pass` (one pass over S' per token): 22 bundles per column pair against
chess's 13, not software-pipelined.

1. **Walking pointers.** `pair_bcast(k, col)` and `S_rd + col * V_chunk` made Peano rebuild every
   address from `col` (shift, add, indexed load) in each iteration, and that stopped its
   pipeliner. The Peano path steps `k`, `q_s`, `k_next` as `int32_t` pointers and `S_rd`/`S_wr`
   by one pair (64 floats) per iteration. 22 -> 16 bundles; 14.1 ms at 8k.
2. **`AIE_TRY_INITIATION_INTERVAL(4)`** on the same loop (empty under chess). It asks the Peano
   pipeliner for a tighter schedule: the main version (no fold, has next token) goes 16 -> 12
   bundles, one better than chess. The value is not a real target -- no loop gets 4 -- and the
   pipeliner's answer is not monotonic in it: 4 gives 12 / 6 bundles for the two no-fold
   versions, but 6..11 push the no-next-token version from 6 to 19 bundles, and 16+ give 16.
   Recount the loop bundles (below) after any change to this loop or a Peano update.

The two fold versions stay at 52 bundles. They only run when the lazy decay A drops below
2^-64 (about every 45-60 tokens with the test's random alphas), so they cost little.

Counting loop bundles: the Peano build writes `build/kernel/*.asm`; a hardware loop body runs
from its `.LBB..:` label to `.L_LEnd..:` plus the one bundle after it.

**`PROFILE=1` needs chess.** The profile counters use `aie::tile::current().cycles()`, which is
`get_cycles()`, and Peano has no such symbol (link error). Reading the core's Timer_Low with
`read_tm(0x340F8)` instead links, but hangs the core.

### What conv8 needed

Only the build path (`makefiles/kernel.mk`, `bitstream.mk`, and the BFP16 flag on the Peano line
of `common.mk`, as in `Qwen3_5/conv1d_prefill`) and the ping/pong flags in `conv.cc`. Those are
`static bool`s that must start false and keep their value across runs: one run flips `is_x_ping`
7 + L times and `is_o_ping` L times, so after an odd L the next run starts on the other buffer,
where the DMA is. Peano's loader does not zero `.bss`, and Peano does not treat a lock acquire
as a memory barrier, so under Peano they are `AIE_PING_FLAG` = `.data` + `volatile`
(`aie_kernel_utils.h`, the same fix as `dequant_mm_512x512x512` on main). No speed work was
needed.

Both Makefiles now include `ENABLE_CHESSCC` in their kernel flag stamp, so switching compilers
rebuilds the kernel objects.

## vision_mm and vision_attn (2026-11-04)

Copies of `Qwen3_8/vision_mm` and `Qwen3_8/vision_attn` (sources only, no build output), with main's Peano
commit dd96bf8a applied to the copies: the ping/pong selectors are locals under Peano, attention's `neg_inf` /
`one` are not function-local statics, and the makefiles build with Peano by default (`ENABLE_CHESSCC=1` builds
Chess, unchanged). `Qwen3_8/` itself is untouched; MiniCPM-OCR's shipped vision xclbins still come from there.

Peano output is bit-identical to the Chess builds (whole encode, icon / 720p / 1080p); RTP addresses unchanged
(A 24576, B 7936, attention 57344), so the engine needs no change. Device time: matmuls the same, attention
14-16% faster. Program size: mm A 4,464 B, mm B 3,856 B, attention 8,272 B (Chess 6,084 / 6,724 / 10,964).

They are also the sources of `vision_overlay/` (attention + design A in one xclbin; see its README).

## dequant_mm (2026-11-04)

Copy of `dequant_mm_512x512x512` (sources only; MiniCPM-OCR's `mm.xclbin`, the language model's prefill matmuls),
with main's Peano commit 3e3c0daf applied: Peano build path (default), ping/pong selectors in `.data` + volatile,
the mmul B operand converted with the accumulator constructor, and the odd tile's buffers pinned so x and w sit in
different banks. `makefiles/bitstream.mk` checks that the RTP buffers stay where FLM_DLL writes them (K_div_k /
rtp_act at 54784 / 54848, +6912 on odd rows). The shared `dequant_mm_512x512x512/` is untouched.

Only other change: `makefiles/host.mk` `HOST_COMMON_DIR` is `../../host_common` (one level deeper), and the test
takes `DQMM_DIN`, `DQMM_DOUT`, `DQMM_L`, `DQMM_ACT`, `DQMM_ITERS`, `DQMM_XCLBIN`, `DQMM_SAVE` from the environment.
Its `gen_dequant_mm_512` is the engine's (`minicpm_ocr_npu_sequence.cpp`) but for two unused variables.

Against the shipped Chess `mm.xclbin` (`f09ce18ed017`), the seven shapes the engine runs (1024->6144,
1024->2048 with SiLU, 2048->1024, 1024->2048, 1024->512, 1024->7168, 3584->1024) at L 512 / 2048 / 8192: output
bit-identical in all 21 cases; Peano/Chess time 0.90-1.03 (2 interleaved runs each, mostly equal or a few % faster).
Program size ~4.8 KB per core (even and odd rows).

## dequant_mm_prefill (2026-11-05)

Copy of `origin/main:FLM_Xclbin/Qwen3_5_2B/dequant_mm_prefill` (the Qwen3.5-2B flash prefill matmul: x as bfp16 tiles
made on the host, W converted to bfp16 once per tile, per-core y, 256-k passes), Peano. The kernel needed no change
for MiniCPM-OCR's 7 shapes and is bit-identical to `dequant_mm/` on all of them at L 512 / 2048 / 8192; mm time per
prefill at L 8192 945 -> 599 ms. It then got a GLU epilogue (`rtp_act` 2): the up/gate run writes
`silu(gate) * up` itself, as the down run's bfp16 x tiles. Details, tests and numbers: `dequant_mm_prefill/README.md`.
The engine registers it as `mm_bfp16.xclbin` (host GLU) or `mm_bfp16_glu.xclbin` (GLU on the array).
