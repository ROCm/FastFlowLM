<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Hy-MT2 copy (read this first)

This folder is a copy of `FLM_Xclbin/generic_decoding_layer` (2026-11-03), so Hy-MT2-1.8B
decode work does not touch the 9 other models that share the original. Default
`MODEL_TYPE=HUNYUAN_DENSE_1_8B`.

What changed against the original:

| | |
|---|---|
| `kernels/`, `iron/`, `models/` | `#ifndef DECODE_STUB_*` guards (below), the rescale skip, and bfp16 K and V in decode attention (below) |
| `makefiles/check_peano_stack.sh` | follows the call tree (`stack_depth.py`) instead of adding every frame |
| `host/` | the generic test host (no Hunyuan support) replaced by `host/bench_decode.cpp` |
| `makefiles/host.mk` | builds `bench_decode.exe` against `FLM_DLL` (engine code, Q4NX reader) |
| `makefiles/{kernel,bitstream}.mk` | build folder `build/$(BUILD_NAME)` instead of `build/$(MODEL_TYPE)` |

Checked: the kernel `.o` files are byte-identical to the original's, `rtp_addresses.txt` is
identical, and the engine gives 0/130 diffs on the translation test with this `layer.xclbin`.

## Build and run

```
# mlir-aie env, PEANO_INSTALL_DIR = July llvm-aie or newer (../README.md, Build notes)
make bitstream                   # build/HUNYUAN_DENSE_1_8B/xclbins/layer.xclbin (~45 s, Peano)
make host                        # bench_decode.exe
make test                        # = ./bench_decode.exe --xclbin build/HUNYUAN_DENSE_1_8B/xclbins/layer.xclbin
./bench_decode.exe --ctx 8,1024,32768 --reps 20 --xclbin <any layer.xclbin>
```

`bench_decode` loads the real weights from `/path/to/models/Hy-MT2-1.8B-NPU2` and
builds the decode sequences with the engine's own `hunyuan_desc` + `hunyuan_npu_sequence`
(`FLM_DLL/detail/hunyuan_npu`), so it runs exactly what production runs. It times only the
NPU: one runlist = the RTP run + 32 layer runs (one token), best of `--reps`; the lm_head is
timed alone. The KV cache is zeros (timing does not depend on the values). It does not check
outputs; correctness is checked through the engine (translation test, `test_long`).

Columns: `32 layers` ms per token, `per layer` us, `KV MB/token` = cache bytes read for one
token (all layers), `all bytes` = weights + KV, `GB/s` = all bytes / time.

## Timing experiment: STUB

`make bitstream STUB=proj|attn|all` builds into `build/HUNYUAN_DENSE_1_8B_stub_<name>`:

- `proj` skips `_QMM_x_bf16` in `proj_main.cc` (the weight matvec),
- `attn` skips `_attn_qk` in `attn_qk.cc` and `calculate_l` + `attn_fv` in `attn_kv.cc`,
- `all` both.

Every lock acquire/release and every DMA stays, so the same bytes move; only the core math
is gone. Outputs are wrong on purpose. If a stub runs no faster, that math was hidden behind
data movement. RTP addresses are the same as the normal build.

## Result, 2026-11-03 (Strix Halo test machine, NPU power mode Default)

Weights 27.13 MB per layer (868 MB per token), lm_head 139.2 MB, KV 2 KB per token per layer.

| context | normal (ms) | STUB=proj | STUB=attn | STUB=all | normal GB/s | all-stub GB/s |
|---|---|---|---|---|---|---|
| 8 | 16.91 | 16.83 | 16.89 | 16.79 | 51.4 | 51.8 |
| 1024 | 18.58 | 18.50 | 18.19 | 18.09 | 50.4 | 51.8 |
| 4096 | 23.62 | 23.52 | 22.26 | 22.17 | 48.2 | 51.3 |
| 16384 | 43.67 | 43.61 | 38.40 | 38.38 | 44.5 | 50.6 |
| 32768 | 70.40 | 70.32 | 60.01 | 59.93 | 42.9 | 50.3 |

lm_head 2.58 ms (53.9 GB/s). Token rate (32 layers + lm_head): 51.3 tok/s at 8, 13.7 at 32K;
the engine measures the same (48.5 at 512, 13.6 at 32K in `test_long`).

What it says:

1. **Decode is memory bound.** With all math removed the time hardly moves at short context,
   and every byte moves at the same ~50-52 GB/s.
2. **The weight matvec is fully hidden** (`STUB=proj` changes nothing at any context).
3. **Attention math shows at long context**: 10 ms of 70 ms at 32K (KV moves at 40 GB/s with
   the math, 50 GB/s without). Below ~4K it is lost in the noise.
4. **The ceiling is shared, not per channel.** Weights use 8 shim channels (shims 0,1,6,7),
   the KV cache 4 (shims 3,4), yet both run at ~50 GB/s. So spreading the KV read over more
   channels should not help; fewer bytes (smaller KV, smaller weights) or hiding the
   attention math would.
5. Not yet tried: NPU power mode Performance (may raise the ceiling, system-wide setting).

## Decode attention cost (Krackan, 2026-11-04)

STUB options added for this: `qk` (skip `_attn_qk`), `kv` (skip `calculate_l` + `attn_fv`),
`l` (skip `calculate_l` only), `fv` (skip `attn_fv` only). Per token, one runlist (32 layers + lm_head):

| | 4K | 12K | 24K |
|---|---|---|---|
| normal | 23.7 ms | 37.4 ms | 57.9 ms |
| `qk` | 23.6 | 37.4 | 56.3 |
| `l` | 23.1 | 35.6 | 54.2 |
| `fv` | **22.1** | **32.5** | **48.2** |
| `kv` | 22.1 | 32.4 | 48.0 |

* The Q·K core keeps up with the KV stream; the P·V core does not, and within it `attn_fv` (P·V with the
  fp32 running output y) is the cost: ~0.2 us per 16-key chunk per CU at 24K.
* `attn_fv`'s loop is ~290 bundles per iteration: 8 `vmac.f` (the P·V) against 80 `vmul.f` + 64 `vadd.f`
  (y's fp32 rescale, emulated) -- the rescale is most of the work.
* **Rescale skip (in, 2026-11-04):** the Q·K core writes exactly 1.0 as the correction when a row's max did
  not move (the LUT exp(0) is not exactly 1), and `attn_fv` skips the rescale when the correction of every
  *real* head (first `Q_HEADS_PER_CU` rows) is 1.0, compared as bits. The first try checked all 8 rows and
  never fired: Hy-MT2 has 4 real heads per CU, the 4 padding rows keep c = 0 (never updated, never emitted).
  Bench (zero KV, skip always taken): 12K 37.5 -> 33.6-35.0, 24K 57.1 -> 50.0-51.5 ms. Real text (test_long,
  16 steps): decode 8K 32.4 -> 33.2, 16K 22.7 -> 24.1, 24K 17.5 -> 18.8 tok/s. Translations identical.
  Decode logits change slightly (cos >= 0.99957 vs before; the LUT exp(0) is not exactly 1). Against the
  all-CPU reference the two builds are equally close (decode cos mean 8K 0.98856 / 0.98856, 16K 0.99014 /
  0.99015, 24K 0.98279 / 0.98258): accuracy-neutral.
* 1K in / 22K out (test_long, real text): decode 810.9 -> **744.2 s** (27.1 -> 29.6 tok/s), -8.2%.
* **bfp16 KV feasibility (2026-11-04):** a one-core Peano test computed a 16-key chunk with today's emulated
  `aie::mmul<8,8,8,bf16>` and with native `mac_8x8_8x8T_conf` on host-packed K rows / V^T tiles (Qwen3.5-2B's
  `_bfp16_tile_group` / `_bfp16_vt_tile_group`): 200 chunks, 5 value classes (normal, wide exponent spread,
  1e-30, 1e30, zeros), Q·K^T 0 / 25,600 and P·V 0 / 204,800 values differ -> bit-identical. Test kept in
  /tmp/bfp16_kv_test (not in the repo).
* Q·K cores have a 1 KB stack (built at -O1): an early-return form of the change overflowed it (1344 B);
  the `?:` form fits.

## bfp16 V (2026-11-04, design B step 3)

The P·V cores read the finished 16-key chunks of V as host-packed bfp16 V^T tiles instead of
bf16. Same results bit for bit, V is 44% smaller to read, and the per-mmul transpose + bf16 ->
bfp16 conversion of V is gone (P is converted twice per chunk instead of 32 times).

* **Data path.** Main V: shim 3/4 channel 1 -> mem tile 3/4 S2MM1 (4,608 B = 2 heads per chunk) ->
  MM2S2/3, plain linear 2,304 B per head (no reordering) -> P·V core S2MM1 (2,304 B ping/pong).
  Newest chunk (always bf16; the array writes its newest row in the same run): shim 5 MM2S0/1 ->
  mem tile 3/4 S2MM2 (8 KB) -> MM2S4/5 with the old V reordering -> P·V core S2MM0 (4 KB
  `v_tail`, locks 6/7). All of these ports were free.
* **Kernel** (`attn_kv.cc`): `L[1]` = number of bfp16 chunks n. Chunks 0..n-1 go through
  `attn_fv_bfp16` (native `mac_8x8_8x8T_conf` on `to_v64bfp16ebs8(P)` and the packed tiles, read
  in order with one `block_vector_input_buffer_stream`), then one chunk through the old `attn_fv`
  from `v_tail`. The S buffers now flip n + 1 times per call and the V buffers n times, so they have
  separate ping/pong flags. The rescale skip works the same in both paths. Q·K core unchanged.
* **Engine:** `FLM_DLL/detail/hunyuan_npu/README.md`, "bfp16 KV copy". `RTP_L_attn_kv` moved
  33792 -> 50176 (all other RTP addresses unchanged).
* **Stack check.** The old check added up every frame in the ELF (5,888 B for the P·V cores, over
  their 4 KB). The deepest real chain is 3,456 B (main -> attn_kv -> the bf16 `attn_fv_impl`), so
  `check_peano_stack.sh` now gets the deepest chain from `makefiles/stack_depth.py` (same parser as
  `../prefill_overlay/tools/check_stack.py`; an indirect call falls back to the sum).
* **Checked (Krackan):** logits of 8K / 16K / 24K prompts + 16 decode steps each: 51 / 51
  bit-identical with the rescale-skip build. `HY_BFP16_CHECK`: 0 bytes differ. Translations: 130 / 130
  identical text, 0 empty.
* **Speed.** Bench (one runlist per token): 4K 22.3, 12K 33.1, 24K 48.6 ms -- the same as the old
  `STUB=fv` build with no P·V at all (22.1 / 32.5 / 48.2). Real text (test_long, 16 steps): 8K 33.2
  -> 35.1, 16K 24.1 -> 25.7, 24K 18.8 -> 20.2 tok/s. 1K in / 22K out (test_long, real text): decode
  744.2 -> **728.9 s** (29.6 -> 30.2 tok/s, -2.1%); the 1K prefill 763 -> 645 ms (only V is packed now).
  Small, because the average context of that run is ~12K, where the bench gains only ~1-2 ms.
* **What limited decode after this: the K stream.** A `STUB=attn` build of this layout (no attention
  math at all) was as fast as the real one (24K 48.4 vs 48.5 ms), so the time was data movement. The
  K channel (one per column, K of 2 heads, still bf16) carried 12.6 MB per layer at 24K = ~12.8 GB/s,
  near one shim channel's ~14 GB/s; per Q·K core that is ~6.4 GB/s. V's channel had room. Fix: next section.

## bfp16 K (2026-11-04, design B step 4)

K goes the same way: finished chunks as host-packed bfp16 (blocks along dh, the layout
`_bfp16_tile_group` writes), the newest chunk bf16. Same results bit for bit, K 44% smaller.

* **Data path.** Main K: shim 3/4 channel 0 -> mem tile S2MM0 (4,608 B) -> MM2S0/1 linear 2,304 B per
  head -> Q·K core S2MM1 (2,304 B ping/pong). Tail: shim 5 now sends, per column, the K tail rows then
  the V tail rows (two BDs on the same channel) into one 16 KB mem tile buffer; MM2S4/5 send [K tail
  with the old K reordering][V tail with the old V reordering] = 8 KB into the P·V core's `v_tail`.
* **The Q·K core has no free input**, so it reads its K tail from the P·V core's `v_tail` (the tile
  above, the same way it already writes the shared S buffers). The tail lock on the P·V tile counts 2:
  the DMA waits for 2 and releases 2; each core takes 1 and gives 1 back (`_up_lock_*` from the Q·K core).
* **Mem tile BDs:** a BD that acquires a lock must also release one (aiecc asserts otherwise), so the
  K-tail BD and V-tail BD of each output are linked by one extra lock (9 / 10).
* **Q·K kernel** (`attn_qk.cc`): `L[1]` = n. One compute function on packed tiles (native
  `mac_8x8_8x8T_conf`, Q converted per dh block as the emulation does). The bf16 tail is first packed
  on the core (`pack_k_tail`: `to_vector<bfp16ebs8>` of each 8 x 8 tile as loaded = the emulation's
  `to_bfp16(transpose(transpose(K)))`), then goes through the same function. `qk_chunk` / `pack_k_tail` are
  `noinline` so the code exists once: stack 960 / 1,024 B, code ~2.9 KB. K and S flip ping/pong separately.
* RTP: `RTP_L_attn_qk` moved 5120 -> 3328 (smaller K buffers); `RTP_L_attn_kv` stays 50176.
* **Checked (Krackan):** 51 / 51 logits bit-identical with the rescale-skip build (8K / 16K / 24K + 16
  steps); `HY_BFP16_CHECK` 0 bytes differ (K and V); translations identical, 0 empty.
* **Speed.** Bench (one runlist): 4K 22.3 -> 20.2, 12K 32.8 -> 26.3, 24K 48.5 -> **35.8 ms**. Real text
  (16 steps): 8K 35.1 -> 41.0, 16K 25.7 -> 32.6, 24K 20.2 -> 26.9 tok/s. 1K in / 22K out (test_long, real text):
  decode 728.9 -> **596.9 s** (30.2 -> 36.9 tok/s, -18%; -26% against the 810.9 s before the rescale skip);
  the 1K prefill 645 -> 658 ms (K is packed again).
* Left to try: fewer y round trips per key (2 chunks per call), more P·V cores per KV head, Q converted
  to bfp16 once per call instead of once per chunk, more shim channels for the KV read.

## Q4_K body + Q8_0 lm_head (2026-11-12, step C3)

The model now ships a Q4_K body (4,736 B per 32 x 256 chunk) and a Q8_0 lm_head
(`/path/to/models/Hy-MT2-1.8B-NPU2/model.q4nx`). This folder reads both.

* **Body.** `models/hunyuan-dense-1.8b.h`: `QUANT_TYPE __Q4_K__`, and `iron/models/hunyuan.py`:
  `QuantType.Q4_K` (sets `W_BLOCK_SIZE`, so all weight buffers grow from 4,608 to 4,736 B). The Q4_K
  kernel was already in `kernels/q4_core.h`; this is its first run on hardware. The core buffers moved
  but every RTP address stayed the same (`RTP_PROJ_IS_ATTN_BUFFER_*` still 33280).
* **lm_head.** A Q8_0 32 x 256 chunk is 8,704 B and does not fit one 4,736 B weight buffer, so the
  engine splits it into two 32 x 128 halves (`q80_half_block_t`: 128 bf16 scales + 4,096 int8, 4,352 B)
  and pads each to 4,736 B. In `proj_main.cc`, `proj_lm_head` takes one x buffer (256 columns) and two
  weight buffers (`_Q80_half_x_bf16` in `kernels/q8_core.h`, copied from `MiniCPM_V/decoding`). The
  DMAs did not change: the mem tile loops over its buffers, and the shim just sends more bytes.
* **Half order (engine).** Each mem tile buffer is 2 x 4,736 B: the first half goes to one core of a
  pair, the second half to the other. So a band pair must be stored `[b0 h0][b1 h0][b0 h1][b1 h1]`, not
  `[b0 h0][b0 h1][b1 h0][b1 h1]`. `hunyuan_desc::reorder_cpy_lm_head` now runs the row-band
  interleave over the half blocks (half h of column block c = half block 2c + h), and the host reader
  (`cpu_functions.hpp`, `_dequant_mm(..., q80_pair)`) finds the halves the same way. Same layout as
  `minicpm_ocr_lm_head::reorder_cpy`.
* **`QMM_LIMB_PRECISION 1`** (model header). Q4_K has a min path that Q4_0 did not have. At 0 the group
  sums and the column sums enter the scale / min MACs as one bf16; at 1 as two bf16 parts. Measured
  below: slightly better, same speed (decode is memory bound).
* **Build note.** `models/*.h` is not a make dependency of the kernels. After changing it, delete
  `build/HUNYUAN_DENSE_1_8B/kernels/proj_main.o` (or the whole build folder), or nothing rebuilds.

Projection core (e.g. `core_0_2.elf`, Peano 22): `.text` 3,024 -> 6,144 B of 16,384; deepest stack chain
640 -> 1,920 B; `.bss` 128 -> 896 B (`b_col_reduce_add_buf` is live again for the Q4_K min path).

### Checks (`./bench_decode.exe --check`)

`--check` runs, before the timing:

1. the lm_head (final norm + Q8_0 matvec) on a pseudo-random x, against the engine's CPU reader;
2. one decode layer (layers 0, 16, 31) at context 1, against the engine's CPU functions. With one key
   the softmax is exactly 1, so each Q head's attention output is its KV head's V row and no CPU
   attention is needed. K and V are read back from the cache rows the array wrote (K after the QK norm;
   the bench's RoPE table is cos 1, sin 0). Q is not checked (it drops out); it shares the K/V move and
   kernel. `out - in` is the layer's own update (the input is small, |x| <= 0.01, so the residual does
   not hide it).

Cosine NPU vs CPU (Krackan, 2026-11-12):

| | lm_head | V (L0 / L16 / L31) | K | layer out - in |
|---|---|---|---|---|
| `QMM_LIMB_PRECISION 0` | 0.999991, top-1 same | 0.99995 / 0.99991 / 0.99997 | 0.99990 - 0.99993 | 0.99797 / 0.99855 / 0.99715 |
| `QMM_LIMB_PRECISION 1` (in) | 0.999991, top-1 same | 0.99998 / 0.99999 / 0.99999 | 0.99996 - 0.99997 | 0.99815 / 0.99893 / 0.99776 |

A wrong layout or a misread weight gives a cosine near 0, so the formats are right. The layer update
is ~0.998 even though every projection alone is ~0.9999. The likely cause is the parts this step did
not touch (bf16 SiLU and RMS norm on the array, attention math emulated through bfp16), but this was not
measured against a Q4_0 build (the Q4_0 weights are gone).

### Speed (`bench_decode --reps 20`, ms per token, Krackan)

| ctx | 8 | 1K | 4K |
|---|---|---|---|
| 32 layers, Q4_0 ("Krackan baseline", ../README.md) | 14.91 | 15.70 | 17.96 |
| 32 layers, Q4_K | 15.09 | 15.85 | 18.19 |
| one runlist incl. lm_head, Q4_0 | 17.10 | 17.86 | 20.12 |
| one runlist incl. lm_head, Q4_K + Q8_0 | 19.37 | 20.22 | 22.53 |

Weights per layer 27.13 -> 27.89 MB (+2.8%); the body is +1.0% to +1.3% slower. The lm_head reads 286 MB
(was 139 MB; the Q8_0 tensor is 263 MB, the padding of the halves adds 23 MB) in 4.28 ms at 66.9 GB/s
(was ~2.2 ms). So the lm_head is now ~2.1 ms of the ~2.3 ms extra per token.

Not done in this step: the end-to-end logits test (`test_long` against the host dumps). With XRT 2.27
the dev-build harness fails at start: `Invalid path '../../..//xclbins/Hy-MT2-1.8B-NPU2/layer.xclbin':
path traversal not allowed` (its `exec_path` is `../../../`).

---

# Generic Decoding Layer — Overlay Guide

A single AIE overlay (`layer.xclbin`) that executes **one full transformer decoder
layer** (and, in a second mode, the **LM head**) for batch-1 / one-token decoding on
AMD Ryzen AI **Strix Point (NPU2, `npu2` device, 8 cols × 4 core rows)**.

The same bitstream is re-invoked once per hidden layer; all weights are streamed in
from DDR every invocation, so the array holds no persistent per-layer state other
than the residual accumulator on the RMS tile.

This document describes the **kernel** (`kernels/`) and **IRON/MLIR** (`iron/`)
halves. The `host/` directory is a standalone test harness and may be out of date —
treat it as a reference for the DDR-side data layout only. The production host lives
in `FLM_DLL/`.

---

## 1. Repository layout

```
generic_decoding_layer/
├── Makefile              # top level; MODEL_TYPE selects the build
├── makefiles/
│   ├── common.mk         # tool paths + compiler flag sets
│   ├── kernel.mk         # kernels/*.cc -> build/$(MODEL_TYPE)/kernels/*.o
│   ├── bitstream.mk      # iron/*_aie.py -> mlir -> xclbin  (+ rtp_addresses.txt)
│   └── host.mk           # test host only
├── kernels/              # AIE core code (C++ / aie_api), compiled per model
│   ├── model_spec.h      # derives kernel-side constants from models/*.h
│   ├── aie_array_layout.h# array-level constants (repeats, packet ids)
│   ├── aie_kernel_utils.h# lock helpers, compiler pragmas, small vector utils
│   ├── qmm_core.h        # picks the MVM kernel for the model's QUANT_TYPE
│   ├── q4_core.h         # Q4_1 / Q4_K MVM kernels (one API)
│   ├── q8_core.h         # Q8_0 MVM kernel (same API) + the lm_head's Q8_0 half blocks
│   ├── bf16_limb_math.hpp# two/three-bf16-limb emulation of fp32 MACs
│   ├── typedef.hpp  lut_based_ops.h
│   ├── proj_main.cc      # quantized MVM engine (16 cores)
│   ├── rms_residual.cc   # RMSNorm + residual + LM-head driver
│   ├── rope.cc / rope_phi4.cc
│   ├── glu.cc            # SwiGLU / GeGLU
│   ├── attn_qk.cc        # Q·Kᵀ + online-softmax max/correction
│   └── attn_kv.cc        # P·V + running denominator + normalize
├── models/               # KERNEL-side model headers (C preprocessor)
│   └── all_models.h      # model id defines + includes every model header
├── iron/                 # array construction (tiles, buffers, locks, DMA, routes)
│   ├── layer_aie.py      # TOP LEVEL: floorplan + all inter-IP routing
│   ├── model_config.py   # IRON-side ModelConfig base + registry
│   ├── models/*.py       # IRON-side per-model configs
│   ├── constants.py      # ModelArch enum + packet id enums
│   ├── utils.py          # tile/port/context helpers, AIE_CT_IP / AIE_MT_IP bases
│   ├── proj_tile.py      # q4_proj_engine (16 CTs + 4 MTs + 4 shims)
│   ├── attn.py           # attn_column / attn_cu / qk_tile / kv_tile / attn_memtile
│   ├── rms_residual.py  RoPE.py  glu.py  aux_memtile.py
├── references/ref_gen.py # golden-reference generator (HF/GGUF -> safetensors)
├── visualize.py          # ASCII render of the routed switchbox graph
└── build/$(MODEL_TYPE)/{kernels,mlir,bitstream,xclbins}
```

**A model is defined twice, independently**: once in `models/<model>.h` for the
kernels and once in `iron/models/<model>.py` for the array. Both must agree. See
§8.

---

## 2. Build

```bash
source ../../setup.sh                  # FLM_Xclbin/setup.sh; set MLIR_AIE_DIR, PEANO_INSTALL_DIR first
make MODEL_TYPE=QWEN3_0_6B -j8         # kernels + mlir + xclbin
```

Artifacts:

| Path | Contents |
|---|---|
| `build/$(MODEL_TYPE)/kernels/*.o` | per-core objects (`.o.lst` = chess listing) |
| `build/$(MODEL_TYPE)/mlir/layer.mlir` | generated by `python3 iron/layer_aie.py $(MODEL_TYPE)` |
| `build/$(MODEL_TYPE)/xclbins/layer.xclbin` | the overlay |
| `build/$(MODEL_TYPE)/xclbins/rtp_addresses.txt` | `RTP_<name> <byte address>` — the host needs this to poke run-time params |

Useful targets: `make kernel`, `make link` (MLIR only), `make bitstream`,
`make route` (renders the routing with `visualize.py`), `make clean_all`.

### Build notes / gotchas

- Peano is the default. A Chess build (`ENABLE_CHESSCC=1`) needs Vitis AIE Essentials:
  set `AIETOOLS_ROOT` (and `LM_LICENSE_FILE`), or put `xchesscc` on PATH.
- `ENABLE_CHESSCC=1` uses `xchesscc` for both kernels and `aiecc.py`.
  `ENABLE_CHESSCC=0` (the default) uses Peano/llvm-aie, which needs two
  workarounds already encoded in the tree:
  - `kernel.mk` forces `-O1` on `attn_qk.o` / `attn_kv.o` (at `-O2` Peano
    miscompiles the context-length `do/while`, deadlocking from ctx ≥ 128).
  - `AIE_LOCK_LOOP_CTR` (`aie_kernel_utils.h`) is `volatile int` under Peano so
    the per-iteration lock release is not sunk past the loop back-edge.
  Any new loop whose body ends in a lock release must use `AIE_LOCK_LOOP_CTR`.
- `MODEL_TYPE` is passed to kernels as `-DMODEL_TYPE=<name>` and to IRON as
  `argv[1]`; the string must exist in **both** `models/all_models.h` and
  `iron/constants.py::ModelArch`.
- `FLM_VECTOR_BITS` (default `256`) caps the width of the *elementwise* vector
  ops — load, store, `mul`, `mac`, `mac_square` — that the helpers in
  `kernels/aie_kernel_utils.h` emit, and the width the RMS-norm / residual /
  RoPE loops step in. It reaches the kernels as `-DFLM_VECTOR_BITS=<n>` and
  drives `AIE_BF16_LANES` / `AIE_BYTE_LANES` / `AIE_F32_LANES`; every loop picks
  its own lane count through `_fit_lanes<>`, so a width that does not divide the
  extent narrows instead of failing to compile.

  `512` restores the full-register forms this tree used to emit. It is not the
  default because the 512-bit ops are avoided here, and under Peano it does not
  currently build at all: `aie::neg` on a 32-lane bf16 vector hits an
  unlegalized `G_FNEG <32 x s16>` in the llvm-aie backend, crashing the compiler
  on `rope.cc` / `rope_hunyuan.cc`. `aie::mmul` operand widths are set by the
  matmul `r/s/t` shape and are deliberately *not* covered by this flag.

---

## 3. Floorplan

Rows: 0 = shim (`IT`), 1 = memtile (`MT`), 2–5 = compute (`CT`).
In `iron/`, `CT[r][c]` uses `r ∈ 0..3` → physical row `r+2`.

```
col:      0        1        2         3         4         5        6        7
       ┌────────┬────────┬─────────┬─────────┬─────────┬────────┬────────┬────────┐
row 5  │ proj   │ proj   │  (free) │ attn_kv │ attn_kv │ (free) │ proj   │ proj   │
row 4  │ proj   │ proj   │  (free) │ attn_qk │ attn_qk │ (free) │ proj   │ proj   │
row 3  │ proj   │ proj   │  RoPE   │ attn_kv │ attn_kv │ (free) │ proj   │ proj   │
row 2  │ proj   │ proj   │  RMS    │ attn_qk │ attn_qk │  GLU   │ proj   │ proj   │
       ├────────┼────────┼─────────┼─────────┼─────────┼────────┼────────┼────────┤
row 1  │assemble│ MAIN   │  (free) │ attn KV │ attn KV │  aux   │assemble│ weight │
  (MT) │  col   │ mt     │         │  mem    │  mem    │  mt    │  col   │  only  │
       ├────────┼────────┼─────────┼─────────┼─────────┼────────┼────────┼────────┤
row 0  │ W str  │ W str  │ x in /  │ K cache │ V cache │ (free) │ W str  │ W str  │
  (IT) │        │        │ x out   │ logits  │         │        │        │        │
       └────────┴────────┴─────────┴─────────┴─────────┴────────┴────────┴────────┘
```

Note the attention rows: `attn_qk` sits at rows 2 and 4, `attn_kv` at rows 3 and 5
(`attn_column` is constructed with `attn_qk_tile=[CT[0], CT[2]]`,
`attn_kv_tile=[CT[1], CT[3]]`). Each (qk, kv) pair is one **attention CU**; there
are 4 CUs, 2 per column, in columns 3 and 4.

Free tiles (col 2 rows 4–5, col 5 rows 3–5, MT[2], IT[5]) are available for new
features.

### Shim (DDR) interface

| Shim | MM2S 0 | MM2S 1 | S2MM 0 | S2MM 1 |
|---|---|---|---|---|
| IT0/IT1/IT6/IT7 | weight stream A | weight stream B | — | — |
| IT2 | hidden state `x` **and** RoPE/QK-norm weights (packet-switched) | RMS weights | — | layer output `x` |
| IT3 | K cache replay | V cache replay | **K** written back from RoPE | logits (LM-head mode) |
| IT4 | K cache replay | V cache replay | **V** written back from RoPE | — |

---

## 4. Dataflow of one layer

`layer_aie.py` wires this; `proj_main.cc` and `rms_residual.cc` drive the phases.

```
 IT2 ──x──▶ RMS ──(pkt rms_to_proj)──▶ proj engine ─┐
 IT2 ──w──▶ RMS                                     │
                                                    ├─ phase 1 QKV  ─(pkt to_rope)──▶ RoPE
 IT2 ──rope/qk-norm w──▶ RoPE                       │
                                                    │      RoPE ─Q──▶ aux MT ─4×─▶ attn CU 0..3
                                                    │      RoPE ─K──▶ IT3 (DDR kv cache)
                                                    │      RoPE ─V──▶ IT4 (DDR kv cache)
                                                    │
                                     IT3/IT4 ─K,V─▶ attn MT ─▶ attn CU 0..3 ─O──▶ aux MT
                                                    │
                                     aux MT ─O──(pkt to_out_proj)─▶ proj engine
                                                    ├─ phase 2 O    ─(pkt to_rms_norm)─▶ RMS  (residual add + 2nd norm)
                                                    │
                                        RMS ────────┤
                                                    ├─ phase 3 up/gate ─(pkt to_glu)──▶ GLU ─▶ aux MT
                                                    │
                                     aux MT ─h──(pkt to_down_proj)─▶ proj engine
                                                    └─ phase 4 down ─(pkt to_rms_norm)─▶ RMS ─(pkt rms_to_st)─▶ IT2
```

### The four projection phases (`proj_main.cc`)

`IS_ATTN[0] == 1` (decoder-layer mode):

| Phase | GEMV | rows / 512 | destination packet id |
|---|---|---|---|
| 1 | `x(D) → QKV(DQ+DK+DV)` | `QKV_REPEATS` | `pkt_id_to_rope` (1) |
| 2 | `attn_out(DQ) → D` | `MODEL_DIM/512` | `pkt_id_to_rms_norm` (4) |
| 3 | `x(D) → up‖gate(2·I)` | `UP_GATE_REPEATS` | `pkt_id_to_glu` (8) |
| 4 | `h(I) → D` | `MODEL_DIM/512` | `pkt_id_to_rms_norm` (4) |

`IS_ATTN[0] == 0` (LM-head mode): a single `x(D) → VOCAB_SIZE_PADDED` GEMV, and
`rms_residual` switches to the `y_out_0/y_out_1` path that streams logits out via
`IT3.S2MM[1]`.

### Packet IDs (`iron/constants.py` ↔ `kernels/aie_array_layout.h`)

These two files define the **same numbers twice** and must stay in sync.

| Stream | id |
|---|---|
| `proj → rope` | 1 |
| `proj → rms_norm` | 4 |
| `proj → glu` | 8 |
| `rms → proj` | 0 |
| `rms → shim` | 1 |
| `IT2 → rms` / `IT2 → rope` | 0 / 1 |
| `rope K → IT3` / `rope V → IT4` | 14 / 15 |
| `aux → down_proj` / `aux → out_proj` | 0 / 2 |

Routing is a mix of circuit-switched `aie.flow` and packet-switched
`aie.packetflow`. The projection engine's output is packet-switched all the way
from the compute tile to the final consumer: the **packet header is written by the
kernel** into `y_buffer[14..15]` (a `uint32` alias) each round, and the memtiles
forward it with `keep_pkt_header=True` on the first contributor of each assembly
stage. This is how one physical output stream fans to RoPE / RMS / GLU without
reconfiguring routes.

> In `layer_aie.py`, **the order of `connect_port` calls matters** — the router
> gives earlier (longer) connections priority. Reordering can turn a routable
> design into an unroutable one.

---

## 5. The projection engine (`iron/proj_tile.py`, `kernels/proj_main.cc`)

16 compute tiles (cols 0, 1, 6, 7 × rows 2–5) run a Q4-NX quantized GEMV.

- **Weight format**: a `32×256` block of one of three formats (two 4-bit, one
  8-bit), selected at compile time by the model's own `QUANT_TYPE` — see §5.1.
  The same tiles serve the LM head; in this copy the LM head is Q8_0 half blocks
  (see "Q4_K body + Q8_0 lm_head" at the top).
- **Core pairing (`proj_combo`)**: two vertically adjacent CTs form one MVM unit.
  The lower tile (`send_x_out=True`) owns the output buffer and the output DMA;
  the upper tile writes its 32 rows into the lower tile's buffer at offset `16+m`
  through the `_down_lock_*` helpers. Result: 64 contiguous output rows per pair.
- **Output round**: 16 cores × `m=32` rows = **`M_PER_ROUND = 512`** elements per
  round, assembled memtile-side as `2` header + `512` payload:
  `CT (2m+2) → assemble_col MT (8m+2) → main MT (16m+2) → consumer`.
- **`x` broadcast**: `main_mt` (MT[1]) receives `x` in 512-element chunks on
  `S2MM[3]` and broadcasts on `MM2S[4]` to all 16 CTs, which consume it as two
  256-element (`x_slice_size`) ping/pong buffers.
- **Weight fan-out**: each of the 4 weight memtiles takes 2 shim streams of
  `2·W_BLOCK_SIZE` and issues 4 streams of `W_BLOCK_SIZE`, one per CT in its
  column group.
- Memtile roles: `MT[0]`/`MT[6]` = `proj_assemble_col` (y + weights),
  `MT[1]` = `proj_assemble_weight_x` (**main**: y + weights + x broadcast),
  `MT[7]` = `proj_assemble_weight_only`.

### 5.1 Quantized weight formats (`kernels/qmm_core.h`, `kernels/model_spec.h`)

Every model header picks a format with `QUANT_TYPE` and the packing stride it was
converted with, `PARALLEL_ROWS` (§5.2). All three formats are implemented and build
clean (Q4_0 was removed from this copy on 2026-11-12, when Hy-MT2 moved to Q4_K); `__Q4_1__` is what the layer has shipped from the start and is the fallback
if a header omits `QUANT_TYPE`.

| `QUANT_TYPE` | struct | dequant | per 32-group | per 256-super-block | bpw | block bytes | `PARALLEL_ROWS` |
|---|---|---|---|---|---|---|---|
| `__Q4_1__` | `q41_block_t` | `w = s·q + m`, `q` unsigned | bf16 `s`, bf16 `m` | — | 5 | 5120 | 16 or 32 |
| `__Q4_K__` | `q4k_block_t` | `w = S'·s'·q − M'·m'` | uint8 `s'`, uint8 `m'` | bf16 `S'`, bf16 `M'` | 4.625 | 4736 | 32 only |
| `__Q8_0__` | `q80_block_t` | `w = s·q`, `q` signed int8 | bf16 `s` | — | 8.5 | 8704 | 32 only |

`QUANT_BLOCK` is the selected struct, `QUANT_BLOCK_BYTES` its size, and
`QUANT_HAS_MIN` says whether there is a min path at all (`0` for Q8_0 —
it is symmetric, so `proj_main.cc` skips the `Σx` reduction entirely).
`static_assert`s in `model_spec.h` pin all three sizes, because a mismatch against
the DMA descriptor is a silently misread weight rather than a fault.

`proj_main.cc` does not branch on the format: it includes `qmm_core.h`, which
selects `q4_core.h` (Q4_1 / Q4_K) or `q8_core.h` (Q8_0). Every kernel
exposes the same four calls, and what carries the running row across the K loop is
the format's business:

```
qmm_accum_t c_accum;                                  // empty for Q4_1
_QMM_zero(c_accum, y_acc);                            // per output row block
_QMM_x_bf16<m, k>(c_accum, y_acc, x, w, b_reduce);    // per K-loop block
_QMM_finalize(y_acc, c_accum);                        // once the K loop ends
```

Q4_K keeps its scale path and min path in two registers for the whole K loop and
lets them meet once in `_QMM_finalize`; Q8_0 keeps its single row accumulator in a
register the same way; Q4_1 round-trips the float32 `y_acc` buffer, because at
stride 16 its two half-width row passes would otherwise both have to stay live
— that costs bandwidth but no bits, `y_acc` is float32 end to end.

### 5.2 `PARALLEL_ROWS` — the packing stride (`models/<model>.h`)

`PARALLEL_ROWS` is how many rows of a weight block one vector operand covers, and
it is exactly the row stride `FLM_Q4NX_Converter` packed the checkpoint with. It
is a **per-model** constant, declared in `models/<model>.h` next to `QUANT_TYPE`,
not derived from the format — a format does not by itself determine how a given
checkpoint was packed, and guessing wrong misreads every weight silently.

| value | packing | passes |
|---|---|---|
| `16` | parallel-strided re-order: a 32-row block is two 16-row planes (`_pack_q4nx`) | `ROW_PASSES = 2`, half width |
| `32` | plain column major over the whole 32-row span (`_pack_q4k`, `_pack_q8nx`) | `ROW_PASSES = 1`, full width |

`aie_array_layout.h` derives `ROW_PASSES` from it and asserts that Q4_K and Q8_0
get `PARALLEL_ROWS == Q4NX_ROW_BLOCK_SIZE`, since the converter only emits those
plain column major. `q4_core.h` derives `COLS_PER_LOAD`, `LOADS_PER_GROUP` and
`ACC_CHAINS` from it (`ACC_CHAINS = 2` only below full width, where one MAC chain
would serialise on its own accumulator), so the shared `_group_reduce()` covers
both strides; `q8_core.h` does the same for 64-int8 loads. Omitting
`PARALLEL_ROWS` from a new model header is a compile error, by design.

### 5.3 Precision: `QMM_LIMB_PRECISION` (`kernels/qmm_core.h`)

All four kernels **reduce first and scale after**: a whole 32-value group is
reduced against the bf16 activations in one float32 accumulator, and only then
meets the group scale. That makes every multiplier operand bf16-exact — `q` is a
4- or 8-bit integer, Q4_K's `s'`/`m'` are uint8 (bf16 holds every integer up to
256), and every bf16 scale is *stored* bf16 — so a rounding can only ever land on
an accumulator. There is exactly one such rounding left: the group sum on its way
into the scale MAC.

`QMM_LIMB_PRECISION` decides what happens there:

| | group sum | min path (`Σx`) | inner loop, QWEN3_0_6B Q4_1 |
|---|---|---|---|
| `0` (default) | rounded to bf16, 8 mantissa bits, ~2e-03 relative | high limb only, one MAC | **963 bundles/row pass** |
| `1` | `bf16_limb::mac_f32_bf16_low`, two bf16 limbs, ~17 bits | both limbs, two MACs | 1228 bundles/row pass (+28%) |

The default is `0` because the projections *are* the decode rate, and at `0` this
kernel schedules to the same 963 bundles and the same opcode mix as the
hand-written `q4_k.h` it replaced — verified by diffing the two `.o.lst` files. A
model header can `#define QMM_LIMB_PRECISION 1` before including anything if that
checkpoint needs the bits; `proj_main.cc` always hands the kernel `Σx` as two
interleaved bf16 limbs, so only the kernel side changes.

Why it is worth having at all: the bf16 cast costs ~2e-03 relative once per group,
correlated across the groups sharing a scale, and it does not average away over
the K loop — `|y|` ends up far smaller than the group sums that build it, so the
error is measured against the small number. `bf16_limb_math.hpp` also has
`mac_f32_bf16_safe` (3 MACs, fp32-exact) and the pre-split forms, with measured
accuracy and cost per form in its header comment.

**If you touch these kernels, check the schedule, not just the build.** The
inliner is the thing to watch: a helper the compiler leaves out of line turns the
group loop into a call per group and cost ~3x here before it was caught. Compare
`build/<MODEL>/kernels/proj_main.o.lst` against a known-good build — the bundle
count between `.begin_of_loop`/`.end_of_loop` in `_QMM_x_bf16` is the number that
matters, and an accumulator held by value or an `aie::add` on two accumulators
shows up in it as VMOVs.

---

## 6. Attention

### Shapes and the GQA padding scheme

Q heads are split across **4 CUs**. Because the `mmul` intrinsic works on fixed
`r×s×t` tiles, `Q_HEADS_PER_GROUP` (= `num_heads / num_kv_heads`) is padded up to
`GQA_SEGMENT_SIZE`; the pad rows are zero-filled by RoPE and dropped again when
`attn_kv` writes `O`.

`kernels/model_spec.h` selects one of three implementations from `ATTN_IMPL`:

| `ATTN_IMPL` | `mmul` r×s×t | `GQA_SEGMENT_SIZE` | typical use |
|---|---|---|---|
| `ATTN_IMPL_2x4x1` | 8×8×8, two blocks | 4 | 8 KV heads, 2–8 Q per group |
| `ATTN_IMPL_1x4x1` | 4×8×8, one block | 4 | 4 KV heads, ≤4 Q per group |
| `ATTN_IMPL_1x8x1` | 8×8×8, one block | 8 | 4 KV heads, 5–8 Q per group |

Derived (`model_spec.h`):
```
KV_HEADS_PER_CU        = NUM_KV_HEADS / 4
Q_HEADS_PER_GROUP_PADDED = ceil(Q_HEADS_PER_GROUP / GQA_SEGMENT_SIZE) * GQA_SEGMENT_SIZE
Q_HEADS_PADDED_PER_CU  = KV_HEADS_PER_CU * Q_HEADS_PER_GROUP_PADDED
```

### Q layout

`aux_memtile` (MT[5]) receives the padded Q vector (`PADDED_DQ`) from RoPE and
splits it into 4 slices of `q_offset = (NUM_Q_HEADS_PADDED/4) · DH`, each reordered
on the fly to the `mmul` A-operand layout:

```
dimensions = [(DH//8, 8), (heads_per_cu, DH), (8, 1)]   # -> (DH/8, heads, 8)
```

### K/V path

RoPE writes K to `IT3.S2MM[0]` and V to `IT4.S2MM[0]` (two BDs of `DV/2` each).
The host appends them to the DDR KV cache, then replays the whole cache back into
`IT3.MM2S[0,1]` and `IT4.MM2S[0,1]`. `attn_memtile` buffers `lk = 16` tokens ×
`(NUM_KV_HEADS/2)·DH` and splits it into two CU slices (`k_cu_offset`), reordering
K and V into their respective `mmul` operand layouts.

### Online softmax split across two cores

| | `attn_qk` (rows 2, 4) | `attn_kv` (rows 3, 5) |
|---|---|---|
| computes | `S = scale · Q·Kᵀ`, running max `m`, correction `c = exp(m_old − m_new)` | `l += Σ exp(S)` with rescale by `c`; `y += P·V`; final `y/l` |
| shared state | **locks** (`s_prod=4`, `s_cons=5`, `l_cons=7`) live on the **qk** tile | **buffers** `s_shared_buffer_{0,1}` live on the **kv** tile |

The kv core reaches the qk core's locks with `_down_lock_acquire/_release`
(lock-offset convention in `aie_kernel_utils.h`: `+0` = down, `+16` = left,
`+32` = up, `+48` = own). The qk core writes into its up-neighbour's memory.
**Anything that changes the qk/kv row assignment must change these offsets too.**

Context is walked in 16-token chunks (`lq = lk = 16`); the tail chunk is handled by
`aie::mask<16> mask = aie::le(idx, min(i,16))` in `attn_qk`, so `L` need not be a
multiple of 16 — but the DMAs always move a full 16-row tile.

The output DMA in `kv_tile` un-shuffles the accumulator back to head-major:
`dimensions = [(NQ_PER_CU, 8), (DH//8, NQ_PER_CU*8), (8, 1)]`, length
`NQ_PER_CU · DH` with `NQ_PER_CU = num_heads/4` (**unpadded** — the pad heads are
dropped here). `aux_memtile` concatenates the 4 CU outputs into `DQ` and forwards
it as the O-projection input.

---

## 7. Run-time parameters (RTP)

Buffers created with `ctx.buffer(..., use_write_rtp=True)` are named `RTP_*` and
their addresses are extracted into `build/$(MODEL_TYPE)/xclbins/rtp_addresses.txt`
by `bitstream.mk`:

| RTP buffer | on tile | meaning |
|---|---|---|
| `RTP_RMS_IS_ATTN_BUFFER` | CT(2,2) | `1` = decoder layer, `0` = LM head |
| `RTP_PROJ_IS_ATTN_BUFFER_<r>_<c>` | 16 proj CTs | same flag |
| `RTP_L_attn_qk_core_<r>_<c>` | 4 qk CTs | context length `L` |
| `RTP_L_attn_kv_core_<r>_<c>` | 4 kv CTs | context length `L` |

All RTP buffers are `L_ty = int32[8]`. The RMS and proj cores gate on
`rtp_available_lock` (**lock id 6**) at the top of every invocation, so the host
must, per layer: write the RTP value, then release lock 6 by writing to
`0x1F000 + 16·6` on that tile. The attention cores read `L[0]` directly without a
gate. See `host/decoding_layer.cpp::_gen_sequence` for the exact sequence.

### 7.1 `SELF_SEQUENCED_LM_HEAD` — opting out of the IS_ATTN handshake

A model header may define `SELF_SEQUENCED_LM_HEAD` together with
`constexpr int NUM_HIDDEN_LAYERS`. With the switch on, `rms_residual.cc` and
`proj_main.cc` stop asking *which* phase this is and count instead: one call is
one whole token —

```c
for (AIE_LOCK_LOOP_CTR layer = 0; layer < NUM_HIDDEN_LAYERS; layer++)
    <decoder layer>;
<lm head>;
```

The loop counter must be `AIE_LOCK_LOOP_CTR` (`kernels/aie_kernel_utils.h`): the
body ends in a lock release, and Peano will otherwise sink that release past the
back-edge and deadlock the array.

The two schemes are mutually exclusive at run time. With the switch on the host
must **not** write `RTP_RMS_IS_ATTN_BUFFER` / `RTP_PROJ_IS_ATTN_BUFFER_<r>_<c>`
and must **not** release lock 6 — nothing acquires it any more, so releasing it
only saturates the semaphore. `RTP_L_attn_*` is a different buffer on different
tiles and is unaffected; the context length still comes over RTP as before.

`int* IS_ATTN` stays in both kernel signatures, and `iron/rms_residual.py` /
`iron/proj_tile.py` still allocate the buffer, so RTP addresses do not move
between a self-sequenced and a non-self-sequenced build. The parameter is simply
unread. A header that defines the switch without `NUM_HIDDEN_LAYERS` fails on a
`static_assert` in `kernels/model_spec.h`.

Opt-in, per model: every model that does not define it keeps the RTP protocol
above unchanged. Currently only `HUNYUAN_DENSE_1_8B` uses it.

---

## 8. Adding / supporting a new model

A model is defined **three** times: the kernel header, the IRON config, and
whatever `FLM_Q4NX_Converter` config produced the `.q4nx` weights. Only one pair
is machine-checked — `ModelConfig.__init__` reads `QUANT_TYPE` back out of
`models/<model>.h` and refuses to build if it disagrees with `quant_type` (§8.3).
Everything else you check by eye. The order that wastes the least time:

1. Convert the checkpoint first (`FLM_Q4NX_Converter`) — it tells you the format
   and the packing stride the other two have to agree with.
2. `models/<model>.h` + `models/all_models.h` (§8.1), then `make kernel
   MODEL_TYPE=<M>` — this is where a shape or format mistake still fails loudly.
3. `iron/constants.py` + `iron/models/<model>.py` (§8.2), then `make
   MODEL_TYPE=<M>` for the xclbin.
4. Check the numbers in §8.3 agree by eye, and the constraints in §8.5 hold.
5. Host side (`FLM_DLL/`): see the `QUANT_TYPE` landmine in §9 before using
   anything other than Q4_1.

**Nothing about the weights is inferred — you declare it.** The overlay cannot
look at a `.q4nx` file, so every property of how that checkpoint was quantized is
a constant *you* write into `models/<model>.h`, and it is model-specific even
when two models share an architecture:

| you must specify | where | what it is |
|---|---|---|
| `QUANT_TYPE` | `models/<model>.h` | the format of the projection **and LM head** weights (§5.1) |
| `PARALLEL_ROWS` | `models/<model>.h` | the row stride the converter packed with (§5.2) |
| `Q4NX_ROW_BLOCK_SIZE` / `Q4NX_COL_BLOCK_SIZE` / `Q4NX_GROUP_SIZE` | `models/<model>.h` | the block and group geometry it packed with |
| `quant_type` | `iron/models/<model>.py` | the same format again, this time sizing the weight DMA (cross-checked against the header) |

The LM head streams through the same 16 proj tiles and the same `QUANT_BLOCK` as
q/k/v/o/up/gate/down, so **one `QUANT_TYPE` covers both**. Checkpoints that
quantize the output head differently from the body (a common convert-time choice)
cannot be expressed today — convert the head in the same format, or the overlay
will read it as if it were.

Get any of these wrong and the build still succeeds: the DMA hands the cores the
right number of bytes with the wrong meaning, and you get plausible-looking
garbage logits rather than a fault. The one exception is a `QUANT_TYPE` /
`quant_type` disagreement, which now fails the build. §8.3 and §8.5 are the rest,
to check by eye.

### 8.1 Kernel side

1. `models/<model>.h` — guard with `#if MODEL_TYPE == <MODEL_ID>`, define
   `MODEL_DIM`, `NUM_ATTN_HEADS`, `NUM_KV_HEADS`, `INTERMEDIATE_SIZE`,
   `GLU_SLICE`, `DH`, `VOCAB_SIZE`, `ATTN_SCALE`, `ATTN_IMPL`, `A_FUNC`,
   `Q4NX_{ROW_BLOCK,COL_BLOCK,GROUP}_SIZE`, `QUANT_TYPE` (§5.1) and
   `PARALLEL_ROWS` (§5.2), plus optional `HAS_QK_NORM`.
   Copy the nearest existing header and edit — that is the whole recipe, but
   re-derive the quantization constants from *your* checkpoint rather than
   inheriting the neighbour's; they are per-model, not per-architecture.
   Use `__<NAME>_H__` include guards.
   `QUANT_TYPE` falls back to `__Q4_1__` if omitted; a missing `PARALLEL_ROWS`
   is a compile error on purpose, there is no safe value to guess.
2. `models/all_models.h` — add `#define <MODEL_ID> <n>` and the `#include`.

### 8.2 IRON side

3. `iron/constants.py` — add to `ModelArch` **and** `ModelArchNames`. The string
   must exactly match the kernel `#define` name (it is the `MODEL_TYPE` you pass
   to `make`).
4. `iron/models/<model>.py` — subclass `ModelConfig` with
   `model_arch=ModelArch.<NAME>`; export it from `iron/models/__init__.py`
   (the subclass hook self-registers it, but the import must execute). Pass
   `quant_type=QuantType.<FORMAT>` matching the header's `QUANT_TYPE` — that is
   what sizes every weight DMA (`W_BLOCK_SIZE`). Every existing config states it
   explicitly rather than leaning on the Q4_1 default; a value that disagrees
   with the header raises at config construction, so this cannot silently drift.

### 8.3 The two definitions must agree

| kernel (`models/*.h`) | IRON (`iron/models/*.py`) |
|---|---|
| `MODEL_DIM` | `hidden_size` |
| `NUM_ATTN_HEADS` | `num_heads` |
| `NUM_KV_HEADS` | `num_kv_heads` |
| `INTERMEDIATE_SIZE` | `intermediate_size` |
| `DH` | `head_dim` |
| `GLU_SLICE` | `glu_slice` |
| `VOCAB_SIZE` | `vocab_size` |
| `HAS_QK_NORM` defined | `has_qk_norm=True` |
| `GQA_SEGMENT_SIZE` (from `ATTN_IMPL`) | `q_heads_padding` |
| `Q4NX_*` | `q4nx_*` |
| `QUANT_TYPE` | `quant_type` (sets `W_BLOCK_SIZE`) — **verified automatically** |

`QUANT_TYPE` is the only row here that is checked for you: `ModelConfig` parses
the `#if MODEL_TYPE == <M>` block of every `models/*.h` and compares. A header
that omits `QUANT_TYPE` is compared against the `__Q4_1__` fallback, and a model
with no kernel header at all is skipped.

`PARALLEL_ROWS` is kernel-side only — the IRON side never sees the packing
stride, only the block *size*, which `quant_type` gives it.

`ATTN_SCALE` and `A_FUNC` exist **only** on the kernel side; there is no IRON
counterpart, so a wrong `ATTN_SCALE` will build cleanly and produce wrong numbers.

### 8.4 Non-default kernels

If a model needs a variant kernel, add the `.cc` to `kernels/` (it is compiled
unconditionally by the wildcard rule) and point the config at it via the
`*_link_with` fields — e.g. Phi-4 uses partial-rotary RoPE:

```python
rope_link_with = "rope_phi4.o"
```

`RoPE.py` derives the external-func name from `link_with[:-2]`, so the object
basename **is** the symbol name (`rope_phi4.o` → `extern "C" void rope_phi4(...)`)
with the same signature as `rope`. The other overridable slots are
`rms_residual_link_with`, `glu_link_with`, `attn_qk_link_with`,
`attn_kv_link_with`.

### 8.5 Constraints a new model must satisfy

Checked by `ModelConfig.__init__`:
- `hidden_size`, `DQ`, `intermediate_size` all multiples of `x_slice_size = 256`.
- `quant_type` matches the kernel header's `QUANT_TYPE` (§8.3).

Not checked — violating these produces a silently wrong or hanging overlay:
- `hidden_size % 512 == 0` — `M_PER_ROUND`; also `aux_memtile.repeats = D/512`
  must be an integer.
- `(DQ + DK + DV) % 512 == 0` — `QKV_REPEATS`.
- `2 * intermediate_size % 512 == 0` — `UP_GATE_REPEATS`.
- `2 * intermediate_size % GLU_SLICE == 0` — GLU consumes fixed `GLU_SLICE`
  chunks. (Nanbeige pads `10496 → 10496+256` for exactly this reason.)
- `num_kv_heads % 4 == 0` — 4 attention CUs, and `attn_memtile` splits
  `num_kv_heads / 2` per column.
- `head_dim % 16 == 0`, and `head_dim/2 % 16 == 0` for the RoPE half-rotation.
- `head_dim` a multiple of 8 for the `mmul` K-dimension walk (`DH/8` columns).
- Sliding-window attention is fine; anything that is not a "traditional" decoder
  layer (cross-attention, MoE, per-layer KV sharing, non-GQA) is not supported by
  this overlay.
- `QUANT_TYPE` / `PARALLEL_ROWS` (kernel) must match `quant_type` (IRON) **and**
  how `FLM_Q4NX_Converter` actually packed the `.q4nx` file. Nothing checks this
  triple; every mismatch is silently wrong weights, not a fault.
- The LM head must be quantized in the same format as the projections — there is
  one `QUANT_TYPE` for all weights (§8).
- Q4_K and Q8_0 additionally require `PARALLEL_ROWS == Q4NX_ROW_BLOCK_SIZE` (this
  one *is* asserted, in `aie_array_layout.h`), and Q4_K requires
  `Q4NX_COL_BLOCK_SIZE == 256` — one block is exactly one super-block.

### 8.6 Known model table

| Model series | Head dim | Q heads | KV heads | Q/group | `ATTN_IMPL` | QK norm | Notes |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B / 3.1-8B | 64 / 128 | 32 | 8 | 4 | 2x4x1 | no | no padding |
| Llama-3.2-3B | 128 | 24 | 8 | 3 | 2x4x1 | no | 1 pad head per group |
| Qwen3-0.6B / 1.7B | 128 | 16 | 8 | 2 | 2x4x1 | yes | 2 pad heads per group |
| Qwen3-4B / 8B | 128 | 32 | 8 | 4 | 2x4x1 | yes | no padding |
| Phi-4-mini | 128 | 24 | 8 | 3 | 2x4x1 | no | partial-rotary `rope_phi4.o` |
| Nanbeige-3B | 128 | 20 | 4 | 5 | 1x8x1 | no | `q_heads_padding=8` |
| Gemma3-4B | 256 | 8 | 4 | 2 | 1x4x1 | yes | **IRON config only — no `models/gemma3-4b.h`, cannot build yet** |
| Hy-MT2-1.8B (`hunyuan-dense`) | 128 | 16 | 4 | 4 | 1x4x1 | yes | Q4_K body, Q8_0 lm_head halves / `PARALLEL_ROWS=32`; QK norm **after** RoPE → `rope_hunyuan.o`; `SELF_SEQUENCED_LM_HEAD`, `NUM_HIDDEN_LAYERS=32` (§7.1) |

---

## 9. Landmines

- **`PADDED_DQ` ignores `q_heads_padding`.** `ModelConfig.__init__` computes
  `PADDED_DQ` with a hard-coded `4`, while `RoPE.py` recomputes `DQ_PADDED` from
  `q_heads_padding`. They happen to agree for every model in the tree (Nanbeige:
  `ceil(5/4)·4 = ceil(5/8)·8 = 8`), but a model with e.g. 3 Q per group and
  `q_heads_padding=8` would get two different Q sizes and a silently corrupt
  layer. Fix `model_config.py` if you add such a model.
- **Duplicated constants.** Packet IDs live in both `iron/constants.py` and
  `kernels/aie_array_layout.h`. Lock IDs live in both the IRON IP class
  (`aie.lock(..., lock_id=N)`) and the kernel `constexpr int *_lock = N`. There is
  no cross-check; a mismatch deadlocks the array.
- **Lock neighbour offsets are hard-coded to the floorplan.** `proj_main.cc` uses
  `_down_lock_*` to reach its pair partner and `attn_kv.cc` uses `_down_lock_*` to
  reach the qk tile. Moving an IP to a different row breaks both.
- **Kernel buffer names encode positions that are off by one.** `kv_tile`
  allocates `s_shared_buffer_*_{row+1}_{col}` on the row-`row` tile. Cosmetic, but
  do not trust the name when reading `rtp_addresses.txt` or a listing.
- **`connect_port` ordering is load-bearing** for the router (see §4).
- **The infinite core loop keeps ping/pong parity across invocations.** Every core
  body is `for _ in range_(0xFFFFFFFF)` with `static bool is_*_ping` state, so
  buffer alternation is continuous across layers and across kernel launches. A
  phase whose slice count changes between launches will desynchronise the DMA BD
  chain from the kernel's ping/pong choice.
- **`QUANT_TYPE` is understood by the kernels and by IRON, but not further.**
  All three formats build clean, and `iron/model_config.py` now derives
  `W_BLOCK_SIZE` from `quant_type` (`iron/constants.py::QuantBytesPerWeight`), so
  the overlay's DMA lengths follow the format. What is still hardwired to Q4_1's
  5 bpw is the DDR side: `host/decoding_layer.cpp` has `* 5 / 8` baked into the
  weight offsets and `a_block_size`, and `FLM_Q4NX_Converter`'s `_pack_q4nx`
  returns an unpacked bf16 tensor when `m is None`, so it does not emit Q4_0
  blocks at all. A non-default `QUANT_TYPE` end to end means fixing those two
  first — and neither will fail loudly. `FLM_DLL/` is the production host; check
  its block-size arithmetic too.
- **`PARALLEL_ROWS` is a property of the checkpoint, not of the format.** It
  lives in `models/<model>.h` for that reason (§5.2). Q4_0/Q4_1 weights exist in
  both strides; only the converter run that produced the `.q4nx` file knows
  which, and reading it with the wrong one produces plausible-looking garbage.
- **The overlay has exactly one `QUANT_TYPE` for every weight it reads**, the
  seven per-layer projections *and* the LM head. A checkpoint whose head is
  quantized differently from its body does not fault — the block *count* is
  unchanged, only the block *size*, so LM-head mode reads misaligned bytes and
  returns garbage logits while decoder-layer mode looks fine. Check the head's
  packed block size in the `.q4nx` header, not just `default_tensor_type` in the
  converter config. (Hy-MT2-1.8B in this copy: `lm_head` Q8_0 over a Q4_K body,
  read by its own loop as padded half blocks, see "Q4_K body + Q8_0 lm_head".)
- **`VOCAB_SIZE_PADDED = ceil(VOCAB_SIZE / MODEL_DIM) · MODEL_DIM`** — the LM-head
  GEMV computes the padded width; the host must slice the valid logits out.

---

## 10. Debugging

- `make route MODEL_TYPE=<M>` writes `routes/route.json` and renders the
  switchbox usage with `visualize.py` — the fastest way to see why routing failed.
- `iron/*.py` emit `// …` comment lines with every derived shape into the
  generated MLIR; `head -60 build/<M>/mlir/layer.mlir` is a quick shape audit.
  `python3 iron/layer_aie.py <M> | head -60` gives the same without a build.
- `build/<M>/kernels/*.o.lst` is the chess schedule listing — use it to check
  whether a loop actually pipelined.
- A hang (`ERT_CMD_STATE_TIMEOUT`) is almost always a lock mismatch: an
  acquire/release count that differs between the IRON `init=` value and the kernel,
  or an RTP lock 6 that the host never released.
