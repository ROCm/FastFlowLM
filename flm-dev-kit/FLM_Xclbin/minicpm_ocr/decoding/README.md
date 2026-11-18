<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# FLM Decoding Layer generator

Only for generateing decoding layers on AMD Ryzen AI Strix+ chips.

## MiniCPM-OCR copy (2026-11-05)

Copied from `origin/main:FLM_Xclbin/Qwen3_5_2B/qwen3_5_decoding` (8688e6aa8): the 2B design with the lm_head
inside `layer.xclbin` (`PLAN_LM_HEAD_IN_LAYER.md`). MiniCPM-OCR's text tower is the Qwen3.5-0.8B shape, which
this generator already has (`models/qwen3.5-08b.h`, `MODEL_TYPE=QWEN3_5_08B`), so no new model type was added.

Changes from the 2B copy:

* `MODEL_TYPE ?= QWEN3_5_08B`; the host test defaults to `/path/to/models/MiniCPM-OCR-1.0-NPU2` and
  `build/QWEN3_5_08B/xclbins/layer.xclbin`.
* **`RTP_rms_mode` is at 34816, not 36864**: the RMS core's buffers are smaller at D 1024, so the linker puts it
  lower. The build check (`makefiles/bitstream.mk`) and the host (`rms_mode_addr`) use 34816; the engine must too.
  `RTP_proj_mode` (33280) and the attention RTPs (53248 / 53536) did not move.
* `run.exe --lm-head-in-layer --lm-head <lm_head.xclbin>`: also runs the separate `lm_head.xclbin` on the same final
  x and compares the logits (gate: cos >= 0.99999 and same top-1).

Vocab: `model.q4nx` has 248,128 lm_head rows (7,754 Q8_0 row blocks; 248,100 real tokens). The design rounds the
vocab up to 512-row rounds: 248,320 rows (485 rounds), the same as the 0.8B header. Rows past the checkpoint are
zero (`reorder_cpy_lm_head_in_layer` clears the buffer first).

Build and test (Peano, the CI toolchain; NPU runs under `flock /tmp/mco_npu.lock`):

```bash
source /tmp/mco_peano_env.sh      # same lines as ../README.md "Peano"
make bitstream                    # build/QWEN3_5_08B/xclbins/layer.xclbin, ~10 s
make host
./run.exe --check                 # every layer vs the fp32 CPU reference
./run.exe --lm-head-in-layer --lm-head <shipped lm_head.xclbin> [--len N]
./run.exe --xclbin <shipped layer.xclbin> --no-proj-mode --lm-head <shipped lm_head.xclbin> [--len N]   # old path
```

Results (Strix Halo test machine, NPU idle, 20 iterations, random context, seed 1234):

| check | result |
|---|---|
| new lm_head vs shipped `lm_head.xclbin`, same final x | **bit-identical** (cos 1, max abs 0) at 128 and 10K |
| new lm_head vs fp32 CPU | cos 0.999991 (10K), 0.999989 (128); same top-1 |
| next token's layers after an lm_head run | same state checksum |
| `--check` worst cos, new vs shipped layer.xclbin | seed 1234: 0.997996 vs 0.998525 (attn new K, layer 3); seed 7: 0.998449 vs 0.998505; other checkpoints equal to 3-4 digits. Not bit-identical (another build), equally close to the CPU |

| cached tokens | old: layers + switch + lm_head.xclbin + switch | new: one run list | layers alone old / new | lm_head alone old / new |
|---|---|---|---|---|
| 128 | 20.81 ms (switches ~2.90 + ~3.19) | **14.88 ms** | 9.83 / 9.67 | 4.89 / 5.30 (55 GB/s on 8 channels) |
| 10,240 | 27.46 ms | **20.04 ms** | 16.48 / 14.85 | 4.91 / 5.32 |


## bfp16 K and V in decode attention (WP6, 2026-11-06)

`make bitstream KV_BFP16=1` builds `build/QWEN3_5_08B_kvb/xclbins/layer.xclbin`, installed as
`layer_with_lm_head_kvb.xclbin` (engine: `FLM_DLL/detail/minicpm_ocr_npu/README.md`, "Decode attention on bfp16 K
and V"). The default build is unchanged: the same MLIR and, after the preprocessor, the same kernel sources.

**Why.** At long context the decode is bound by reading the KV cache: K and V each come through one shim channel
(shim 3, channels 0 / 1) at ~12.3 GB/s with the attention math on, ~13.5 GB/s with it removed (`PROBE_QK_SKIP` +
`PROBE_KV_SKIP`), which is one channel's measured ceiling (13.7 GB/s, `../../Qwen3_5_2B/dma_bandwidth`). bfp16 (8
values share one exponent byte: 9 bytes per 8 values) is 44% fewer bytes. The emulated bf16 mmul converts K and V to
exactly these blocks on every call anyway, so the result does not change.

**Data path.** Same streams and routes as before; only what flows on them changes. Every unit is 9,216 B:
* `L[1] = (L - 1) / 16` finished 16-key chunks, each one unit of K and one of V, made by the engine:
  `[KV head][DH / 8][2 key blocks][72 B]`. A K block is one key's 8 dh values, a V block one dh column's 8 keys (V^T):
  the blocks the emulation would make (`FLM_DLL/detail/minicpm_ocr_npu/bfp16_pack.hpp`).
* then the newest chunk, still bf16 (the array writes its newest row in the same run), as 2 units: one KV head each,
  18 cache rows of 256 values. The shim reads them after the new row is written. They start 2 rows **before** the
  chunk (when there are older rows), so a tail unit is 9,216 B like the others without reading past the cache; the
  cores skip those 2 rows.
* Mem tile 3 passes units through unchanged (no reordering), K/V buffers 9 KB instead of 16 KB.

**Kernels** (`#ifdef KV_BFP16` in `attn_qk.cc`, `attn_kv.cc`):
* Q.K: the native `mac_8x8_8x8T_conf` on the packed K, with Q converted per dh block exactly as the emulation does.
  The 2 tail units are packed on the core first (`pack_k_tail`: 8 rows x 8 dh -> `to_vector<bfp16ebs8>`), then go
  through the same code.
* P.V: the same with V^T; the tail is transposed per 8 x 8 tile and packed (`pack_vt_tail`). P is converted to bfp16
  once per chunk (the emulation did it per call, to the same blocks).
* P.V, y's update in two passes: (1) the rescale by c, or, when no row max moved, the emulated multiply by exactly
  1.0 the installed kernel does (`mul_by_one`), (2) y += P V. Peano does not overlap loop iterations, so pass (1)
  works on 4 independent 32-lane values per iteration to keep the vector unit busy. "No max moved" is one vector
  compare of the 8 corrections' bits with 1.0 instead of 8 soft-float compares (the core has no scalar FPU).
* The packed tail buffers (9 KB each) are IRON buffers passed as an extra argument: kernel statics only get a fixed
  12 KB region per core, and the P.V core's `y` already uses 8 KB of it.
* RTPs of this build: `RTP_L_attn_qk` **25600**, `RTP_L_attn_kv` **53248** (the build checks them; `RTP_proj_mode`
  33280 and `RTP_rms_mode` 34816 did not move). `L[2]` of the P.V core: rescale skip (below).
* Program memory: Q.K core 5.5 KB, P.V core 4.6 KB of 16 KB; deepest stack chain 704 B / 256 B of 2 KB (call tree
  followed as in `Hy_MT2_1.8B`'s `stack_depth.py`, run by hand). Packet-route check passes.

**Test host:** `./run.exe --xclbin <kvb xclbin> --kv-bfp16 ...` writes the bfp16 copy of the random cache into arg 3
with the engine's packer; `--dump FILE` writes the raw outputs (with `--check`: every attention layer's x; else the
final x) for a byte compare between two builds; `--rescale-skip` sets `L[2] = 1`.

**Rescale skip (`L[2]` of the P.V core).** The exp LUT gives exactly 1.0 when a row's max did not move (both table
entries are 0x3F80), so at long context nearly every chunk takes the "no max moved" path, and there the installed
kernel still runs the emulated multiply of y by exactly 1.0 (`mul_by_one`: 3 bf16 limbs, 2 subtractions, 3 multiplies,
2 adds per 32 values). For a normal y the three limbs are exact and add up to y again, but y can get tiny after
rescales (c = exp of a large negative difference), and there the result can differ in the last bits. With `L[2] = 1`
the P.V core leaves y as it is in that case. It is a run-time flag (engine: `FLM_MINICPM_OCR_DECODE_RESCALE_SKIP`,
default on), so one xclbin serves both. **Not bit-identical**: on random data (below) it was, but in the engine the
decode logits of a 24K prompt and of a resumed 3K follow-up differ from step 46 / 6 on (min cos 0.999978, top-1 the
same at every step, replies unchanged).

**Where the time goes** (one attention layer run alone, `run.exe --per-layer`, us; probe builds = timing only):

| cached tokens | installed | installed, no attention math | kvb exact | kvb, no math | kvb, no P.V math | kvb + skip |
|---|---|---|---|---|---|---|
| 128 | 316 | 311 | 310 | | | |
| 1,024 | 399 | 379 | 392 | | | |
| 8,192 | 1,002 | 920 | 888 | 649 | 652 | 693 |
| 24,576 | 2,366 | 2,169 | 2,028 | 1,347 | 1,355 | 1,406 |
| 32,000 | 2,973 | 2,726 | | | | |

* Installed design: K and V stream at ~12.3 GB/s each, 13.5 GB/s without math: bound by one shim channel per stream.
* kvb, exact: the data alone would take 1,347 us at 24K, the Q.K core keeps up (1,355), so the P.V core's math binds
  (~2,000 cycles per 16-key chunk, most of it the emulated multiply by 1.0, which Peano does not overlap).
* kvb + skip: within ~4% of the data time, so the shim channels bind again (13.7 GB/s each). Two cores per KV head
  (positions split, WP6 Step 2) would not add channels; more channels for K / V would be the next lever.

**Bit-identical, checked:** `./run.exe --check --dump` (every attention layer's output) and a 3-step 24-layer run list
(`--dump`), installed `layer_with_lm_head.xclbin` vs this build on the same random state, KV lengths 0, 1, 15, 16, 17,
31, 32, 33, 100, 1,000, 1,024, 4,095, 8,192, 24,576, 32,000, seeds 1234 (kv-scale 1) and 7 (kv-scale 8): 45 / 45
byte-identical, without and with `--rescale-skip` (random data never hit the tiny-y case above).


## Constraints

- Must be traditional layer, sliding window is allowed

## Models

| Model Series | Head Dim | Num Heads | Num KV Heads | Num Heads per group | Num heads per CU | KV Bias | QK Norm | intrinsic mm | Pad Q | Note                       |
|--------------|----------|-----------|--------------|---------------------|------------------|---------|---------|--------------|-------|----------------------------|
| Llama        | 64, 128  | 32        | 8            | 8                   | 2 x (4 -> 1)     | No      | No      | 8x8x8        | No    | single 8x8 q, different k  |
| Llama-3B     | 256      | 24        | 8            | 8                   | 2 x (3 -> 1)     | No      | No      | 8x8x8        | Yes   | row 3, 7 are removed       |
| Qwen-small   | 128      | 16        | 8            | 2                   | 2 x (2 -> 1)     | No      | Yes     | 8x8x8        | Yes   | row 2,3,6,7 are removed    |
| Qwen-large   | 128      | 32        | 8            | 8                   | 2 x (4 -> 1)     | No      | Yes     | 8x8x8        | No    | single 8x8 q, different k  |
| *Gemma-small | 256      | 4         | 1            | 4                   | 4 -> 1           | No      | Yes     | 4x8x8        | No    | Only one CU is used        |
| Gemma-large  | 256      | 8         | 4            | 2                   | 2 -> 1           | No      | Yes     | 4x8x8        | Yes   | zero pad to 4x8 in kernel  |


## Attention implements

For kernels

| Type | Kernel spec  | intrinsic mm |
|------|--------------|--------------| 
| 1    | 2 x (4 -> 1) | 8x8x8        |
| 2    | 4 -> 1       | 4x8x8        |

Dataflow of Q:

- Always from (NH, DH) -> 4 x (NH // 4, DH)
- NH always pads to multiple of 4
- DIM: ((DH // 8, 8), (NH // 4, DH),(8, 1))

Inside Kernel:

- Requires process padding and remove useless data according to parameters

## Compiler (2026-11-03): Peano 22

Already Peano. Rebuilt with the CI pin (Peano = the pinned CI toolchain `llvm-aie 22.0.0.2026090201+a36c62b9` (`.github/actions/wheel-aie-env`); the June 21.0 that `~/setup.sh` brings is older.): prefill logits bit-identical to the June build; decode not bit-identical
but as close to the CPU reference (`--check`: worst cos 0.998865 both, 0.998100 vs 0.997611 at `--kv-scale 8`), and
10K decode 31.4-31.5 vs 31.2 tok/s. The shipped layer.xclbin is this build.
