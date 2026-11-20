<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# gated_attention_prefill_chunked (MiniCPM-V 4.7 1B prefill attention)

Causal prefill attention for 8 q heads / 2 kv heads of 256 (GQA 4:1), scale 1/16, 32 query rows per core. It is
`FLM_Xclbin/Qwen3_5/gated_attention_prefill` (lazy running max, q / K / V^T as bfp16 tiles made on the host) with a faster
P.V and a faster q.k. **Output is bit-identical to that kernel** (same hashes, `-L 2048`: `38ee823d7414beb7`; `-L 4096 -B 12288`:
`eefbb4d265049d85`; `-L 4096`: `1bd6a257736f936e`; `-L 4096 -B 4096`: `93839d884ebaa912`).
Shipped as `attn_bfp16c.xclbin` of `MiniCPM-V-4.7-1B-NPU2`; the engine (`FLM_DLL/detail/minicpm_v_4_7_npu`) makes the tiles.

## What changed against the 2B kernel

* **P.V keeps y in the accumulators over 8 key tiles.** The 2B kernel loaded and stored the fp32 y tiles once per 16-key tile:
  per tile 512 vector stores, the loop was 33 bundles for 8 MMULs (store bound). Now V^T arrives chunk-major and one chunk (4608 B,
  same size as before) is 4 dh blocks of all 128 keys of a block, so the four y tiles of a (half, pair of dh blocks) stay in
  accumulators over the 8 key tiles: 11 bundles for 8 MMULs.
  V^T layout per 128 keys (36,864 B): `[chunk c 0..7][pair dp 0..1][key tile kt 0..7][V00 V01 V10 V11][72 B]` where `V(j, kb)` is
  dh block `4c + 2dp + j`, keys `8 kb .. 8 kb + 7` of the tile. The host (`update_kv_bfp16`) scatters its per-16-keys tiles into it
  (288 B pieces); the shim BDs are unchanged (linear blocks).
* **P is converted to bfp16 once per block, in place** at the start of `s` (`convert_p_bfp16`, 288 B written per 512 B read).
* **q.k unrolled by two k steps**: 8 MMULs in 10 bundles (4 in 8 before). Accumulation order is unchanged, so results are bit-identical.

## Numbers (`run.exe`, us per call, 5 iterations, idle NPU)

| `-L` / `-B` | 2B kernel | this kernel |
|---|---|---|
| 4096 / 0 | 6461 | 4734 |
| 4096 / 4096 | 17911 | 12767 |
| 4096 / 8192 | 28966 | 20454 |
| 4096 / 12288 | 39719 | 27679 |
| 2048 / 0 | 1928 | 1589 |
| 1024 / 0 | 799 | 666 |

Ablations of the 2B kernel at `-L 4096 -B 12288`: full 39.6 ms, MMUL only (no mask / softmax) 29.6, no compute (DMA only) 19.5.
The K/V stream (one broadcast stream, 576 B per key) is ~13.5 GB/s there, so ~19.5 ms is the floor of this mapping.

## Build / test

Same as the 2B design (`make bitstream`, `make host`, `./run.exe -L 4096 -B 12288 -x build/final/xclbins/attn.xclbin`). The python
carries compat shims (mlir-aie of CI and newer); a local build with the pinned Peano gives the same hashes and speed as the
CI-built 2B xclbin. `host/host.cpp` produces the chunk-major V^T layout.
