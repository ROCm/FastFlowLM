<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# vision_attn_kv (MiniCPM-V)

MiniCPM-V vision-tower attention, derived from `../../Qwen3_8/vision_attn`. `Qwen3_8/vision_attn` itself is untouched.

**Contract.** The engine loads this kernel as `vision_attn_kv.xclbin` whenever that file is in the model folder. In that case
the host sequence sends K/V in 128-row chunks (`vision_attn_rtp_addresses::kv_once`). Everything else is the same as
`Qwen3_8/vision_attn`: IRON tile layout, RTP address (57344), buffer arguments, and q/o streams.
`FLM_MINICPM_V_VISION_ATTN_KV=0` keeps `vision_attn.xclbin` instead.

**Not bit-identical** to `Qwen3_8/vision_attn`; see the accuracy table below.

## Kernel changes (`kernel/`)
- **K/V once per round.**
  - Each 128-key chunk (`lk_mt`) arrives exactly once and is used by all four 16-row q tiles.
  - Only `ceil(L_valid / 128)` chunks are sent. Before, every 512-row chunk of `L_pad` was replayed once per q tile.
  - Blocks past `L_valid` inside the last chunk are delivered (their locks are taken) but not computed.
- **bfp16 operands converted once** (`bfp_ops.h`).
  - With `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`, every bf16 8x8x8 mac converts both of its operands.
  - Here each operand is converted once, from the same values:
    - Q is converted per round.
    - A K or V block is converted once for all four q tiles.
    - An S block is converted once for all nine d blocks.
  - The products and their accumulation order are those of the emulation.
- **Lazy rescale.** A row's running max only moves when a chunk max beats it by more than 8 (`lazy_tau`).
  Without a move, exp(s - m) <= e^8, and y and l need no rescale.
- **Rounding.** fp32 -> bf16 conversions round to nearest even (`aie::rounding_mode::conv_even`).
- **From the bit-identical MiniCPM-V attention work:**
  - flat exp loop, compiled alone with the post-pipeliner (`softmax.cc`, linked by `makefiles/kernel.mk`);
  - per-row max over all blocks;
  - softmax, l/y and FV only over blocks that hold a valid key.

## Measured
HX 370, minicpm-v-4.7:1b, 120 frames of 398x224 in one request, tower time:

| attention | tower s |
|---|---|
| `Qwen3_8/vision_attn` path, same engine (`FLM_MINICPM_V_VISION_ATTN_KV=0`) | 9.86 |
| this | 7.57 |

Accuracy is the cosine of the tower output against the tower computed exactly in fp32 on the host
(`FLM_MINICPM_V_VISION_BACKEND=cpu`):

| attention | cosine vs fp32 tower |
|---|---|
| bit-identical NPU attention | 0.991233 |
| this | 0.992006 |

The NPU tower's distance from fp32 is set by the matmuls' bf16/bfp16 arithmetic. This kernel adds nothing measurable to it.

## Build / test
- Build: `make bitstream` (Peano, `ENABLE_CHESSCC ?= 0`). The output is `build/xclbins/attn.xclbin`, deployed as `vision_attn_kv.xclbin`.
- Test: `make host && ./run.exe -d npu2 -l <L> -r 57344`. `host/seq_gen.hpp` emits the kv_once sequence, and the result is checked against `host/cpu_ref_vision_attn.hpp`.
  On HX 370 it PASSES at L = 264 / 1056 / 1500 with cosine 0.99959 / 0.99955 / 0.99955.
