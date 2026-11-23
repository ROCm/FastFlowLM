<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# gemma3:4b decode layer with lm_head (`layer_lm.xclbin`)

A copy of `Gemma3/gemma_npu_bin` (the decode layer that medgemma:4b, medgemma1.5:4b and translategemma:4b
also use) for **gemma3:4b only**. It adds a way to run lm_head on the same xclbin as the layers, so a decode
token needs no hw_context switch (today: layer.xclbin <-> lm_head.xclbin, ~2.4 + ~2.7 ms a token on Strix Halo).

## Build

Peano only, with the pinned toolchain in `FLM_Xclbin/README.md`.

| Command | Output |
|---|---|
| `make bitstream` | `build/GEMMA3_4B/xclbins/layer_lm.xclbin` + `rtp_addresses.txt` |
| `make bitstream STUB=1` | `build/GEMMA3_4B_STUB/...`: same locks and DMA, projection math skipped (data floor) |
| `make txn` | the same design built with `--aie-generate-txn` (kernel contract check only) |
| `make test_lm_head` | `build/test/test_lm_head` |

`make bitstream` fails when:
- one of the 25 RTPs the engine writes is not at its address (`IS_ATTN_PROJ_BUFFER_*` 33280,
  `IS_ATTN_RMS_BUFFERS` 54272, `L_attn_qk_core_*` 9216, `L_attn_kv_core_*` 33792; see
  `FLM_DLL/detail/gemma_npu/gemma_npu_sequence_detail.hpp`);
- a core's program is over 16 KB (the size is printed for every core; a bigger program links but aiecc's CDO
  step then fails without a reason).

## Modes (RTP `IS_ATTN`, written by the host before lock 6 is released)

| proj cores | rms core | What runs |
|---|---|---|
| 1 | 1 | one decoder layer (unchanged) |
| 0 | 0 | lm_head with the final RMS norm on the rms core (in the shared kernels too, never used by an engine) |
| 0 | **2** | lm_head, x already normalized on the host: the rms core copies x instead of normalizing it |

Mode 2 is new here. It keeps the engine's host RMS norm, so the logits are the same bits as `lm_head.xclbin`
gives today. Mode 0 would move the norm to the NPU (saves the x round trip, ~0.05 ms, but different bits).

lm_head in the layer design (`gen_lm_head_in_layer_seq` in `gemma_npu_sequence.cpp`):
- args: 0 x (D bf16), 1 lm_head weights, 2 any D bf16 (the rms core takes one weight buffer even in mode 2),
  3 unused, 4 logits;
- the kernel is built for vocab 262,208 (`common.h`), padded to 263,680 rows = 103 rounds of D logits = 515
  rounds of 512 rows (16 cores x 32);
- weights are `lm_head.weight` laid out like the layer weights: `reorder_cpy(..., 2)` (pairs of 32-row blocks
  per 256-column block), not the `reorder_cpy(..., 4)` of `lm_head.xclbin`;
- logits leave the rms core on MM2S 1 to IT1 S2MM 0; the per-round ping/pong selectors move one step per
  buffer on both sides, so a layer run after an lm_head run starts in step.

## Test (`test_lm_head`)

```
build/test/test_lm_head --model <Gemma3-4B-NPU2 dir> --old <lm_head.xclbin> \
    --new build/GEMMA3_4B/xclbins/layer_lm.xclbin --x <file of D-bf16 rows> --random 4 --iters 10 --cpu-ref 3
build/test/test_lm_head --model <dir> --new build/GEMMA3_4B_STUB/xclbins/layer_lm.xclbin --stub 1
```

The engine no longer uses `lm_head.xclbin`; the test keeps its 32-core sequence (`gen_old_lm_head_seq`) as the
reference. The same x goes to `lm_head.xclbin` and to `layer_lm.xclbin` in mode 2; every logit must have the same bits
(PASSED / FAILED). Both are also compared with a CPU Q4_1 reference (`w = q * scale + min`, block layout as
`proj.cc` reads it). Then it times each back to back and alternating (alternating pays a context switch).

Results 2026-11-23, Strix Halo, Performance mode, 24 real normalized x from gemma3:4b decode at 1K and 4K
context + 4 random:
- logits bit-identical for all 28 (0 of 262,208 differ), same top-1, cos vs CPU 0.999987 or better (both);
- time, back to back: `layer_lm.xclbin` 9.84 ms, `lm_head.xclbin` 7.56 ms; alternating 12.23 / 10.20 ms;
- stub `layer_lm.xclbin`: 9.59 ms, so this lm_head is limited by the layer design's data flow (~44 GB/s),
  not by the 16 cores' math. `lm_head.xclbin` reads at ~55.8 GB/s.

## Program size (`.text`, bytes)

| Core | `gemma_npu_bin` | this copy |
|---|---|---|
| proj (16 cores) | 10,864 | 10,864 |
| rms (`core_1_2`) | 6,608 | 6,736 |
| others | 1,184-4,192 | same |
