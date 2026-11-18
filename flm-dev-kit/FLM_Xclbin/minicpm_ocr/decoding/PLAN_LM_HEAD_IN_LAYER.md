<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Plan: lm_head inside layer.xclbin (no xclbin switch per token)

Goal: run lm_head (Q8_0, 248,320 x 2048) on this design's projection engine as a 25th run per token, so decode
no longer switches between `layer.xclbin` and `lm_head.xclbin` (~5.6 ms per token today, ANALYSIS.md section 15).
Expected: lm_head 7.4 ms + 5.6 ms switching (13.0 ms) -> ~9.8-10.7 ms on 8 weight channels: **~-2.3 to -3.2 ms
per token**, no accuracy change (stays Q8_0).

## Design

| piece | today | change |
|---|---|---|
| projection cores (`kernels/proj_main.cc`) | fixed endless schedule: 3 GDN layers + 1 attention layer; Q4_K only | each run starts by reading a **mode RTP** (behind an RTP lock released by the host): 0 = GDN layer, 1 = attention layer, 2 = lm_head. Mode 2 runs the **Q8_0 half-chunk** loop. |
| Q8_0 math (`kernels/q8_core.h`) | `_Q80_x_bf16` over a 32 x 256 chunk (8 groups) | half-chunk version: 32 x 128 (4 groups, 4,352 B) |
| weight buffers / DMA lengths | sized for one Q4_K chunk (4,736 B) | unchanged: a Q8_0 half-chunk (4,352 B) is padded to 4,736 B in DDR (+8.8 % bytes) |
| RMS core (`kernels/rms_residual.cc`) | per-layer norm + residual, re-sends x once per 512-row round | mode 2: final RMS norm with `model.norm`, no residual, re-send x for 485 rounds (248,320 / 512) |
| output route (`iron/proj_tile.py`, `iron/layer_aie.py`) | packet ids to_rope / to_conv / to_aux / to_glu | new packet id `to_out`: projection cores -> assemble mem tiles -> main mem tile -> a free shim S2MM -> logits buffer in DDR |
| host sequence (`host/qwen3_5vl_npu_sequence.cpp` copy) | 24 layer runs | mode RTP writes in every layer sequence; new lm_head-in-layer sequence: final norm weights, x from DDR, 485 rounds of weights on the 8 channels (one multi-dimensional BD per channel if possible), receive logits, wait |
| host weight load | lm_head for `lm_head.xclbin` (`reorder_cpy_lm_head`) | re-lay lm_head as padded Q8_0 half-chunks in the projection engine's channel/round order (~588 MB) |
| run arguments | x, weights, rms, rope/ssm, cache | same 5 slots: x, lm_head weights, final norm weights, (unused), logits |

## Steps (each checked before the next)

1. **Kernel: Q8_0 half-chunk loop + mode RTP in `proj_main`.** Modes 0/1 must give bit-identical results to
   today (`--check` numbers and `--repeat` identical to `baseline/layer.xclbin`), speed unchanged (`make test`).
2. **RMS core mode 2 + IRON route `to_out` + RTP buffers.** Keep the attention RTP addresses (53248 / 53536,
   the build checks them). Program memory: projection cores ~7 KB + ~2 KB, limit 16 KB.
3. **Host: lm_head-in-layer sequence + weight re-layout** in this folder's `host/`. New `run.exe` checks:
   logits vs a CPU Q8_0 lm_head (cosine, max error, top-1 agreement) on the real 10K context
   (`--prompt-ids`); speed of 24 layers + lm_head in one run list vs today's 34.8 ms (layers + switching).
4. **Engine copy (`../engine/`)**: lm_head as the 25th run of the per-token run list, logits from the new
   buffer, prefill's final logits through the same path; then FastFlowLM end to end (`make test`, FOCUS line).

## Risks

- Packet routing: a new packet id on shared arbiters (this design family had arbiter deadlocks; routing changes
  per build). The new flow is only active in mode 2.
- RTP placement: new RTP buffers must not move the attention RTPs (build fails if they do).
- Host instruction size: 485 rounds x 8 channels of weight BDs per token if issued one by one; prefer one
  multi-dimensional BD per channel.
- 8 weight channels (~55 GB/s) instead of lm_head.xclbin's 16: lm_head itself gets slower; the gain comes
  only from removing the switches.

## Progress

**Step 1 done (2026-11-02).**
- `kernels/proj_main.cc`: `proj_main` now handles one run per call: waits on lock 6, reads `rtp[0]` (mode), runs
  the GDN / attention projections (unchanged `mvm()`) or `mvm_q80_half()` for lm_head (`LM_HEAD_ROWS / 16` rows,
  K = 2048, two Q8_0 half blocks per 256-wide x slice, output packet id `pkt_id_to_out` = 12).
- `kernels/q8_core.h`: `_Q80_half_x_bf16` (same math as `_Q80_x_bf16`, 4 groups); `kernels/model_spec.h`:
  `q80_half_block_t`; `kernels/aie_array_layout.h`: `LM_HEAD_ROWS`, `PROJ_MODE_*`, `pkt_id_to_out`.
- `iron/proj_tile.py`: `RTP_proj_mode_<row>_<col>` (8 x int32) on every projection core, passed to `proj_main`;
  it lands at **33280** on all 16 cores; `makefiles/bitstream.mk` fails the build if it moves.
- `host/qwen3_5vl_npu_sequence.*`: `_set_modes()` (was `_set_proj_mode()`) writes the mode and releases lock 6 (`0x1F000 + 16 * 6`) on
  the 16 projection cores at the start of every layer sequence. `run.exe --no-proj-mode` skips this, for
  xclbins built before the RTP existed (e.g. `baseline/layer.xclbin`).
- Checks: state checksum after one step identical to the baseline (`d7fd26d20f33f7fc`), 0 of 2 repeats differ
  in both run styles, 10K speed 22.26 ms (baseline 22.22-22.31), `--check --kv-scale 8` worst 0.997611.
  Projection core code 7,520 B (was 6,960; limit 16 KB). Mode 2 is compiled but not exercised yet.

**Step 2 done (2026-11-02).**
- `kernels/rms_residual.cc`: reads a mode RTP (`RTP_rms_mode`, lock 6) per run. Modes 0/1/2 = first GDN layer /
  other GDN layer / attention layer (the old branches); mode 3 (`RMS_MODE_LM_HEAD`): final norm, no residual,
  x re-sent for 485 rounds. Host `_set_modes()` also writes this RTP; `RTP_rms_mode` is at **36864** (build checks).
- `iron/proj_tile.py`, `iron/layer_aie.py`, `iron/constants.py`: `to_out` = 12 route, projection cores ->
  assemble mem tiles -> main mem tile -> `IT[1].S2MM[0]`, header dropped.
- The new route made the router capture the RoPE flow (id 1); fixed by RoPE id **17**, see ANALYSIS.md section 17.
  New build gate `tools/check_packet_routes.py`.
- Checks: checksum `d7fd26d20f33f7fc`, 0 of 2 repeats differ, 22.22 ms at 10K, `--check --kv-scale 8` worst
  0.997611. Program memory: projection cores 7,520 B, largest core 10,544 B (limit 16 KB). Mode 3 still unused.

**Step 3 done (2026-11-02).** Host side in `host/` (details: `host/README.md`, "lm_head inside layer.xclbin").
- `reorder_cpy_lm_head_in_layer()` (model def copy): Q8_0 chunks cut into padded half chunks, 8 streams stored one
  after another (588 MB); `gen_lm_head_in_layer_seq()`: modes, x + `model.norm`, **one 73.5 MB linear BD per weight
  channel** (no per-round BDs), logits on IT1 S2MM 0; `cpu_ref::lm_head_logits()`; `run.exe --lm-head-in-layer`.
- Random 10K: cos 0.999985, top-5 5/5. Real story 10K: cos 0.999995, top-1 279 on NPU, CPU and the text.
  Layers after an lm_head run: same state checksum as without it.
- Speed at 10K: 24 layers + lm_head in one run list **30.9-31.3 ms** vs 34.8 ms with the xclbin switch
  (**-3.5 to -3.9 ms/token**); lm_head alone 9.05-9.36 ms (63-65 GB/s), layers unchanged (22.22 ms).
- Next: step 4, the engine copy (`../engine/`) and FastFlowLM end to end.

**Step 4 done (2026-11-02).** Engine copy `../engine/` (README there): lm_head from the layer xclbin, loaded as
`layer_with_lm_head.xclbin` in `FastFlowLM/src/xclbins/Qwen3.5-2B-NPU2/` (the tracked `layer.xclbin` there is
untouched, other engines would hang on the new one). lm_head kept as a separate run after the layer run list so
the run-list rebuild still overlaps it. FastFlowLM end to end at 10K: **decode 26.64 -> 31.17 tok/s**, 128:
32.57 -> 37.9 tok/s, TTFT unchanged. Greedy text drifts after a few hundred characters (layer build + lm_head
rounding), stays fluent.
