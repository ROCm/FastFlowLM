<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# ple8: the per layer input (PLE) epilogue for 8 token rows (plan S6.8), results 2026-10-29

Design `DESIGN=ple8` in `FLM_Xclbin/Gemma4/verify_decoding` (iron/ple8_aie.py, iron/ple8_tiles.py, kernels/ple8_{gate,up,norm}.cc + ple8_mvm.h, host/test_ple8.hpp,
tools/ple8_ref.py). It computes for 8 tokens t:  `gate = gelu(W_gate . h_t) * pli_t ; up = W_up . gate ; out_t = (h_t + rmsnorm(up) * w_post_pli) * layer_scalar`.

## What works (hardware, real Bolt weights, quiet NPU)
| layer (type) | worst token cos | worst token rel L2 | 100 dispatches bit-identical | hangs |
|---|---|---|---|---|
| 0 (SWA) | 0.999959 | 0.00909 | yes | 0 |
| 5 (global) | 0.999980 | 0.00633 | yes | 0 |
| 35 (global skip) | 0.999980 | 0.00636 | yes | 0 |
| 40 (SWA skip) | 0.999978 | 0.00659 | yes | 0 |
Reference = fp32 numpy from the safetensors weights (independent of the device buffers), gelu_tanh; the device uses the decode design's gelu LUT, so the ~0.6-0.9 % rel L2
is mostly LUT + bf16 rounding.  Gate was cos >= 0.9995 / rel L2 <= 0.03.  gate-only and up-only debug builds (PLE8_STAGE=gate|up) pass at the same level
(cos >= 0.99997, rel L2 <= 0.014).  The gate and up kernels run the decode's `_mvm_bf16_bf16` arithmetic per token (same accumulation order, bit-identical per token).
The L_TOKENS=1 fused layer still builds (173,548 B, RTP map identical) and passes the decode oracle harness: `verify8 PASSED: 0 failing cursor(s)` on layers 0,5,40,35 x cursors 0,9,16,505,
repeat 2, poison (nothing of the fused layer references the new files; the Makefile only compiles them).

## Dataflow (tile placement, packets, BDs)
```
host -> IT4.MM2S0 -> G(4,2).S2MM0   81 transfers of [8][256] bf16: pli rows, then 8 x 10 h slices [rb][kb][t][256] (shim BD sizes {8,10,8,256} strides {0,256,2560,1}); 3-buffer ring
host -> IT5.MM2S0 -> G.S2MM1         W_gate: one BD of 256*2560 bf16 at w_inp_gate_offset (80 blocks of [256 k][32 rows], ping/pong)
G.MM2S0 -> U(5,2).S2MM0              u = gelu(gate) * pli [8][256], one transfer
host -> IT5.MM2S1 -> U.S2MM1         W_up: one BD of 2560*256 bf16 at w_inp_up_offset (80 blocks)
U.MM2S0 -> MT5.S2MM0                 80 packets y[8][32] -> [rb][t][32] (one 40 KB BD)
MT5.MM2S0 -> N(5,3).S2MM0            token-major read-back: one 3-dim BD (T,32)(80,256)(32,1): up row t = 2560 contiguous
host -> IT4.MM2S1 -> N.S2MM1         [w_post_pli(2560) | layer_scalar, 31 pad] (2592 el), then h rows [8][2560]; 9-BD loop restarting at every dispatch
N.MM2S0 -> IT5.S2MM0 -> host         out rows [8][2560]
```
All flows are circuit-switched, no packets, no RTPs, no objectfifos (hand-written BD chains like the decode design).  Tile ids: G = CT[0][4], U = CT[0][5], N = CT[1][5],
MT5.  The decode design's pli tile (4,3) and merge tile (5,3) (and the W_plmp stream, 1.31 MB/layer) are not needed: pli is computed on the host.
Locks: see the headers of kernels/ple8_*.cc.  Per dispatch counts restart the toggles (80 weight blocks, 81 = 3 x 27 x-ring transfers, 9 N-side BDs), so no state carries over.
BD ids used by the test host: IT5 bd_10 (out), bd_14/bd_15 (weights); IT4 bd_0/bd_1 (G x stream), bd_2/bd_3 (N constants + h).  xrt args as the decode host: 0 = x (inputs), 1 = proj weights, 2 = y out.

## L1 / L2 occupancy (from input_with_addresses.mlir; 64 KB per compute tile)
| tile | role | buffers | + stack | highest address end |
|---|---|---|---|---|
| (4,2) | gate | x ring 3 x 4096, W 2 x 16384, pli 4096, u 4096 = 53,248 B | 4,096 | 61,440 B |
| (5,2) | up | u 4096, W 2 x 16384, y 2 x 512 = 37,888 B | 4,096 | 53,760 B |
| (5,3) | norm | wc 5184, up 2 x 5120, h 2 x 5120, out 2 x 5120 = 35,904 B | 4,096 | 54,272 B |
| MT5 | stage | 20480 bf16 = 40,960 B of 524,288 B | | |
Nothing above 61.5 KB; the gate tile is the fullest (headroom 4 KB).

## Performance (host wall time of one dispatch, create_run + wait, 100 dispatches)
| variant | min us | median us |
|---|---|---|
| gate stage, stream only (compute skipped, -DPLE8_SKIP_COMPUTE) | 164 | 172 |
| gate + up stage, stream only | 263 | 279 |
| gate stage, 8 tokens one after another (per token `_mvm_bf16_bf16`) | 391 | 421 |
| gate stage, 8 tokens interleaved (NT=2, shipped) | 237 | 249-259 |
| gate stage, NT=4 (accumulators spill) | 2130 | 2160 |
| gate stage, all 8 interleaved | 617 | 636 |
| full stage, serial per token (first version) | 807 | 825 |
| **full stage, NT=2 (shipped)** | **478** | **546** (quiet box; 0.7-0.8 ms when other forks use the NPU) |
Stream-bound estimate: W_gate and W_up are 1.31 MB each = 80 blocks of 16 KB; one shim channel streams 1.31 MB in ~170 us (7.6 GB/s), gate and up are dependent, so the
stream floor of the whole stage is ~0.3 ms (the measured stream-only number, gate + up).  The shipped stage is 1.8x that floor: it is compute-bound by the per token
vector x scalar bf16 MAC loop (~3 us per 32 x 256 block at 8 tokens).  Decode (1 token) streams 3.9 MB/layer (W_plmp, W_gate, W_up on three channels, ~0.17 ms each, gate and
pli paths parallel, up after) so its stream floor is ~0.35 ms and it overlaps with nothing else of the layer's tail; 8 tokens here cost ~0.55 ms, i.e. ~1.6x the 1-token tail instead of 8x.
Ideas not done (each worth ~0.1-0.2 ms/layer): split gate and up over two tiles each (free tiles (4,4),(4,5),(5,4),(5,5) minus what GLU relocation takes), bfp16 mmul<8,8,8> (accuracy cost, toy: 1.4x), fuse N's token tail with U.

## Host pli contract (what the engine must hand the stage per layer and token)
See the docstring of tools/ple8_ref.py (reference `pli_row`): x0 = bf16 embedding row; x0_pl = bf16 per-layer embedding row slice [layer*256, +256) of model.per_layer_token_embd
(int8 row * f32 scale per 32); t = bf16(W_plmp . x0) (fp32 accumulate, W_plmp = model.per_layer_model_proj.weight_layerN stored [row block][k][32]);
t = bf16(t * 0.01976423537605237); t = bf16(t * rsqrt(mean(t^2)+1e-6) * w_pn) with w_pn = model.per_layer_proj_norm.weight; t = bf16(t + x0_pl); pli = bf16(t * 0.7071067811865476).
Same ops and roundings as kernels/proj_layer_embedding.cc.  Per verify dispatch the host computes this for the 8 tokens and every layer (the 256-vector of layer N is a slice of one
42 x 256 computation per token: W_plmp . x0 is per layer, so it is 42 mat-vec products of 256 x 2560 per token, ~110 MMAC per token on the host, once per token not per layer type).
The stage then takes per layer in the x BO: h rows [8][2560] | pli [8][256] | w_post_pli [2560] | layer_scalar + pad [32] (bf16).

## Notes for the integration into the fused verify layer
* h comes from the rms tile's resident h (lender tile), not from DDR: G's input stream (pli + 80 h slices, h re-sent per row block) must be re-plumbed (memtile replay of h slices
  like the x replay tile) and N's h rows come from the lender; the shim BD with a stride-0 repeat dimension (sizes {8,10,8,256}, strides {0,256,2560,1}) works on hardware.
* The kernels read their inputs only after the lock acquire; a read of `wc[2560]` before the acquire returned 0 and zeroed the whole output (found by the PLE8_NORM_DEBUG bring-up switch).
* The stage consumes 5 DMA channels of the shim columns 4/5: IT4 MM2S0/1, IT5 MM2S0/1 + S2MM0.
* Debug builds: `STAGE=gate|up|full KFLAGS=... TAG=... work/scripts/build_ple8.sh`, `work/scripts/run_ple8.sh <layer> <repeat> <stage>`, `work/scripts/build_ple8_dbg.sh <1|2|3>` (norm tile out := up | h | constants).
