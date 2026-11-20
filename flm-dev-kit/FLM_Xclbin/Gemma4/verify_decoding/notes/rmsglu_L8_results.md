<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# 8-row rms_residual + GLU stage design (plan S6.6 / S6.7), 2026-10-29

IRON `FLM_Xclbin/Gemma4/verify_decoding`, branch feat/bolt-dflash, commit `5f1c36d3c`. Design `DESIGN=rmsglu`:
`make bitstream MODEL_TYPE=GEMMA4_E4B L_TOKENS=8 DESIGN=rmsglu BUILD_ROOT=build/rmsglu_L8 KERNEL_SRCS="<proj_main.cc rms_residual_l8.cc glu_l8.cc>"`
(`RMSGLU_TAPS=1` + another BUILD_ROOT builds the debug-tap variant). Test: `work/scripts/run_rmsglu.sh <layer> [repeat]`
(`XB=rmsglu_taps_L8 TAPFLAG=--taps` for the taps), host `host/test_rmsglu.hpp` (`run_rmsglu.exe -t rmsglu`), reference `tools/rmsglu_ref.py`.

## What runs on the NPU (one whole layer minus RoPE/attention/PLE per dispatch, real Bolt weights)
```
shim x[8][D]  --pkt 26--> norm tile (3,2): phase 1  h = x (8 rows on lender (3,3)), y = n_in(h)   --pkt 16--> MT3 'main' --> 16 Q4 cores: qkv
                                              (qkv y, pkt 1/2, captured by the shim: stands in for RoPE/attention)
shim o_x[8][DQ] (attention-output stand-in, sent after the qkv capture) --pkt 17/18--> MT3 o landing --> cores: o
cores y (pkt 4) --> MT5 y_rows_mt (5 rounds -> 8 token rows) --pkt 25--> norm tile: phase 2  h += n_pa(o), y = n_pff(h) --pkt 16--> MT3 'main' --> cores: up|gate
cores y (pkt 8) --> MT4 glu_stage_mt ([half][t][256]) --> GLU (4,4) --pkt 19, 20 rounds--> MT3 'down' buffer [slice j][t][256] --> cores: down
cores y (pkt 4) --> MT5 --> norm tile: phase 3  h += n_pf(down); rows --pkt 24--> shim (the layer output h2; PLE tail hook)
```
Every norm is `bf16(bf16(v*w) * 1/sqrt(mean(v^2)+eps))`, the decode kernel's arithmetic; the weights are the four rms vectors of the decode rms BO.

## Results (relL2 vs the fp32 chain with bf16 stepping; cos >= 0.9995 and relL2 <= 0.03 per stage; 100 dispatches per layer, all bit-identical, no hang)
| layer | qkv | h2 (layer out) | o proj | up\|gate (after phase 2) | GLU+down (on the device's own up\|gate) | latency min / median us |
|---|---|---|---|---|---|---|
| 0 swa | 0.0225 | 0.0290 | 0.0183 | 0.0227 | 0.0187 | 1736 / 1793 |
| 5 global | 0.0211 | 0.0150 | 0.0176 | 0.0228 | 0.0081 | 1913 / 1942 |
| 35 global-skip | 0.0258 | 0.0279 | 0.0165 | 0.0277 | 0.0268 | 1865 / 1879 |
| 40 swa-skip | 0.0194 | 0.0290 | 0.0178 | 0.0236 | 0.0233 | 1707 / 1723 |
(relL2 max over the 8 tokens; cos min 0.99958; the stages are measured with the RMSGLU_TAPS=1 build, the latency with the plain build.)
The q4 core alone is 1.4-2% (proj_engine_L8_results.md); the end-to-end h2 of 2.9% on the swa/skip layers is the sum of three q4 projections and two norms, still within the 0.03 gate.

## Perf
- Projection-only stream (proj_engine_L8_results.md) L0/L5/L35/L40: 1587 / 1721 / 1650 / 1502 us. This design adds the norm tile (3 serial phases), the GLU round trip and the down landing: +150 ... +215 us per layer.
- Weight stream bound at 8 tokens: up|gate|down = 49.2 MB of Q4 weights; the decode core streams the whole 65.6 MB of a global layer in 1262 us (52 GB/s), so ~950 us for up|gate|down at 1 token; the 8-token core is compute bound at 1.36x that (the toy's finding). The down projection cannot overlap the up|gate stream: it needs the whole hidden.
- Equivalent decode stages for 8 tokens: 8 sequential decode dispatches of a layer measured 2.5-3.0 ms each (host overhead included, harness baseline), i.e. 20-24 ms against 1.9 ms here (this design has no attention/RoPE/PLE yet, so only the order of magnitude is meaningful).

## Occupancy (build/rmsglu_L8, tools/occupancy.py; L1 limit 65536, memtile 524288)
| tile | role | stack | static buffers | top of buffers |
|---|---|---|---|---|
| (3,2) | norm tile | 4096 | 35,872 (w 20,480, x ping/pong 2x5,120, y 5,152) | 39,968 |
| (3,3) | lender, no core | 0 | 40,960 (h0..h7, 8 x 5,120) | 59,392 (bank round-robin) |
| (4,4) | GLU | 4096 | 49,152 (x ping/pong 2 x 16,384, y ping/pong 2 x 8,192) | 57,344 |
| 16 proj cores | q4 8-token core | 10,240 | 20,672 | 53,312 |
| MT3 | replay | | 303,104 (main 40,960; o_global 65,536; o_swa 32,768; down 163,840) | 303,104 |
| MT4 | glu stage | | 16,384 (2 x 8,192) | 73,728 |
| MT5 | rows stage | | 40,960 | 40,960 |
| MT0/1/6/7 | assemble + weights | | 49-66 KB | <= 331,776 |
All below 63 KB / 480 KB.

## Deviations from the plan (all measured or forced)
1. **No separate 160 KB memtile for the GLU hidden:** the GLU output lands round by round (20 BDs, 4,096 elements each) straight in MT3's `down` x buffer, which the replay machine already owns (163,840 B). Layout `[slice j][t][256]`; the down replay BD reads it as `[kt][t][k8]` with the 4-dim BD (nj, 2048), (32, 8), (8, 256), (8, 1).
2. **The GLU is elementwise on [half][t][256] rounds:** the MT4 stage read BD turns a raw round [b][t][32] into [half][t][8 x 32] (4 dims, 64 B read-side runs), so `glu_l8.cc` is a flat pass over 4,096 elements and its y round is already one 4,096-element chunk of the down x buffer. No write-side permute anywhere (those hang the DMA).
3. **o / down rows aggregator (MT5):** the 5 rounds of an o or down projection (raw [b][t][32]) land in one buffer and leave as 8 token rows [t][2560] in one packet (4-dim read BD (8,32),(5,4096),(16,256),(32,1)); the norm tile needs whole rows (the sandwich norm of o / down needs the full row before the add).
4. **Every BD needs both an acquire and a release** (aiecc assertion `configureLocksInBdBlock`): multi-BD landing chains pass a token through a private chain lock (rmsglu_tiles.py `proj_x_replay_mt_glu`), the rows landing uses one credit per round.
5. **Phase order across replay channels is a data dependency:** the replay channels of all phases merge into one x FIFO and nothing orders them against each other. Sending the o input to its landing buffer early made its replay interleave with the qkv replays (result: garbage with cos 0.1). The host test sends the o input only after the qkv capture completed; the full layer gets this from attention.
6. **Debug taps use a second memtile read channel** (MT5 and MT4 send the raw buffer to IT4 after their main read, ordered with a `mid` lock) because two packet-flow ops with the same id and different destinations are rejected ("a master port can only be tied to one arbiter").
7. **No RTPs in the norm and GLU tiles** (the replay machine fixes the phase counts; the variant-dependent qkv drain stays in the projection cores). GLU `x` is one 8,192-element buffer = [up round | gate round].

## Interfaces for integration (iron/rmsglu_aie.py is the reference wiring)
- Placement: norm tile CT[0][3] = (col 3,row 2), lender CT[1][3] = (3,3) (reachable through the north window; no core, no DMA), GLU CT[2][4] = (4,4), MT3 replay (existing), MT4 glu_stage, MT5 y_rows. The PLE tiles (4,2),(4,3),(5,2),(5,3) and (4,5),(5,4),(5,5) stay free for the PLE design; GLU (4,4) takes the free tile that PLE does not need.
- Packet ids: engine y: 1 (rope), 2 (swa rope), 4 (o and down -> MT5), 8 (up|gate -> MT4); landing on MT3: 16 (norm tile phases 1 and 2 -> 'main'), 17 / 18 (o input global / swa), 19 (GLU -> down); norm tile: 24 out (phase 3 rows), 25 in (MT5 rows), 26 in (layer input); taps 28 / 29. Ids are 5-bit; the norm tile's master port carries two destination sets (16, 24).
- Norm tile channels: S2MM0 x rows (ping/pong, one row per buffer, ids 25/26), S2MM1 the four norm vectors [w_in | w_post_attn | w_pre_ffn | w_post_ffn] (4D, once per dispatch), MM2S0 one packet per row (header word at y[14:16]). Locks 0..5 as in the decode tile; per dispatch 3 phases x 8 rows = 24 rows (even: the ping/pong parity survives dispatches).
- **PLE tail hook:** phase 3 ends with `h2` = h[0..7] on the lender tile; in the stage design it leaves as 8 packets with id 24 to the shim. In the full layer route id 24 (or read h from the lender through the gate tile's DMA) to the PLE gate/up tiles; `h` stays resident until the PLE tail has consumed it. The gate tile (4,2) can no longer read the norm tile's y through its west window; it takes h rows by DMA.
- Host: whole-layer repeats are valid; shim order per dispatch: capture BDs, layer input, norm weights, qkv weights, wait qkv capture, o input, o / up|gate / down weights, wait outputs (host/test_rmsglu.hpp).

## Not covered
RoPE/q park/attention (other fork), PLE (other fork), the fused layer_aie.py integration (refuses L_TOKENS > 1), Chess build (Peano only here), a sweep of qkv-variant drain timing beyond the four layer types.
