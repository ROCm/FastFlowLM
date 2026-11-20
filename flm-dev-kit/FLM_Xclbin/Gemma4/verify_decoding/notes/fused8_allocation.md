<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# fused8: resource allocation of the 8-row verify layer (plan S6.11), written before building

Sources: proj_engine_L8_results.md (proj), rmsglu_L8_results.md (rms/glu), ple8_L8_results.md (ple), attn8_L8_results.md (attn).
Coordinates: AIE (col,row); IRON CT[r][c] = (col c, row r+2). Compute tile DMA = 2 S2MM + 2 MM2S; memtile = 6 + 6; shim = 2 + 2.

## Tiles (no conflicts: the four designs use disjoint compute tiles once PLE pli/merge are dropped)
| tile (col,row) | role | origin |
|---|---|---|
| (0,2)-(1,5), (6,2)-(7,5) | 16 Q4 cores (8-token q4_v4 core) | proj |
| (2,2) | global qk | attn (decode tile) |
| (2,3) | global kv (64,192 B: do not grow) | attn |
| (2,4) / (2,5) | SWA qk / SWA kv | attn |
| (3,2) | rms norm tile (4 phases), residual h on the lender | rmsglu |
| (3,3) | lender (no core): h0..h7 = 40,960 B | rmsglu |
| (3,4) / (3,5) | RoPE global / SWA | attn |
| (4,2) | PLE gate G | ple8 |
| (4,3) | free (decode pli tile is gone: host PLI) | |
| (4,4) | GLU | rmsglu |
| (4,5) | free | |
| (5,2) | PLE up U | ple8 |
| (5,3) | PLE norm N (was the decode merge tile) | ple8 |
| (5,4), (5,5) | free | |

## Memtiles
| MT | users | S2MM ch | MM2S ch | buffers |
|---|---|---|---|---|
| MT0, MT6 | y assemble + weights (decode) | 0-5 | 0-4 | unchanged |
| MT1 | main: x FIFO + y assemble + weights | unchanged | unchanged | unchanged |
| MT7 | weights | 4,5 | 0-3 | unchanged |
| MT2 | KV staging (decode, unchanged) | | | 196,608 |
| MT3 | x replay: landings main(qkv, up|gate) 40,960, o_global 65,536 (= o park), o_swa 32,768 (= o park), down 163,840 (GLU rounds) | 4 landing | replay fan-out | 303,104 |
| **MT4 (conflict: q park / y stage / glu stage)** | hub A: q park (g: S2MM0 MM2S0, swa: S2MM1 MM2S1), y stage global (S2MM2 -> MM2S2, reads [t][512] = one head per token), y stage SWA (S2MM3 -> MM2S3, reads [head][t][256]), GLU stage (S2MM4 -> MM2S4) | 0-4 | 0-4 | 98,304 + 2 x 8,192 + 2 x 8,192 + 16,384 = ~147 KB |
| **MT5 (conflict: rows aggregator / ple stage)** | hub B: rows stage (S2MM0 -> MM2S0), ple stage (S2MM1 -> MM2S1), ple feed (pli from IT4: S2MM2; rms h2 rows: S2MM3; PLE constants from IT4: S2MM4; MM2S2 -> G: pli then 8x10 h slices; MM2S3 -> N: constants then h rows) | 0-4 | 0-3 | 40,960 + 40,960 + (4,096 + 40,960 + 5,184) = ~132 KB |
Merged memtile DMA program: a memtile has ONE `aie.memtile_dma` body; the hubs build one body with a block plan (iron/fused8_mt.py), each stage owning its channels and a lock-id range.

## Packet ids (5 bits; same id is only reused for different sources / destinations)
| id | route |
|---|---|
| 0 | attention o per token (global kv tile -> MT3 o_global landing; SWA kv tile -> MT3 o_swa landing; 8 packets each) |
| 1 / 2 | engine y -> MT4 y stage global / SWA |
| 4 | engine y (o, down rounds) -> MT5 rows stage |
| 8 | engine y (up|gate rounds) -> MT4 GLU stage |
| 9 / 10 | RoPE tables: IT3.MM2S0 -> global RoPE S2MM1; IT3.MM2S1 -> SWA RoPE S2MM1 (new: were pkt 2 of the stage designs) |
| 11 | rms weights: IT3.MM2S1 -> norm tile S2MM1 (packet instead of circuit: MM2S1 is shared with the SWA table) |
| 12 / 13 | KV replay IT2 -> MT2 (decode) |
| 16 | norm tile -> MT3 main landing (qkv, up|gate) |
| 17 / 18 | MT3 o landings (kept as keys; the attention stamps 0, see above) |
| 19 | GLU -> MT3 down landing (20 rounds) |
| 20 | replay merge (MT3 -> MT1 x FIFO) |
| 24 | norm tile phase 3 rows -> MT5 ple feed (h2) |
| 25 | MT5 rows stage -> norm tile |
| 26 | IT3.MM2S0 -> norm tile (layer input rows) |
| 27 / 28 | IT4.MM2S0 pli -> MT5; IT4.MM2S1 PLE constants -> MT5 |

## Shim channels
| shim | MM2S0 | MM2S1 | S2MM0 | S2MM1 |
|---|---|---|---|---|
| IT0, IT1, IT6, IT7 | weights (decode) | weights | y (decode) | |
| IT2 | K append/replay | V append/replay | KV append global | KV append SWA |
| IT3 | x rows (26) + global RoPE table (9) | rms weights (11) + SWA RoPE table (10) | - | - |
| IT4 | pli (27) | PLE constants (28) | - | - |
| IT5 | W_gate | W_up | **layer output rows (from N)** | |
The layer output leaves on IT5.S2MM0 (as in the ple8 stage), not IT3.

## Conflicts and resolutions
1. MT4: projstage y stage vs q park vs glu stage -> one hub program (above); the y stage is instantiated twice (global read BD `[t][512]`, SWA read BD `[head][t][256]`) because the SWA RoPE wants one head per `[L][DH]` chunk while a 512-row round holds two SWA heads.
2. MT5: rows aggregator vs ple stage -> hub B; PLE feed (pli / constants / h2 with the stride-0 repeat for G) moved into the memtile because G.S2MM0 and N.S2MM1 each need two ordered sources (shim constants then h).
3. Shim IT3 MM2S1: circuit flow to the norm tile blocks sharing -> packets (id 11, 10).
4. o phase landing: the attention emits 8 packets (one per token) while the replay tile lands one packet of 8 rows -> `land_rounds` landing in proj_x_replay_mt_glu (n BDs of one row each, [t][DQ] layout, normal replay read BD).
5. Norm tile phase 3 output: shim (stage design) -> MT5 ple feed (id 24).
6. RoPE y feed: circuit from the y stage instead of packets from the shim.
7. Tiles (4,4)/(4,5)/(5,4)/(5,5): GLU keeps (4,4); the others stay free.
8. DMA channel budget per memtile checked above (MT4 5+5, MT5 5+4 of 6+6).
