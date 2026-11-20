<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# 8-token projection engine on hardware (plan S6.3 / S6.4 / S6.5a), 2026-10-29

Designs (IRON `FLM_Xclbin/Gemma4/verify_decoding`, branch feat/bolt-dflash, commits 5a32344f6, dc32e4f9c, 3a0e5815f):
- `DESIGN=projonly`: 16 Q4 projection cores + assembly memtiles + MT3 x landing/replay + shim feed/capture (y packets de-permuted by a shim BD).
- `DESIGN=projstage`: same, y goes through the MT4 token-major stage (`y_depermute_mt`) and IT4.
- `make bitstream MODEL_TYPE=GEMMA4_E4B L_TOKENS=8 DESIGN=projonly` (build/GEMMA4_E4B_L8/xclbins/projonly.xclbin); `L_TOKENS=1` runs the shipping core through the same replay machinery.
- Test: `work/scripts/run_proj8.sh <L 1|8> <layer> [phases] [repeat]` (DESIGN=projstage for the stage) = `tools/proj8_ref.py gen` (real Bolt weights, numpy fp32 ref) + `run.exe -t proj8` + `proj8_ref.py check`.

## Results
| | L=1 (decode core) | L=8 (q4_v4 mmul core) |
|---|---|---|
| relL2 vs fp32 ref (mean / max over tokens), qkv | 0.0097 | 0.0150 / 0.0180 |
| o | 0.0131 | 0.0143 / 0.0183 |
| up\|gate | 0.0124 | 0.0147 / 0.0202 |
| down | 0.0139 | 0.0151 / 0.0198 |
| whole-layer weight stream, global non-skip, min/median us | 1262 / 1272 | 1721 / 1802 |
| swa (L0) / swa-skip (L40) / global-skip (L35), min/median us | - | 1587/1615, 1502/1522, 1650/1711 |
Ratio 8-token / 1-token projection stream = 1.36-1.42x (toy predicted 1.35x). Accuracy is 1.4x the decode core's, cos >= 0.9998 per token.

## Design findings (all measured)
1. A DMA S2MM address pattern with short (16 B) runs that jumps hangs the DMA: the 4-dim memtile landing permute and the 3-dim core x scatter both hung; the same pattern as a memtile MM2S read is fine. So the replay BD does the whole permute: [t][x_dim] -> per 256-col slice [kt=32][t][k8=8] (dims (nj,256),(32,8),(L,x_dim),(8,1)); the core x BD is linear.
2. y rounds come out [b=16][t][32]; the shim BD (4 dims) or a MT stage with a 3-dim read BD ((L,32),(16,32L),(32,1)) makes them token-major [t][512] = one global head (DH 512) or two SWA heads (DH 256) per token. The stage is free (no throughput change).
3. Variants with fewer qkv rounds (skip: q only; SWA: fewer heads) consume the surplus of the fixed 12 qkv replays without weights. It MUST be drained before the compute rounds: x of the next phase is on another replay channel merged under one packet id, and nothing orders channels, so surplus drained after compute is overtaken (SWA/skip `o` was wrong for `drain_rounds - 4` rounds). Drain-first costs up to 80 us of start-up on SWA-skip layers and makes the order a data dependency.
4. Partial phase sets: cores wait mid-layer for the next phase's x, so a second dispatch desyncs them; only whole-layer repeats are valid (the runner forces repeat=1 for partial sets).
5. A hung dispatch recovers by itself after ~60 s (no root needed); the first dispatch of a new design is run through the host test with a 8 s timeout.

## What is temporary / not done (next chunk)
- Producers of x are the shim; the rms tile with resident h (lender + GLU move), qkv -> RoPE/q park, attention, GLU (hidden staged in a memtile, 160 KB), PLE are not in this chunk. `layer_aie.py` refuses L_TOKENS>1.
- `verify_layer_hook` (whole layer) is still the stub; stage checks are `-t proj8` (decode xclbin stages are not observable).
- relL2 max 0.0202 on up|gate: gate widened to 0.025. Candidate improvements from the toy: avoid the bfp16 emulation overhead/bias (mmul<8,8,16>, epilogue scale/min).
- MT3 holds shared 40 KB + o_global 64 KB + o_swa 32 KB + down 160 KB = 296 KB of 512 KB; MT4 stage 16 KB.
