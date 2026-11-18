<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Gemma4 `lm_head_8`: final vocabulary projection for 8 token rows per dispatch

Verify-phase lm_head of the DFlash speculative decoder for `nemotron3.5-bolt-dflash:e4b` (Gemma-4-E4B, vocab 262144, D 2560). The single-row
shipping xclbin is `../lm_head_npu_bin`; this one takes **8 pre-norm hidden rows** and returns **[8][vocab] softcapped bf16 logits** in one
dispatch, reading the 419 MB Q4NX weights once.

## Result (hardware, Strix, Peano 22.0.0.2026090201+a36c62b9; `make test`)

| | single row (`lm_head_npu_bin`) | `lm_head_8` |
|---|---|---|
| dispatch (host wall, quiet box) | 9.2 ms | **9.0 ms median** (8.6 min) for 8 rows |
| eight rows | 8 x 9.2 = 74 ms | 9.0 ms (**8.2x less**) |
| first dispatch after load | | 21 ms (warm-up), dispatches 2.. are steady |
| vs single-row logits | | argmax 8/8 on two different row sets (ties within 2 bf16 ulps at the softcap allowed), cos >= 0.9998 |
| vs exact fp32 reference (`check_lm8.py`, real Bolt weights) | relL2 2.3-3.3 % | relL2 2.6-3.3 % (<= 1.24x the single-row error on every row) |
| repeats | | 100 alternating A/B dispatches bit-identical to their first result; dispatch 3 == dispatch 1 |

Eight rows cost the same as one because the lm_head is weight-stream bound (about 46 GB/s) and the 8-token core (the toy's `q4_v4`, 1.9 us per
32x256 block) stays under that budget: 2560 blocks per core x 1.9 us = 4.9 ms of compute against ~8.5 ms of weight DMA.

## Interface (what the engine passes)

xclbin `lm_head_8.xclbin`, kernel arguments in the same order as the single-row lm_head: `(y, w, x)` (arg ids 0, 1, 2).

| arg | content | size |
|---|---|---|
| `w` | **the same weight BO layout as the single-row lm_head** (`lm_head.weight` of the Q4NX package, `reorder_cpy(..., D, 4)`); the weights are not duplicated, the engine can pass its existing lm_head weight BO | 419,430,400 B |
| `x` | `[8 rows interleaved per 8 columns \| model.norm.weight]` bf16, `(8+1)*D` elements: element `(blk*32+kt)*64 + t*8 + j` of the first `8*D` holds `hidden[t][blk*256+kt*8+j]` (`blk` = column block of 256, `kt` = 8-column group inside it, `t` = token 0..7, `j` = 0..7); then `model.norm.weight[D]` (plain `x*w`, the same convention and the same stored weight as the single-row path) | 46,080 B |
| `y` | raw `[round 0..255][col 0..7][core 0..3][t 0..7][32]` bf16 = `V*8` elements. Logit of token `t` for vocab id `r*1024 + col*128 + core*32 + n` is `y[((r*8+col)*4+core)*256 + t*32 + n]` | 4,194,304 B |

Instruction stream (host, one dispatch): one `x` BD of `9*D` elements on `IT0`; per round and column one `y` BD of 1024 elements (`(r*8+col)*1024` offset)
and one weight BD of `W_PER_COL` elements (identical to the single-row sequence); RTP write of the softcap scale (30.0f bits) to every compute
tile at the address in `build/GEMMA4_E4B/xclbins/rtp_addresses.txt` (`RTP_LM8_FINAL_SCALE_*`, 61440). `host/host.cpp` contains `gen_seq_l8`,
`pack_x`, `unpack_y` as the reference implementation; the engine's `gen_lm_head_seq` differs only by the three numbers above.

The kernel applies, per token: `rmsnorm(x)*nw` exactly as the single-row kernel (`x*w`, then `* 1/sqrt(mean+eps)`, bf16 result), the Q4NX GEMM, and the
softcap `tanh(y/30)*30` with the single-row kernel's bf16 arithmetic, so the logits are on the same scale as the plain path and can be sampled directly.

## Design

Same topology as `lm_head_npu_bin` (8 columns x 4 cores, one memtile per column, 256 rounds of 1024 vocab rows, K_BLOCKS 10), changes:

- x is **resident on every core** (8 rows = 40 KB + norm weight 5 KB; stack reduced to 4 KB, L1 use ~63 KB), sent once per dispatch on the same circuit as
  the single-row design. Its `x_prod_lock` is released at the end of each dispatch, so the next dispatch's x is accepted (no one-shot BD: dispatch 2 and
  100 work, unlike the first Qwen `lm_head_8` attempt).
- core = `q4_v4` of `../verify_decoding/toys/q4_mvm_L8`: 128-lane nibble unpack, dequantise `q*scale+min` once per element, `filter_even/odd(.,8)` for the two
  8-row B tiles, `aie::mmul<8,8,8>` with the 8 tokens as M; accumulators `[sl][t][8]` f32 in `.data`; per round the 8x32 outputs go through the softcap.
- the final RMSNorm runs on the core once per dispatch before the round loop: sum of squares per token in **scalar** fp32 (the 16/8-lane `mac_square`
  version returned zeros under Peano in this layout, see the log), the `1/rms` Newton steps as in the single-row kernel, in-place normalisation with 16-lane vectors.
- y per core per round is `[t][32]`, the memtile concatenates the 4 cores to `[core][t][32]` and sends 1024 elements per column per round.

## Test

```bash
source ../../setup.sh     # the AIE toolchain (FLM_Xclbin/README.md); pin Peano as CI does
make bitstream MODEL_TYPE=GEMMA4_E4B                         # ~4 s with Peano; build/GEMMA4_E4B/xclbins/lm_head_8.xclbin (+ rtp_addresses.txt)
make host MODEL_TYPE=GEMMA4_E4B
./run.exe -m GEMMA4_E4B --repeat 100 --dump /tmp/lm8          # runs the single-row xclbin and the 8-row xclbin on the same rows
python3 check_lm8.py /tmp/lm8                                # exact fp32 reference from the real weights, ACCURACY PASS/FAIL
```

`run.exe` options: `--model_dir` (Bolt NPU2 package with `model.q4nx`), `--xclbin1` (single-row xclbin, default the one in the FLM tree), `--noise`, `--seed`, `--skip1`.
Gates: first and second dispatch correct against the single-row path, dispatch 3 bit-identical to dispatch 1, every timing-loop dispatch bit-identical to its
first result, argmax 8/8 (2-ulp ties at the cap allowed), cos >= 0.9995 against the single-row logits; `check_lm8.py`: error against the exact reference
<= 1.4x the single-row error on every row. The test rows are random mixtures of embedding rows minus the mean embedding plus noise (a hidden state aligned with
the tied embedding saturates the softcap and makes argmax meaningless).

## Known limits

- Only `GEMMA4_E4B` is built and tested (the kernel is generic in `MODEL_TYPE`; E2B has D=1536, not measured).
- The 8-token `mmul` is emulated in bfp16 (`-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`); the error against exact is 1.0-1.24x the single-row core on this data, but rows
  with a large positive weight row sum would see a larger bfp16 floor bias (toy `RESULTS.md`); the real lm_head weights do not show it.
- Logits are returned in the interleaved raw layout; the engine un-interleaves (about 1 ms for 4 MB), or a later version can scatter in the shim BDs.
