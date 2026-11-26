<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# dequant_mm_mix_bfp16

The prompt prefill GEMM of `r2t2:1.7b` (Qwen3-1.7B decoder): `Y[L, Dout] = X[L, Din] @ W[Dout, Din]^T`, bf16 out, on all 32
cores, reading the **decode** weights in place. The decoder's mixed-precision layout keeps q, k, o, up and gate as Q4_1 and v,
down as Q8_0, in 10240-byte units (`FLM_DLL/detail/qwen3_asr_npu`, `reorder_units`); the prefill uses that same buffer, so
the model needs no second copy. The core pairs dequantize the weights to block float (`bfp16ebs8`) once per pass and multiply
them with X, which the host packs as block float in the order the cores consume it.

It is `dequant_mm_Q4_1_bfp16` with these changes:

| | `dequant_mm_Q4_1_bfp16` | this kernel |
|---|---|---|
| weights | Q4_1, prefill layout | Q4_1 **and** Q8_0 blocks of the decode layout; the format is a runtime parameter, per column |
| outputs per core and call | 128 rows x 64 | the same 128 x 64 as `nb` slabs of 64 outputs x `128 / nb` rows (`nb` = 1, 2, 4, 8) |
| short prompts | padded to 512 rows | `nb` = 8 for L <= 64: every weight is read once, X is 64 rows |
| one call | one weight | stacked weights (`mix_band`), e.g. the K, V and Q projections in one call |

## Measured

Ryzen AI Max+ 395, `host/host.cpp` (5 warm-up calls, median of 20), per decoder-layer call of Qwen3-1.7B, in us:

| L | K, V, Q (4096 x 2048) | O (2048 x 2048) | up, gate (12288 x 2048) | down (2048 x 6144) | layer |
|---|---|---|---|---|---|
| 28 | 317 | 244 | 610 | 444 | 1.6 ms |
| 255 | 515 | 264 | 1081 | 576 | 2.4 ms |
| 904 | 1280 | 633 | 3405 | 1605 | 6.9 ms |
| 2048 | 2466 | 1168 | 6866 | 3125 | 13.6 ms |

Short prompts are bound by DDR (20-26 GB/s of weights), long ones by the cores (14-16.5 TFLOPS). Against an fp32 product of the
same dequantized weights and X, both on the block-float grid, the relative L2 error is 0.0041 for every shape: the bf16
truncation of the output.

## Constraints

* `Din % 512 == 0`. Every band's rows are a multiple of `64 nb` for the `nb` that `mix_nb` picks (it lowers `nb` until they are).
* X and Y hold `mix_rows_padded(L, Dout)` rows; padding rows of X are zero, those of Y are ignored.
* The runtime parameters (`K_div_k` = k rounds, format, `nb`; `rtp_act`) are written per column whenever its format changes, after
  the previous pass has drained (`gen_dequant_mm_mix`).
* Built with Peano (`ENABLE_CHESSCC=0`).

## Layouts

**W**: the decode layout. A 32 x 256 block is `bf16 scales[8][32]`, (Q4_1) `bf16 mins[8][32]`, then the quants column major
(16 B per column for Q4_1, low nibble = even row; 32 B for Q8_0); a 10240 B unit is two Q4_1 blocks of one block row or one Q8_0
block and padding; units go per pair of block rows and per unit, the first block row's then the second's. `host/cpu_mix.hpp` builds
it from random weights.

**X**: per pass `mp` (`512 / nb` rows) and core row `r`, one block of

    [512 k round] [256 k slice (2)] [16-row group (8 / nb)] [128 k half (2)] [k tile (16)] [8-row tile (2)]

tiles of 8 rows x 8 k, 72 B each (per row a shared exponent and 8 mantissas, truncated). `host/host.cpp` has the reference packer
(`pack_x_mix`) and the AVX-512 one (`pack_x_mix_fast`); the test checks they agree byte for byte.

**Y**: bf16, row-major `[rows_padded, Dout]`.

## Things learned while building it (Peano / AIE2P)

* A memtile buffer descriptor dimension wraps at most 512 words: the 1280-word chunk of the weight reorder is 4 x 320.
* The odd core of a pair must not take weight chunks from the DMA lock itself: with one credit each, either core can take both
  credits of one chunk and read the other buffer before it is filled. The even core takes both and hands one on through a lock on
  the odd tile.
* A core reads its runtime parameters once the call's first weight chunk is in; it may enter the next call as soon as the previous
  one ends, before the host has written new ones.

## Build and test

    make kernel PEANO_INSTALL_DIR=<peano>
    make bitstream PEANO_INSTALL_DIR=<peano>      # build/xclbins/dequant_mm.xclbin
    make host
    KT_L=28,255,904 ./run.exe                     # K, V, Q bands; prints RESULT lines, PASSED / FAILED
    KT_BANDS=q8:2048 KT_DIN=6144 ./run.exe        # the down projection

`KT_BANDS` (`q4:rows,q8:rows,...`), `KT_DIN`, `KT_L` (comma list), `KT_WARMUP`, `KT_ITERS` set the shape and the timing loop.
