<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# vision_mm_p

MiniCPM-V vision-tower matmul, design P: design C (`../vision_mm_c`) with x and w handed over **already packed as bfp16**
(bfp16ebs8), so the cores skip the bf16 -> bfp16 conversion C runs on every chunk and DDR carries 9/16 of the operand bytes.
y stays bf16 row-major. Optional `vision_mm_p.xclbin`: when the model folder has it the engine runs every matmul with
N % 384 == 0 on P for every M (C is then not loaded; `FLM_MINICPM_V_VISION_MM_P=0` falls back to C / A / B).
**Bit-identical to design C** on pre-rounded operands (the engine always pre-rounds), because the host's pack is the core's
own conversion of those values.

## Operand formats (bytes)
A bfp16 block is 9 B: the largest biased fp32 exponent of 8 consecutive k of one row, then 8 int8 mantissas, each the signed
24-bit mantissa shifted right by (17 + max_exp - exp), capped at 31, **rounded to nearest even** (`to_v64bfp16ebs8` under
conv_even). On pre-rounded input the shift is exact except where a block's -128*step value (= -2^(E+1)) lifts the shared
exponent by one; the other values then lose their LSB and it must round as the core does (truncating it: 16.6% of outputs differ).
`host/pack_bfp16.hpp` is the scalar reference; the engine's vectorised packer is `vision_bfp16.hpp` (`pack_bfp16_rows`).

    x: [M/64 tile][K/384][4 groups of 16 rows][48 k8 pairs][2 halves (rows 0-7, 8-15)][8 rows][9 B]
    w: [N bias, bf16][N/96 strip][K/384][6 groups][48][2][8][9 B]
    y: (M, N) bf16 row-major

A 16-row x 8-k pair is 144 B, the two 72 B A0/A1 (or B0/B1) tiles C's kernel builds in L1, so a column's x for one round and a
row's w for one strip are each one contiguous run, issued as 3456 B pieces (a shim BD dimension wraps at 1024 words).

## Kernel / sequence
`kernel/mm.cc`: C's mac loop on L1 chunks that are already tiles (no convert, no converted-tile buffer). `host/seq_gen.hpp`:
C's mapping, byte offsets; bias prologue sends one memtile w buffer read linearly from the strip's bias (first 2n bytes used).
RTP address 44416 (C's is 65024); the engine's `vision_mm_rtp_addresses::p` must equal `build/VISION_MM_P/rtp_addr.txt`.

## Measured
Standalone (`run.exe -L 4096 --Ref 0`, HX 370, each design timed in its own block; alternating xclbins per call times the
hw-context switch): all shapes bit-identical to C, redispatch identical, pad inert.

| shape (M=4096) | C us | P us | |
|---|---|---|---|
| qkv 1152->3456 | 3977 | 2850 | -28% |
| proj 1152->1152 | 1339 | 1018 | -24% |
| fc1 1152->4608 | 6704 | 5616 | -16% |
| fc2 4608->1152 | 4277 | 3086 | -28% |
| merger.fc1 4608->4608 (M 1024) | 4730 | 3478 | -27% |
| vit_merger.fc1 4608->17280 (M 1024) | 16634 | 12428 | -25% |

Against the fp32 reference (L=1024): qkv / proj cosine 0.999999, fc1 0.999977 (as C), rotated-weight control ~0.
Raw (not pre-rounded) input is NOT C's bits (diagnostic only).

Engine, N=120 frames 398x224, chunk 4096, warm: tower 12.07 -> 10.72 s, total 14.92 -> 13.55 s, tower hash e8758f49b8b30383
and text md5 0eeb548a unchanged. Device vision_mm 5864 -> 4738 ms; host x pack 715 ms (narrow+preround was 403).

## Build / test
```
source <CI mlir-aie + pinned Peano env>
make bitstream            # build/VISION_MM_P/xclbins/vision_mm_p.xclbin (+ rtp_addr.txt)
make host                 # needs ../vision_mm_c/build/VISION_MM_C for the C side of the comparison
./run.exe -L 1024         # P vs C memcmp, cosine vs fp32 reference (slow), redispatch, pad, timing
./run.exe -L 4096 --Ref 0 # skip the CPU reference;  --Only <shape>, --Preround 0 (diagnostic)
```
