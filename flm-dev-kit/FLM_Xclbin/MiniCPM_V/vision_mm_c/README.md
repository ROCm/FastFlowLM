<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# vision_mm_c

MiniCPM-V vision-tower matmul, design C: `y = act(x W^T + bias)` for every tower matmul whose N is a multiple of 384
(qkv 3456, proj 1152, fc1 4608, fc2 1152, vit_merger.fc1 17280, merger.fc1 4608, patch embed 1152). Optional
`vision_mm_c.xclbin`: the engine uses it when the model folder has it (`FLM_MINICPM_V_VISION_MM_C=0` turns it off) and
otherwise stays on `Qwen3_8/vision_mm` A / B. **Bit-identical to design A** (checked on plain, GELU and SiLU epilogues and
odd row counts) because the host pre-rounds both operands onto the bfp16 grid and the multiply forms the same products in the
same order. 1.2-1.6x faster from 512 rows up; slower at 256 rows (512-row quantum), so the engine keeps A there.

## Mapping
Columns split M (8 x m 64 = 512 tokens), rows split N (4 x n 96 = 384 channels). x: memtile c holds column c's tokens,
broadcast down the column; w: memtile i (i < 4) holds core row i's channels, broadcast along the row; the four row outputs of a
token join in the memtile into one (m, 384) block of full token rows, so the shim writes 768 B row segments. The old mapping
(rows split M, n 48) wrote 96 B pieces at a row stride, re-streamed x and w per 256 tokens and cost ~28 host commands per round.
DDR layouts of x, w, y are unchanged.

## Kernel
Operands reach L1 as `[k8][16 rows][8 k]` (x) and `[k8][n][8 k]` (w), so every 64-element load is an 8x8 A tile or a "T"-form
B tile. Each chunk is converted once into bfp16 tiles (`to_v64bfp16ebs8`) and the multiply runs block-vector streams
(`mac_8x8_8x8T`), no per-use conversion, no W transposes. L1: acc 24 KB, w 2 x 12 KB, x 2 x 2 KB, converted tiles 8 KB.

## Host sequence (`host/seq_gen.hpp`)
One BD per (strip, operand) covers up to 32 consecutive rounds (shim BD repeat). The repeat strides of x and y must be below
2^21 bf16 elements: a larger stride does not fail, it wraps and the output is wrong (uncorrelated with the reference), so such
shapes run one round per group.

## Measured (kernel only, HX 370, vs shipped design A)
8192x1152x3456: 12.0 -> 7.0 ms; 8192x4608x1152: 11.1 -> 8.0; 1024x1152x3456: 1.62 -> 1.00; 256 rows: 0.48 -> 0.58 (slower).
Sending 9/16 of the x or w bytes changes nothing now, so bfp16-packed operands would not help; the zero-core-work floor is 5.9 ms.

## Build / test
```
source <CI mlir-aie + pinned Peano env>
make bitstream            # build/VISION_MM_C/xclbins/vision_mm_c.xclbin (+ RTP address in build/VISION_MM_C/rtp_addr.txt)
tools/extract_rtp_addr.sh VISION_MM_C   # the engine's vision_mm_rtp_addresses::c must equal it (65024)
make host && ./run.exe    # shapes against an fp32 reference (cosine, rotated-weight control)
```
