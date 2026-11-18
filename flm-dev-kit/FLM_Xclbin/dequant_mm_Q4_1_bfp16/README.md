<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# dequant_mm_Q4_1_bfp16

An 8-column prefill GEMM for the FLM q4nx **Q4_1** weights: `Y[L, Dout] = act(X[L, Din] @ W[Dout, Din]^T)`, bf16 out, with the
weights read in their packed form (no bf16 copy in DDR) and both operands multiplied as native block float (`bfp16ebs8`).
It builds on the structure of `dequant_mm_Q4_1_512x512x512` (same weight layout, same core pairs, same host call), with
these differences:

| | `dequant_mm_Q4_1_512x512x512` | this kernel |
|---|---|---|
| weight tile for the mmul | bf16, converted to block float inside every multiply | converted **once** by the dequant core, stored as `bfp16ebs8` |
| X (activations) | bf16 rows, re-tiled by DMA descriptors, converted in the multiply loop | `bfp16ebs8` tiles packed by the host in consumption order, every copy is linear |
| n slices per X tile | 1 (X is sent twice per k slice) | 2 (each X tile feeds both 32-wide slices of the core's 64 outputs) |
| multiply loop | 17 bundles per 4 mmuls (Peano) | 5 bundles per 4 mmuls (`AIE_TRY_INITIATION_INTERVAL`) |
| 4096 x 2560 x 10240 (gate GEMM), Peano | 26.3 ms, 8.2 TFLOPS | 14.1 ms, 15.2 TFLOPS |

Measured on a Ryzen AI Max+ 395 (warm: 30 warm-up calls, median of 100). Accuracy against an fp32 reference of the same
quantized weights: cosine 0.99983 (0.999956 against a reference that also truncates X to block float).

## Constraints

* `Din % 512 == 0`, `Dout % 512 == 0`, `L % 512 == 0` (pad the batch; padding rows are computed and ignored).
* Q4_1 weights in the FLM layout: 64-row slabs, each slab 32 x 256 blocks (see `host/weights_desc.hpp`), reordered with `reorder_cpy`.
* The activation (`rtp_act`) is 0 = none, 1 = SiLU, 2 = GELU, applied to the bf16 output tile.
* Built with Peano (`ENABLE_CHESSCC=0`, the default here); the result was measured with llvm-aie `22.0.0.2026090201+a36c62b9`.
  A different compiler version can change the multiply loop's schedule (check with the loop bundle count in the assembly).

## Layouts the host must produce

**X** (one `Din`-wide row block per core row): the matrix is cut into blocks of 128 rows; for each block, `Din / 512` slabs of
128 rows x 512 k follow each other. A slab is 1024 tiles of 8 rows x 8 k, 72 bytes each (8 blocks of `[shared exponent][8 mantissas]`,
one block per row of the tile), ordered

    [k slice of 256 (2)] [16-row group (8)] [k half of 128 (2)] [k tile (16)] [row tile in the group (2)]

so the core eats each 2304-byte packet front to back. `host/host.cpp` has the reference packer (`pack_x_bfp`). The converter
truncates (the core's own conversion does too).

**W**: the FLM q4nx Q4_1 blocks (as for `dequant_mm_Q4_1_512x512x512`). Inside the odd core of a pair the dequantized tiles are stored
`[n slice (2)][n pair (2)][k tile (32)][low n block, high n block]` so one stream pops both tiles of a pair.

**Y**: bf16, row-major `[L, Dout]`.

## Things learned while building it (Peano / AIE2P)

* Block-float vectors (576 bit) can only be read through the core's two load FIFOs (`aie::block_vector_input_buffer_stream`),
  so a loop can pop from at most **two** streams without spilling; that is why X and W are one stream each and why W is stored pair-interleaved.
* `seek()` on a block stream is **relative** to the current position, and resets the FIFO.
* Only five accumulator registers (`dm0`..`dm4`) are usable: a 2 x 4 mmul blocking (8 accumulators) spills and is slower than 2 x 2.
* `aie::mmul<8,8,8,bfp16ebs8,bfp16ebs8>::mac(a, aie::op_transpose(b))` expects B already transposed ([n][k], k fastest).
* The runtime parameters (`K_div_k`, `rtp_act`) sit at core-local addresses that move whenever a core's buffers change; re-read
  them from `build/<MODEL_TYPE>/bitstream/from_iron/dequant_mm.mlir.prj/input_with_addresses.mlir` after every build and
  update `host/seq_gen.hpp`. A stale address shows up as a hang.

## Build and test

    make -j4 bitstream MODEL_TYPE=<name>          # Peano, ~90 s
    make host MODEL_TYPE=<name> && mkdir -p build/xclbins && cp build/<name>/xclbins/dequant_mm.xclbin build/xclbins/
    KT_DIN=2560 KT_DOUT=10240 KT_L=4096 ./run.exe  # prints the median call time and the accuracy

`KT_DIN`, `KT_DOUT`, `KT_L`, `KT_ITERS`, `KT_WARMUP` override the shape and the timing loop.
