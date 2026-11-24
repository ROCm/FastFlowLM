<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# dequant_mm_bfp16_rtm2_glu_512x512x512

`dequant_mm_bfp16_rtm2_512x512x512` (parallel memtile y join, runtime row count, bit-identical to the fixed bfp16 design for every plain
dispatch) **plus** the fused GLU epilogue of `dequant_mm_bfp16_rtm_glu_512x512x512` (`rtp_act = 3`: y = silu(gate) * up). It is a drop-in
superset of `mm_rtm2.xclbin` (ships as `mm_rtm2_glu.xclbin`), so one mm hw context serves plain, reduced-row and GLU dispatches. Created by
applying the rtm -> rtm_glu changes (kernel epilogues, host sequence, `check_memtile_bds.py`, `glu_test`) on the rtm2 design.

The text below the line is the rtm2 README; the GLU description follows it.

---

# (rtm2 part)

`dequant_mm_bfp16_rtm_512x512x512` (runtime row count per dispatch, same host sequence layout, same BD ids and
RTP addresses, same multiply) with the core-side and memtile-side stalls removed that an in-kernel stall split found.
Results are **bit-identical** to the rtm / fixed-m design on every projection shape; it ships as `mm_rtm2.xclbin`.

What changed, in the order of what it bought (qkv 1024 -> 6144 at L = 2048, per 512-row round per core, cycles at 1.81 GHz):

1. **Parallel memtile y join** (`iron/dequant_mm_tile.py`, `dequant_mm_memtile`). The four cores of a memtile join
   delivered their 16 KB output tiles through a chain of locks (core 0 writes, then releases core 1's channel, ...), so the tiles
   drained one after the other: the epilogue of columns 0..3 took 3.5 k / 10.9 k / 18.3 k / 25.5 k cycles and, because the
   x broadcast of a core row is lockstep, every column waited for the last one (x waits of columns 0..3: 23.7 k / 16.4 k / 9.0 k / 2.0 k cycles
   per round). Now each S2MM channel takes one of 4 producer credits of a y buffer and adds one consumer credit; the output
   DMA waits for 4 credits and gives 4 back. Locks: 5, 6 = producer credits (init 4), 7, 8 = consumer counters (init 0); lock 9 is gone.
   Round period 94 k -> 72 k cycles (instrumented build); epilogue 3.5 k on every column; x waits 2 k.
2. **No accumulator zeroing** (`kernel/dequant_mm_bfp16.h`, `mm_aie_bfp16<FIRST>`): the first k slice of a call starts its
   four accumulators from zero instead of loading a zeroed 32 KB `y_acc` (3.1 k cycles per round).
3. **Ping/pong flags in registers** (`kernel/dequant_mm_{e,o}.cc`): every call toggles each flag an even number of times, so
   they are `false` at every call start; the global byte flags cost a load/store/xor sequence with ~30 idle cycles per toggle.

Host side: `host/seq_gen.hpp` (`MT_Y_LOCKS` = {5, 6, 7, 8}, `MT_Y_LOCK_INIT` = {4, 4, 0, 0}) re-arms the y locks for a reduced-row
dispatch; the MiniCPM-V engine has the same arrays (`minicpm_v_4_7_npu_sequence::mm_rearm`, `mm_yjoin_par`) and prefers `mm_rtm2.xclbin`
over `mm_rtm.xclbin` when the file is there. `check_memtile_bds.py` checks the new lock values in the built CDO.

## Stall split (how the numbers above were found)

Peano has no cycle counter intrinsic, and a design whose memtile joins 4 core rows has all 4 southbound channels of every column
in use, so mlir-aie's trace flow cannot route even one core trace (`Unable to find a legal routing` for every core tile).
What works: the core reads its own timer through the processor bus.

* the host sequence writes **Core_Processor_Bus = 1** (tile register 0x32038) on every core (`-DDIAG_BUS_EN`),
* the kernel reads **Timer_Low** (0x340F8) with `read_tm()` (aie2pintrin.h) at every phase boundary (`-DDIAG_TS`, `kernel/diag_ts.h`),
  keeps 32-bit base + saturating 16-bit deltas in a static buffer (the top 1.4 KB of L1 are free), and overwrites the last y chunk of
  the call with them (`y` is garbage in this build),
* `-DDIAG_MAP` writes `chunk * 256 + element` into every y element so `tools/ts_test map` can learn where chunk elements land in the y matrix,
* `tools/ts_test ts` decodes the timestamps of all 32 cores x all rounds; `tools/ts_analyze.py` aggregates them per phase.

The first timer read of the first call must come after the host has enabled the bus (before it the core hangs): the kernel's first sample
of a call is guarded by a flag set at the end of the first call, all other samples come after a lock acquire that only completes once the
dispatch's sequence has run. The instrumented build runs ~5-10 % slower than the real kernel.

Per round budget of the old design (cycles per 512-row round and core, qkv L = 2048; 1 round = 4 k slices of 256):
MAC 33.4 k (2.0 cycles per 8x8x8 MMUL; the loop's steady state is 1.125), dequant 13.4 k (2 x 1.67 k per k slice), x conversion 11.1 k
(8 columns convert the same x), accumulator zeroing 3.1 k, epilogue 3.5 k (+ up to 22 k waiting for the serialised y join), x waits 5.6 k
(col 0: 9.3 k, col 3: 1.9 k), barriers + weight waits 1 k. The cores are lockstep on the x broadcast, so the slowest column sets the pace.

## Results (HX 370, XRT 2.25, 8 columns x 4 rows, kernel only; `tools/ab_bench`, per-side minimum over 140 interleaved dispatches because other NPU users time-slice the context and inflate medians)

| shape (Din -> Dout) | L | rtm (us) | rtm2 (us) | gain |
|---|---|---|---|---|
| qkv 1024 -> 6144 | 1024 / 2048 | 1101 / 2050 | 988 / 1867 | 1.11x / 1.10x |
| up/gate 1024 -> 7168 | 1024 / 2048 | 1263 / 2370 | 1197 / 2170 | 1.06x / 1.09x |
| down 3584 -> 1024 | 1024 / 2048 | 575 / 1062 | 557 / 1022 | 1.03x / 1.04x |
| attn_o 2048 -> 1024 | 1024 / 2048 | 383 / 690 | 364 / 656 | 1.05x / 1.05x |
| attn_q/g, z 1024 -> 2048 | 1024 / 2048 | 440 / 752 | 393 / 676 | 1.12x / 1.11x |
| z silu 1024 -> 2048 | 1024 / 2048 | 473 / 839 | 389 / 669 | 1.22x / 1.25x |
| attn_k/v 1024 -> 512 | 2048 | 266 | 245 | 1.09x |

* Bit-identical outputs to rtm on all 18 shape/L combinations of `gemm_shapes_bench` (`DUMP_PREFIX`), `rtm_test` 504/504 (21 row counts x 8 shapes, shuffled, 3 reps),
  soak of 400 dispatches per shape/L on qkv, up/gate, down, attn_k with 0 mismatches.
* MiniCPM-V engine (`mm_rtm2.xclbin` vs `mm_rtm.xclbin`, same engine library, server TTFT median of 3 rounds x 3-5 reps): 24 tokens 78.7 -> 77.8 ms (1.2%), 264 tokens 123.2 -> 121.5 ms (1.3%),
  1019 tokens 260.8 -> 251.7 ms (3.5%), 4090 tokens 824.9 -> 793.1 ms (3.8%), 16379 tokens 3614 -> 3472 ms (3.9%); decode unchanged; greedy outputs of the 17-prompt identity set identical.

### Where the kernel is now (qkv, L = 2048, cycles per 512-row round and core)
Period 67.6 k (37.3 us; 1744 us per dispatch): MAC 34.2 k (2.0 cycles per 8x8x8 MMUL), dequant 13.4 k, x conversion 10.1 k, epilogue 3.0 k, x waits 1.9 k,
accumulator zeroing 23. The no-compute variant of this design (`-DDIAG_NO_MAC -DDIAG_NO_DEQ -DDIAG_NO_CONV`) takes 1730 us: **the kernel is at the data-movement floor and at its
core-chain limit at the same time** (87 MB of DDR traffic at ~50 GB/s: x 55% because it is re-read per 512-column n round, y 29%, w 17%), which is why removing a single cost gives <= 5%.
Timing-only bound of the next step, bfp16 x panels (x at 9/16 of the bytes and no in-core conversion): qkv 1744 -> 1504 us, up/gate 2031 -> 1749 us, down 1023 -> 862 us at L = 2048 (-14 to -16%).
Not helpful (measured): padding the y row pitch (DRAM channel alignment of the 512-byte y rows) leaves the floor unchanged.

## Build and test

```
source <env with CI-pinned Peano + mlir-aie>
make bitstream                 # build/xclbins/dequant_mm.xclbin  (+ check_rtp.py, check_memtile_bds.py)
tools/build.sh                 # HC=<host_common> NLOH=<dir with nlohmann/json.hpp>
tools/gemm_shapes_bench <xclbin> [iters] [shape-filter] [noacc]      # per-shape timing; DUMP_PREFIX=<p> dumps outputs (bitwise compare)
tools/ab_bench <xclbinA> <xclbinB> <iters> <rounds> [filter] [Ls]    # interleaved A/B of two xclbins (blocks alternate; robust against other NPU users)
tools/rtm_test ref <fixed-m xclbin> <dir>; tools/rtm_test rtm <this xclbin> <dir> [reps] [shape]   # reduced-row dispatches bit-exact vs the fixed kernel
# stall split:
make bitstream EXTRA_KARGS=-DDIAG_MAP   -> xclbin A ; tools/ts_test A map 1024 6144 2048 map.bin
make bitstream EXTRA_KARGS=-DDIAG_TS    -> xclbin B ; tools/ts_test B ts  1024 6144 2048 map.bin ts.csv ; python3 tools/ts_analyze.py ts.csv
```

`DIAG_NO_MAC` / `DIAG_NO_DEQ` / `DIAG_NO_CONV` turn the multiply, the dequant or the in-core x conversion into a no-op for bound analysis
(outputs are garbage then).

Like the other bfp16 designs this was built locally with a newer mlir-aie than CI's and the CI-pinned Peano; the CI-built xclbin must
be re-verified (bit-identical outputs of `gemm_shapes_bench` with `DUMP_PREFIX`, `rtm_test`) before it ships.


## How it works

* **Weights.** A core of the array owns a pair of 32-row weight blocks (its two n slices). The model stores the gate/up projection as
  slabs `[up 512 rows | gate 512 rows]`, so up and the matching gate sit in different cores. `glu_repack()` (host/seq_gen.hpp; the engine has
  its own copy, `minicpm_v_glu_repack`) builds a second copy in which pair `P = 8 (2 slab + half) + c` is
  `[up block 32 slab + 8 half + c | gate block 32 slab + 16 + 8 half + c]`: n slice 0 = up channels, n slice 1 = the gate channels of the same
  32 hidden channels. A pass of 512 weight rows then yields 256 hidden channels, core c channels `32 c .. 32 c + 31`. Pure block copies,
  4.2 MB per layer; the engine keeps the copy behind the regular weights of every layer's weight BO (decode keeps reading the original).
* **Kernel.** After the K loop the accumulator holds `[m16][m8 row (2)][n8 tile (8)][64]`; tiles 0-3 are up, 4-7 the gate of the same
  rows/channels. Per m8 row group: round both to bf16, `silu(gate) = gate (1 + tanh(gate / 2)) / 2` on the vector tanh, multiply by up, store one
  256-element chunk (half of the plain path's two). Every narrowing in the epilogue rounds to nearest even (the design's default, floor,
  biases a chain of six roundings by about -1%); the mode is put back before the next dispatch.
* **Descriptors.** A core now sends 32 instead of 64 columns per row. Three words of the memtile y descriptors differ from the CDO's (S2MM D2 wrap
  8 -> 4, MM2S D0 wrap 32 -> 16 words, the lengths halve); strides and base addresses do not (a core's region keeps its 64-column pitch).
  `gen_dequant_mm_rtm` (`rtp_act == 3`) re-arms the memtile y channels with them before the dispatch and puts the design's configuration back after
  it (`rtm_rearm(..., RTM_Y_GLU / RTM_Y_RESTORE)`), exactly like the reduced-row dispatches. The shim y descriptors become `m x 128` per memtile
  group at row pitch `Dout / 2`.

## Numerics

The result is not bit-identical to `plain dispatch + host GLU` (tools/glu_test.cpp): the array's tanh is bf16 and the product rounds again.
Against an fp64 reference from the dequantized weights (random N(0, 0.03) weights, N(0, 1) x, 1024 -> 7168):
plain dispatch + host GLU 2.16% rel-rms (the array matmul's own bf16 / bfp16 error), fused 2.63%. An exp2 + reciprocal variant is 2.38% but the
reciprocal is a per-element scalar op (+25 us per 512-row pass), so the tanh form is used. The engine flag `FLM_MINICPM_GLU_FUSED` (default on; `=0` switches it off).

## Build and test

```
source <env with CI-pinned Peano + the CI mlir-aie>   # check_memtile_bds.py accepts both CDO names (aie_cdo_init.bin / main_aie_cdo_init.bin)
make bitstream                # build/xclbins/dequant_mm.xclbin (+ check_rtp.py, check_memtile_bds.py)
tools/build.sh                # HC=<host_common> NLOH=<dir with nlohmann/json.hpp>
tools/rtm_test rtm build/xclbins/dequant_mm.xclbin <ref dir> [reps]   # plain dispatches stay bit-exact vs the fixed 512-row kernel (as the rtm design)
tools/glu_test build/xclbins/dequant_mm.xclbin [reps]                  # fused vs plain + host GLU vs fp64; fused/plain interleaved in shuffled order
                                                                         # (GLU_ROWS=a,b,c  GLU_DIAG=1 for the error by |gate|)
```

Built locally with the CI mlir-aie (old API) and the CI-pinned Peano; the CI-built xclbin must be re-verified with `rtm_test` and
`glu_test` before it ships.
