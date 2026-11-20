<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# dequant_mm_prefill (Qwen3.5-2B prefill matmul)

Every projection of the Qwen3.5-2B prefill: `y = x W^T` with Q4_K weights, L a multiple of 512, Din a multiple of
256, Dout a multiple of 512. Started as a copy of `FLM_Xclbin/dequant_mm_512x512x512` (the shared `mm.xclbin`);
its output is **bit-identical** to that kernel on every 2B shape, at about half the time.

Installed as `FastFlowLM/src/xclbins/Qwen3.5-2B-NPU2/mm.xclbin` (CI artifact `xclbins-Qwen3_5_2B_dequant_mm_prefill`).
The engine (`FLM_DLL/detail/qwen3_5vl_npu`) drives it through its `kernels_2b` path, chosen from the model shape;
the other Qwen3.5 sizes keep the shared kernel.

## Build and test

```bash
source ../../setup.sh   # FLM_Xclbin/setup.sh
make bitstream          # -> build/final/xclbins/dequant_mm.xclbin (Chess)
make host               # -> run.exe
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib ./run.exe
```

`run.exe` runs every 2B shape at L = 10752 (`-L`), or one shape (`-s Din,Dout`), `-i` timed calls, `-c` rows checked
against the CPU, `-x` xclbin. It makes the bfp16 x tiles on the host like the engine does, checks against a CPU
matmul, and prints per shape the time, ms per prefill (weighted by how often a 2B prefill calls the shape) and a
hash of the whole output. **Same hash = bit-identical output**; that is how every change was checked. Hashes at
L 10752: qkv `4d44161197077afa`, 2048x2048 `a89aac01d9b89f3d`, up|gate `d39f2a499d78d175`, down
`24dd0bc7fe7a1d63`, k / v `a9ffd05ffc6463a8`.

Before timing, make sure nothing else uses the NPU (look for `accel` in `/proc/*/fd`): another process on it makes
calls take seconds and looks exactly like a deadlock.

## How it works

Each core owns 128 rows x 64 output columns; the two cores of a column pair (even / odd row) share one weight tile.

- **Weights**: Q4_K blocks of 32 rows x 256 k arrive per pair. The even core dequantizes rows 0-15, the odd core
  rows 16-31 (`_Q4_K_bfp16<ROW>` in `q4_core.h`; `ROW` is a template parameter, a runtime row put a branch in the
  tile loop and stopped Chess from pipelining it), straight into one 64-column x 256-k tile in **bfp16**
  (18 KB, odd core memory). A lock barrier between the two cores separates "tile written" from "tile used".
- **x arrives already as bfp16 tiles**, made on the host (`cpu_func::_bfp16_tile_group` in the engine,
  `x_to_bfp16_tiles` in the test): `[16-row group][Din / 8][2 row blocks][72 B]`, a tile being 8 rows x 8 k, each
  row one 9-byte block (shared exponent byte + 8 two's complement mantissas). The rounding is the AIE2P one,
  truncation: `q = signed 24-bit mantissa >> (17 + max_exp - exp)`, the shift capped at 31.
- **Multiply**: `mac_8x8_8x8T` on block vectors popped from block-vector input streams (`mm_aie_bfp16`), k in passes
  of 256. Why bfp16 everywhere: the kernel is built with `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`, so a bf16
  `mmul::mac` converts both operands to bfp16 and then runs this same instruction. Converting each tile once
  (weights on the core, x on the host) gives exactly the same numbers without the repeated conversions.
- **Only two streams**: a core has two block-vector FIFO register sets; a third input stream was spilled to the
  stack every iteration (12 cycles for 4 VMACs). The weight tile stores the two column tiles used together next to
  each other, so one W stream and one x stream feed the loop, one VMAC per cycle.
- **y**: fp32 accumulators per core (128 x 64), sent out as bf16 in 512-value chunks. In the mem tile each
  (core, y buffer) has its own free / full lock and each core's tile goes out as its own BD, so a fast core never
  waits for the others; the shim writes `{4 cores, 128 rows, 64 cols}` with strides `{64, Dout, 1}`.
- **Lock trick**: the weight buffer's "full" lock is one counter for ping and pong, +2 per block (one per core).
  Taking two blocks before the first barrier let one core take both counts of the first block while the second was
  still arriving; a barrier between the two takes fixes that.
- Memory: both cores at ~64 KB with 2 KB stacks. RTP addresses (`input_with_addresses.mlir`): even rows
  `K_div_k` 55552 / `rtp_act` 55616, odd rows 64512 / 64576.

The shim sends x per core as one 4-D byte transfer, sizes `{Din/256, 128/16, 9, 512}`, strides
`{4608, Din*18, 512, 1}`; weights and y as in the shared kernel (`host/seq_gen.hpp`, a copy of the engine's
`gen_dequant_mm_512`).

## Results (Krackan test machine, idle NPU)

ms of mm per 2B prefill at L = 10752:

| shape (Din x Dout) | calls | shared `dequant_mm_512x512x512` | this kernel |
|---|---|---|---|
| linear qkv 2048x6144 | 18 | 500.7 | 254.3 |
| g / o / q 2048x2048 | 54 | 533.7 | 233.4 |
| mlp up+gate 2048x12288 | 24 | 1319.4 | 686.3 |
| mlp down 6144x2048 | 24 | 621.0 | 258.8 |
| attn k / v 2048x512 | 12 | 28.8 | 12.5 |
| **total** | | **3003.6** | **1445** |

## How it got here (2026-11-02)

Each step kept the hashes; 10K TTFT is the whole engine at that point.

| step | change | mm per 10K prefill | 10K TTFT |
|---|---|---|---|
| shared kernel | weights to bf16 32 columns at a time, so the mem tile sent every x chunk twice | 3,004 ms | 7.18 s |
| 1 | weights 64 columns x 128 k: each x chunk sent once | 2,744 ms | 6.77 s |
| 2 | bfp16 tiles converted once instead of inside every emulated multiply; two streams only | 2,419 ms | 4.51 s |
| 3 | y per (core, buffer) locks, 512-value y chunks (a profile build showed the cores waiting <1.5% for data, i.e. compute bound, and y queueing behind the other cores) | 2,230 ms | 3.65 s |
| 4 | `ROW` as a template parameter (no branch in the dequant loop) | 2,014 ms | 3.47 s |
| 5 | 256-k passes (half the fixed work per column pair) | 1,819 ms | 3.34 s |
| 6 | x as bfp16 tiles from the host (no x conversion on the core, 44% fewer x bytes) | 1,425 ms | 3.07 s |

Tried and dropped (correct, slower): 64 rows x 128 columns per core (2,577 ms), the same with 64-k passes and 4-deep
buffers (3,319 ms), deeper x buffers (2,565 ms). Less x per MAC lowers the data floor, but the weight conversion is
then shared by fewer rows and the per-pass overhead grows; the math added on top instead of hiding under the data.
Fewer, merged mm calls were not tried: the time per call scales exactly with the work, so there is no fixed cost to
save.

## Other changes from the copied folder

- `makefiles/host.mk`: `HOST_COMMON_DIR` is `../../host_common` (one folder deeper).
- `Makefile`: `MODEL_TYPE` defaults to `final` (the build goes to `build/final/`).
- `host/host.cpp`: rewritten for the 2B shapes, timing, per-call times, output hash.

## Compiler (2026-11-03): Peano

Default is Peano (`ENABLE_CHESSCC ?= 0`, user decision; `ENABLE_CHESSCC=1` builds Chess) (Peano = `llvm-aie 22.0.0.2026090201+a36c62b9`; the June 21.0 used before is older.). Peano output is bit-identical (same hashes, engine logits identical)
but the mm is **24 % slower**: 1814-1821 ms vs 1457-1476 ms per prefill at L 10752. Peano pipelines the k loop at an
initiation interval of 8 cycles for the 4 `mac_8x8_8x8T` (Chess ~1.65 cycles per mac). Giving Peano the loop hints
(`AIE_PREPARE_FOR_PIPELINING AIE_LOOP_RANGE`) changed nothing; closing the gap needs the loop rewritten for Peano.
With GDN also on Peano, 10K TTFT went 2.94 -> 3.24 s (prefill 3507 -> 3178 tok/s), decode unchanged.
Its ping / pong flags are zero-initialized and persist across calls; Peano puts those in `.bss`, which its CDO loader does not zero, so they are marked `AIE_PERSIST_ZERO` (`.data` under Peano, nothing under Chess, see `kernel/aie_kernel_utils.h`). Outputs unchanged with the fix.
Tried to close the Peano gap with a split bfp16 layout (64-byte mantissas, then 8-byte exponents, read with plain
aligned loads instead of the block-vector streams). Bit-identical, but slower: x and w split 3362 ms (the dequant then
writes 8-byte exponents through scalar registers and is not pipelined), x split with w still streamed 2343 ms, vs
1842 ms streamed. Under Peano the exponent half of a bfp16 register only loads fast through the stream pop
(`vldb.pop.576`); built from memory it takes scalar moves (13 per two steps). The dequant tile loop is not
software-pipelined by Peano either (minimum II 31 > the default 27; `-mllvm -pipeliner-max-mii=64` did not change the
total). Banks: a pinned layout with x and w in different banks gave -1.3 % (not kept).

## MiniCPM-OCR copy (2026-11-05)

This folder is a copy of `origin/main:FLM_Xclbin/Qwen3_5_2B/dequant_mm_prefill` (8688e6aa8; first commit unchanged).
It is MiniCPM-OCR's prefill matmul: `mm_bfp16.xclbin` / `mm_bfp16_glu.xclbin` in the engine
(`FLM_DLL/detail/minicpm_ocr_npu`, README "Prefill mm on bfp16 tiles"). The old `FLM_Xclbin/minicpm_ocr/dequant_mm`
(MiniCPM-OCR's `mm.xclbin`) is untouched.

### Build and test

```bash
source /tmp/mco_peano_env.sh        # the CI Peano, see ../README.md "Peano"
make bitstream                      # -> build/final/xclbins/dequant_mm.xclbin, ~5 s; checks the RTP addresses
make host                           # -> run.exe
./run.exe -L 2048 --old ../dequant_mm/build/xclbins/dequant_mm.xclbin    # 7 shapes, bit compare vs the old mm
./run.exe -L 2048 --glu                                                  # the GLU epilogue check (below)
```

`run.exe` now runs the 7 shapes the engine uses (1024 -> 6144, 1024 -> 2048 with SiLU, 2048 -> 1024, 1024 -> 2048,
1024 -> 512, 1024 -> 7168, 3584 -> 1024; `-s Din,Dout[,act]` for one). `--old <xclbin>` also runs the old mm on the
same weights and x (x as bf16 rows, with the old sequence in `host/seq_gen_old.hpp`) and prints a `COMPARE` line per
shape: IDENTICAL means every output value has the same bits. RTP addresses: K_div_k 55552 / rtp_act 55616 on even
rows, +8960 on odd rows (the 2B's); `makefiles/bitstream.mk` stops the build if they move.

The kernel needed no change for MiniCPM-OCR: every shape has Din a multiple of 512 (3584 = 7 x 512) and Dout a
multiple of 512, and L is padded to 512 by the engine. All MiniCPM-OCR kernels are Peano; this one too. On the 2B,
Peano was 24% slower than Chess for this kernel (README above); Chess was not built or timed here.

### Results (Strix Halo test machine, Peano, idle NPU)

Bit-identical to the current `../dequant_mm` build (and to the shipped Chess `mm.xclbin`, checked at L 2048) on all
7 shapes at L 512 / 2048 / 8192 (21 / 21 `IDENTICAL`). ms of mm per MiniCPM-OCR prefill (each shape's call time
weighted by how often a prefill calls it, `run.exe` total line, 5 timed calls):

| L | old `dequant_mm` (Peano) | this kernel | |
|---|---|---|---|
| 512 | 75.0 | 61.7 | -18% |
| 2,048 | 256.7 | 175.9 | -31% |
| 8,192 | 945.1 | 598.5 | -37% |

Per call at L 8192: qkv 1024 -> 6144 12.2 -> 8.1 ms, up/gate 1024 -> 7168 14.4 -> 9.5 ms, down 3584 -> 1024 5.9 ->
3.8 ms. At L 512 the big shapes gain little (qkv 0.90 -> 0.84 ms, up/gate 1.03 -> 0.89 ms): the weights are read once
per 512 rows, so a 512-row call is mostly weight reading.

## GLU epilogue (rtp_act 2, 2026-11-05)

In the engine the up/gate projection used to write up and gate (in 512-wide slices) to DDR; the host read them back,
computed `silu(gate) * up` and wrote `hid` (or its bfp16 tiles) for the down projection. With `rtp_act = 2` the array
does that itself and writes `hid` straight in the down projection's x tile format. Acts 0 and 1 are unchanged
(still bit-identical to the old mm on all 7 shapes). Program memory went from 3.9 to 4.6 KB per core.

**Pairing up and gate on one core.** A core's output tile is 128 rows x 64 columns, accumulated in fp32 (`y_acc`,
32 KB). Holding a second tile (up and gate of the same 64 features) does not fit: the odd core already uses 63 of its
64 KB (W tile 18 KB, x 9 KB, y 2 KB, `y_acc` 32 KB, stack 2 KB). So each core makes 32 features instead of 64: the
host sends its core pair the 32 up rows' Q4_K block and then the matching 32 gate rows' block for every 256 k (they
are 512 rows apart in the weights, which the engine stores as [up 512 rows][gate 512 rows] per slice). The kernel's
K loop does not change: block p = 0 lands in columns 0-31 (up), p = 1 in columns 32-63 (gate). A round still
covers 8 core columns, now 8 x 32 = 256 features, so the up/gate call has the same number of rounds as before.
The weight DMA is 4 dimensions (`gen_dequant_mm_glu`): per 256 k, {up, gate}, each block in two halves of 2368 B,
because one DMA dimension wraps at 1024 words and a block is 1184 words.

**The math.** Per 8 x 8 block: up and gate are rounded to bf16 exactly as the plain output is (the kernel's default
rounding mode), the SiLU is the act 1 one (`getSiluBf16_nonLUT`, hardware tanh), and `up * silu` is rounded to
bf16 with round-to-nearest-even (the host GLU rounds the same way; the mode is put back right after, so acts 0 / 1
do not change). The bf16 block is then turned into a bfp16 block vector, which is the x tile format (8 rows x 8 k,
each row one block). This is the conversion the 2B kernel showed to equal the host's `tile_group`, so the only
difference from the host path is the SiLU.

**Where the tiles go, without a memory barrier.** The tiles (4608 B per core tile) are written over `y_acc` while it
is still being read. Group g (16 rows) reads bands 2g and 2g + 1 (bytes 4096 g ..), and its tiles go to bytes
[(g - 1) x 576, g x 576): only bands that are already read. Group 0 has no such space, so its 576 B go to a local
array on the stack (the kernel frame grows 704 -> 1280 B, under the 2 KB stack). No write can then land on data still
to be read, in any order the compiler picks. (A compiler barrier would have been the other way, but Peano cannot
compile `asm volatile("" ::: "memory")`: "unable to translate instruction: call".)

**Why a 16 KB slot per tile.** The mem tile's y buffers are fixed in the design: each core's tile is one 16 KB BD
that reorders 8 x 8 blocks into rows. The GLU tile is only 4608 B, so the core still sends 16 chunks of 1 KB; it
writes the tile bytes into the first chunks in the order that the mem tile's reorder turns into a plain byte
stream (stream bytes `t * 1024 + r * 128 + cb * 16` go to chunk bytes `cb * 128 + r * 16`), and the rest is filler.
The shim writes each core's 16 KB to its own slot, `(row block * (IS / 32) + feature slot) * 16 KB`; a shim's 4 cores
are 4 neighbouring slots, so that is one linear 64 KB write. (Writing the tile twice into the same place with a
zero iteration stride is not possible: the sequence helper turns a zero iteration stride into a single pass.) The
slots of one call take as much DDR as the old up/gate output (L x 7168 x 2 B). The down projection reads the slots
with its own x pattern (`x_from_glu_slots`): per 256 k, 8 groups x the 8 slots' 576 B, the same stream as row tiles.

**Checks** (`run.exe --glu -L <n>`; 1024 -> 7168 -> GLU -> 3584 -> 1024 on random weights):
- down from the slots vs down from the same bytes moved to row tiles: **IDENTICAL** at L 512 / 2048 / 8192.
- hid as the down projection sees it, vs the host GLU's bf16 hid: host path's tiles (only the bfp16 rounding) rel L2
  1.50e-2; GLU-epilogue tiles 1.86e-2 (cos 0.99983); the two tile sets differ by rel L2 1.34e-2. The down output
  of the two paths differs by rel L2 1.36e-2 (cos 0.99991). Same numbers at all three lengths.
- The extra error is the hardware-tanh SiLU, the same one the z projection (act 1) ships with: on the same 1024 ->
  2048 data act 1 has NMSE 3.4e-4 vs the CPU reference against 1.3e-4 without the SiLU. A version that summed
  `(up / 2) g + (up / 2) tanh(g / 2) g` in fp32 (fewer bf16 roundings) was not better (1.90e-2), so the tanh itself
  is the limit; an fp32 exp-based sigmoid would be needed to do better (not done).
- up/gate call time, act 0 vs act 2: L 512 735 vs 663 us, 2048 2534 vs 2338 us, 8192 9332 vs 9287 us (act 2 skips the
  bf16 output pass, which pays for the GLU math).
