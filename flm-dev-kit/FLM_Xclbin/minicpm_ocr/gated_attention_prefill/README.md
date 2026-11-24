<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# gated_attention_prefill (Qwen3.5-2B prefill attention)

## MiniCPM-OCR copy (2026-11-05)

Copied unchanged from `origin/main:FLM_Xclbin/Qwen3_5_2B/gated_attention_prefill` (8688e6aa8). MiniCPM-OCR's
full-attention layers have the same shape (8 q / 2 kv heads x 256, scale 1/16), so the kernel and its sequence need
no change. Built with Peano (the CI toolchain, `source /tmp/mco_peano_env.sh`); the four hashes below are the same
as on the 2B box, so this build behaves exactly like the 2B one. The engine loads it as `attn_bfp16.xclbin`
(`FLM_DLL/detail/minicpm_ocr_npu/README.md`, "Prefill attention on bfp16 tiles").

Added here:

* `run.exe --old <xclbin>`: also runs the shipped `attn.xclbin` (Qwen3.5's shared kernel: bf16 q and KV cache, 16
  rows per core; sequence copied from the engine into `host/seq_gen_old.hpp`) on the same q / K / V, and prints
  `COMPARE` lines: how many output values are bit-identical, cos, rel L2, max abs, and both kernels against the fp32
  CPU on the checked rows.
* `run.exe --npy <dir>`: real q / K / V from an engine dump (`FLM_DUMP_DIR`, a `DEBUG_LEVEL=1` engine; use one
  attention layer's folder, e.g. `<dump>/layers/03`). `L` is cut to a multiple of 256 that fits the dump.
* `ATTN_SHARED_SOFTMAX` (build flag, test use): the shared kernel's softmax arithmetic in this kernel. **It is
  bit-identical to the shipped `attn.xclbin`**, so the bfp16 tiles and the 32 rows per core change nothing, and
  every difference of the default build comes from the softmax steps below.

```bash
source /tmp/mco_peano_env.sh
make bitstream                                   # build/final/xclbins/attn.xclbin (the default, = attn_bfp16.xclbin)
make bitstream MODEL_TYPE=shared_softmax KERNEL_PEANO_EXTRA=-DATTN_SHARED_SOFTMAX   # build/shared_softmax/...
make host
./run.exe -L 2048 [-B 1000] --old /path/to/FastFlowLM/src/xclbins/MiniCPM-OCR-1.0-NPU2/attn.xclbin
./run.exe -x build/shared_softmax/xclbins/attn.xclbin -L 2048 --old <shipped attn.xclbin>
```

### Against the shipped `attn.xclbin` (Strix Halo test machine, random q / K / V, us per call, median of 3)

| L | `-B` | shipped | this kernel | shared-softmax build | this kernel vs shipped: identical values, cos, rel L2 |
|---|---|---|---|---|---|
| 512 | 0 | 457 | 391 | 429 | 20.2%, 0.99990, 1.4e-2 |
| 2048 | 0 | 4,248 | 1,943 | 3,156 | 10.2%, 0.99975, 2.2e-2 |
| 8192 | 0 | 58,391 | 23,611 | 42,127 | 6.7%, 0.99965, 2.7e-2 |
| 2048 | 100 | 4,669 | 2,150 | 3,573 | 8.6%, 0.99949, 3.2e-2 |
| 2048 | 1000 | 7,675 | 3,342 | 5,762 | 6.2%, 0.99919, 4.0e-2 |
| 512 | 3000 | 3,166 | 1,337 | 2,365 | 5.7%, 0.99915, 4.1e-2 |

The shared-softmax build is bit-identical to the shipped kernel in all six rows (same output hash). Against the fp32
CPU (48 checked rows x 8 heads), this kernel is a little closer than the shipped one in all six rows (e.g. 8192: rel
L2 4.14e-2 vs 4.17e-2; `-B 1000`: 0.2013 vs 0.2029).

**Why this kernel is not bit-identical to the shipped one** (the 2B README's first two steps, "within rounding"):

1. the softmax shift: the shipped kernel rounds `s - m` to bf16 and then multiplies by log2(e); this one computes
   `s * log2(e) - m * log2(e)` in fp32 (one multiply-add) before `exp2`;
2. `c = exp(m_old - m_new)`: the shipped kernel computes it in bf16 scalars; this one in a 16-lane vector with an
   fp32 product, and exactly 1 where the max did not move;
3. lazy running max: the max only moves when a block's max is more than 5.5 above it;
4. the fp32 `y` is only rescaled when some row's max moved (the shipped kernel rescales every block with a `c`
   that is not exactly 1 even when the max stayed).

The `ATTN_SHARED_SOFTMAX` build undoes exactly these four, and nothing else.

Causal prefill attention for the 2B's full-attention layers: 8 q heads, 2 kv heads (GQA 4:1), head dim 256,
scale 1/16. Started as a copy of `FLM_Xclbin/Qwen3_5/gated_attention_prefill` (the shared `attn.xclbin`); about a
third of its time, output within rounding of it (see Results).

Installed as `FastFlowLM/src/xclbins/Qwen3.5-2B-NPU2/attn.xclbin` (CI artifact
`xclbins-Qwen3_5_2B_gated_attention_prefill`). The engine (`FLM_DLL/detail/qwen3_5vl_npu`) drives it through its
`kernels_2b` path, chosen from the model shape; the other Qwen3.5 sizes keep the shared kernel.

**Built with Chess** (`ENABLE_CHESSCC ?= 1`). With Peano, any branch around `calculate_y` made the kernel hang or
give NaN, while the same source built with Chess is correct.

## Build and test

```bash
source ../../setup.sh      # FLM_Xclbin/setup.sh
make bitstream             # -> build/final/xclbins/attn.xclbin
make host
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib ./run.exe -L 10752
```

`run.exe` makes random Q / K / V, turns them into the kernel's bfp16 tiles the way the engine does, runs the kernel
and checks `-c` sampled query rows of all 8 heads against an fp32 causal CPU reference; it prints per-call times and
a hash of the whole output. Options: `-L` query rows (a multiple of 256), `-B n` n rows already in the KV cache
before the query rows (the multi-turn case), `-i` timed calls, `-c` checked rows, `-x` xclbin. Hashes:
L 2048 `38ee823d7414beb7`, L 10752 `279f79387606df60`, L 2048 `-B 100` `aa7e64187da28a68`, `-B 1000`
`7459a2290db0579a`.

The cos vs the CPU falls as `-B` grows (0.9993 at 0, 0.959 at 3000), with the shared kernel too. Not a bug: with
random data the softmax is flat, each output is an average of many V rows, and per-row cos falls with the key count
(row 1000 at `-B 0` is 0.987 as well); at `-B 0` the overall number is carried by the early rows with large outputs.
The engine's logit probe is the real accuracy check.

## How it works

The 32 cores are 4 column pairs; a pair works on one q head, and the 4 heads of a round share one kv head, so K / V
come from one mem tile broadcast to all 32 cores. A round is 256 query rows per q head: each core takes 32 rows, as
two 16-row halves with their own scores s, running max m, sum l and fp32 output y. Per 128-key block: QK for 8
16-key tiles, mask + row max, softmax, row sums, then PV for the 8 V tiles.

- **q, K and V^T arrive as bfp16 tiles made on the host.** The kernel is built with
  `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`, so a bf16 `mmul::mac` converts A and the transposed B to bfp16
  (`to_v64bfp16ebs8`, blocks of 8 along the shared dimension) and then runs `mac_8x8_8x8T` (aie_api
  `detail/aie2p/mmul_bf16_bf16.hpp`): QK re-converted the 16 x 256 Q for every 16-key tile, PV every V tile, on
  every core. With the same blocks and the same truncation made on the host (`bfp16_block` here,
  `cpu_func::_bfp16_tile_group` / `_bfp16_vt_tile_group` in the engine), `_attn_qk_bfp16` / `_attn_fv_bfp16` call
  `mac_8x8_8x8T` directly, in the same accumulation order, so the result is unchanged. Only the softmax tile P is
  still converted on the core. Layouts (72-byte tiles of 8 x 8, each tile row one 9-byte block):
  - q `[head][L/16][dh/8][2][72]` (a block: 8 dh of one query row);
  - K `[kv head][rows/16][dh/8][2][72]` (8 dh of one key);
  - V^T `[kv head][rows/16][dh/8][2][72]` (8 keys of one dh column).
  K / V move 9/16 of the bf16 bytes, and the mem tile transfers are plain linear copies.
- **32 rows per core**, so each K / V byte serves twice the queries of the shared kernel's 16. Core memory: y 32 KB,
  s 8, q 9, K / V tiles 2 x 4.5, m 1, stack 3, junk 0.5 -> ends at 64,352 of 65,536 B.
- **q delivery with a junk buffer**: the even mem tile has only 4 free MM2S channels for its 8 cores, so it sends
  both cores of a row the same 64 rows. The even column keeps the first 32, the odd column the second 32, and each
  writes the other half into a 128-word `junk` buffer with an overlapping BD (`[(36, 1), (64, 1)]` words: 2,304 words
  land on 99).
- **Lazy running max**: a row's max only moves when the block max is more than `lazy_max_step` (5.5) above it, so
  `exp(s - m) <= e^5.5 < 2^8` (fine in bf16; the sums are fp32), and most blocks keep every row's max.
- **Rescale only when needed**: `calculate_c` is one 16-lane vector op and gives exactly 1 for rows whose max did not
  move (the native `exp2(0)` is not exactly 1, which would shrink older keys a little every block). It reports whether
  any row moved (comparing the bits of c with 1.0f as integers, no scalar FPU); only then does `calculate_y` rescale
  the fp32 y (AIE2P has no native fp32 multiply; this was 21 ms of the shipped kernel's 104).
- **Mask fast path**: a 16-key tile whose keys all come before the core's first query (and inside the window) only
  needs the running max, no selects.
- **Pipelined softmax**: rows outer, `-m * log2(e)` once per row, the row's 8 tiles in a pipelined loop with separate
  `__restrict` read / write pointers (in-place is safe: each element is read and written in the same iteration), and
  one fp32 multiply-add `s * log2(e) - m * log2(e)` before `exp2`. Before, one pointer for both made Chess run one
  dependency chain per 16 scores (~20 cycles, mostly NOPs).
- Host sequence (`host/seq_gen.hpp`, a copy of the engine's `gen_mha_engine_seq`): q in two halves of 128 rows per
  round, K and V^T as 128-key blocks; the key range per round is `(Lq + 256 + 127) / 128 * 128` and must match the
  kernel's block count. RTP addresses: `L_begin` 64256, `L_chunks` 64288, `window_size` 64320.

## Results (Krackan test machine, idle NPU, 2026-11-02)

| L | shared kernel | + lazy max / rescale only when needed / mask fast path | + pipelined softmax | + bfp16 q / K / V | + 32 rows per core (this kernel) |
|---|---|---|---|---|---|
| 2048 | 4.51 ms | 3.4-3.8 ms | 3.03 ms | 2.02 ms | **1.81 ms** |
| 10752 | 103.8 ms | 77.2 ms | 69.1 ms | 42.1 ms | **35.2 ms** |

The last two steps keep the output bit-identical; the first two change it within rounding (NMSE vs fp32 at 10752:
1.829e-3 -> 1.805e-3).

How the time was split (on the 69 ms build): data movement alone 66 ms (math compiled out), math alone ~50 ms (a
build doing every step twice took 98.9 ms); they overlap. Hence the last two steps: fewer K / V bytes per query and
no per-multiply conversions.

End to end (the engine, see `FLM_DLL/detail/qwen3_5vl_npu/PREFILL_SPEEDUPS.md`): 10K TTFT 4.23 -> 4.10 s (first
steps) and 3.09 -> 2.90 s (last two); logits bit-identical over the last two (12 probe cuts, and a multi-turn run
with decode, checkpoint / restore and clear).

## Compiler (2026-11-03): Peano

Default is Peano (`ENABLE_CHESSCC ?= 0`; `ENABLE_CHESSCC=1` builds Chess), pinned CI toolchain `llvm-aie
22.0.0.2026090201+a36c62b9`. Output bit-identical to Chess (all four hashes above; engine logits identical).

Correctness: the first Peano build gave NaN. Peano's CDO loader does not zero `.bss`: the ping / pong flags and the
`static const` vectors `neg_inf` / `one` (built at run time behind a guard variable) all sat there. The flags are
`AIE_PERSIST_ZERO` (`.data` under Peano, nothing under Chess) and `neg_inf` / `one` are built on every call under
Peano. Both are needed (flags only: NaN; constants only: wrong output). The shared Qwen3.5 attention hit the same bug
(commit e997a9c27).

Speed (`run.exe`, us per call; all Peano-only code, the Chess source is unchanged):

| change | L 2048 | L 10752 | L 2048 -B 1000 |
|---|---|---|---|
| Chess | 1873 | 35772 | 3070 |
| Peano, plain port | 2386 | 51219 | 4469 |
| `apply_softmax`: 4 rows per 64-lane vector (the 256 values of a key block are contiguous) | 2120 | 44017 | 3590 |
| running max of the unmasked block 64 lanes at a time; `calculate_c` count as one vector compare | 2014 | 40576 | 3364 |
| `calculate_l`: the ones matrix converted to bfp16 once (what the emulated mmul does, minus the repeats) | 2018 | 40634 | 3402 |

Where the rest is (probe builds that skip one part, L 10752, Chess vs Peano before the changes): q.k 3.0 vs 8.4 ms,
p.v 8.2 vs 11.1 ms, everything else 24.7 vs 31.7 ms. q.k is Peano's 8-cycle schedule of the two block-vector streams
(as in dequant_mm); p.v is not software-pipelined by Peano (tried: split read / write pointers, two passes, one tile
per pass; none was faster). Attention runs once per attention layer (6 per prefill), so the remaining ~13 % is about
+0.03 s on a 10K prompt.
