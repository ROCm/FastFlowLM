<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# prefill_overlay — Hy-MT2 dequant_mm + attention taking turns on the whole array (Peano)

Added 2026-11-03. Replaces the space-split `../fused_prefill` (deleted 2026-11-12 with the
Q4_0 code; dequant_mm on columns 2-5,
attention on 0,1,6,7, both always loaded, Chess) with a time-split one: one xclbin, and
the host switches the array between the two ops. dequant_mm now gets **all 8 columns**,
attention keeps 4 (columns 0-3). Built with **Peano**. The engine uses it since 2026-11-03
(`FLM_DLL/detail/hunyuan_npu`, "Prefill overlay" in its README).

Why: the engine profile shows dequant_mm is 85-88% of prefill up to 512 tokens and
76% at 4K, and it is compute bound (868 MB of weights read at only ~8 GB/s at 128
tokens, time grows with L). Twice the cores = about half the time.

## Result (Strix Halo, L = 512, `./run.exe`)

| | fused_prefill (4 col, engine profile incl. host work) | overlay (8 col) | |
|---|---|---|---|
| qkv 2048 -> 3072 | 1.48 ms | 0.75 ms | cos 0.99987 vs CPU |
| o 2048 -> 2048 | 0.91 ms | 0.53 ms | cos 0.99989 |
| up+gate 2048 -> 12288 | 6.12 ms (two runs + host GLU) | 2.83 ms (one run) | cos 0.99969 (silu on chip) |
| down 6144 -> 2048 | 2.47 ms | 1.24 ms | cos 0.99989 |
| attention, 16 q / 4 kv heads | 0.76 ms | 0.73 ms | cos 0.99961 (L_begin 0), 0.99933 (256) |
| switch dq <-> attn | — | ~0.12 ms (~1,400 register writes) | |

The CPU references are the engine's own (`FLM_DLL/detail/hunyuan_npu/cpu_functions.hpp`:
`_dequant_mm` on the same reordered weight bytes, `_attention` on the four-way split cache),
so "correct" here means "matches what the engine computes on the host".

## Files

| | |
|---|---|
| `kernel/dq/` | `../fused_prefill/kernel` dequant_mm_e/o + headers (conv_even, silu RTP); x read as bfp16 tiles since WP2; Q4_K since C4 (was Q4_0) |
| `kernel/attn/` | `../fused_prefill/kernel` attention (with the conv_even fix); two changes, below |
| `kernel/fused_{e,o}.cc` | one core program per row parity: both kernels, each `#include`d inside its own namespace |
| `kernel/dispatch.h` | waits on the op lock, reads the RTP block, runs the op's calls |
| `iron/layout.py` | arena offsets (also generates `build/kernel/overlay_layout.h` for the cores) |
| `iron/dq_tiles.py` | `../fused_prefill/iron/dequant_mm_tile.py` on arenas |
| `iron/attn_tiles.py` | Hy-Memory's attention on arenas (same design, fixed BD ids) |
| `iron/overlay_aie.py` | variants `base` (runs), `dq`, `dq32`, `dq64`, `attn` (only dumped as register writes) |
| `tools/extract_txn2.py` | Hy-Memory's: txn dumps -> `host/op_configs.hpp` |
| `tools/check_stack.py` | Peano stack check that follows the call graph |
| `FLM_DLL/detail/hunyuan_npu/hunyuan_overlay.hpp` | switch + the two run generators (shared with the engine) |
| `FLM_DLL/detail/hunyuan_npu/hunyuan_overlay_configs.hpp` | written by `make regs` |
| `host/host.cpp` | the test (`run.exe`): every Hy-MT2 shape incl. kv 2048 -> 1024, matrices at nonzero offsets; output hashes; `--identity` (WP2) |
| `host/probe_rows.cpp` | `make probe` -> `probe_rows.exe`: do q padding rows / cache rows past the end change real rows? (WP12) |

Mechanism (base xclbin, op variants, txn replay, switch steps, fixed BD ids, op lock) is
Hy-Memory's, unchanged: `FLM_Xclbin/Hy_mem_30BA3B/prefill_overlay` on branch
`hy_mem_30A3B` (commit a1635611), whose README explains it in detail.

## What is Hy-MT2 specific

* **Arena layout.** Every offset is the address `../fused_prefill`'s allocator gave that
  buffer, minus the 4 KB stack, so each buffer stays in the same 16 KB memory bank.
  Attention's `q` and `s` move up 1 KB (its stack was 3 KB) and stay in their banks.
  RTP block at 57472, checked by the build.
* **dequant_mm on two halves.** The original standalone design was two independent
  4-column halves; `fused_prefill` kept one. Here both run: x for core row r comes from
  mem tile 2r and is broadcast across all 8 columns, weights come in per column (8 shims),
  y of a half-row goes to mem tile r (left) or r + 4 (right). A round covers 128 tokens x
  1024 output columns: the left half takes the even 512-wide tile, the right half the odd
  one.
* **One run for up+gate.** The engine interleaves up and gate in 512-row slabs (even
  tiles up, odd tiles gate) and used to issue two strided runs because the silu switch
  (`rtp_act`) was one value per run. The RTP is per core, so the left half gets 0 and the
  right half 1: one run, gate already silu'd.
* **Attention on shims 0..3** (`gen_attn_run`): the engine's `gen_mha_engine_seq`
  addressing (split cache, Nd = 2, V at half the cache) with `{IT0, IT1}, {IT2, IT3}`
  instead of `{IT0, IT1}, {IT6, IT7}`, and its settings in the RTP block.

## Peano notes

* `attn.cc`: `neg_inf` / `one` were function-local `static const`. Peano's loader does not
  zero `.bss`, so a static's init guard can read as "done" and the vector stays garbage
  (see the decode `attn_qk.cc`). They are plain locals now. The ping/pong flags moved to
  file scope; the dispatcher resets them after every switch, and the first run after a
  load is always a switch, so their load-time value does not matter.
* **Stack.** `../decoding/makefiles/check_peano_stack.sh` adds every frame in the ELF:
  4608 B > 4096 here, because the core holds both kernels. They never run at the same
  time, so `tools/check_stack.py` follows the call graph instead (`jl #addr` edges,
  `.L*` labels belong to the enclosing function; an indirect call falls back to the sum).
  Deepest chain: `_main_init -> main -> fused -> attn -> _attn_fv` = 2752 B of 4096.
* Program memory: 11.3 KB of the 16 KB per core.
* `ENABLE_CHESSCC=1` builds the same design with Chess, for comparison.

## Build and test

```
# kernel environment: ironenv + env_setup.sh of /path/to/mlir-aie,
# AIETOOLS_ROOT, PEANO_INSTALL_DIR = July llvm-aie or newer (../README.md, Build notes)
make            # ~10 s: kernels, base xclbin, txn dumps, the engine's configs header, run.exe
./run.exe -i 20 [--L 512]   # --L 32 / 64: the small-row plans, plus a byte compare with the 128-row plan
./run.exe --identity [--L 128]   # x as bfp16 tiles, end to end (W = I), see "x as bfp16 tiles"
```

`run.exe` prints a `[hash]` line per shape (and attention): the inputs are seeded, so the same
hash in two builds means bit-identical output.

## Engine result (2026-11-03)

Prefill 12 tokens 121 -> 84 ms, 512 415 -> 265 ms, 4K 4.4 -> 2.6 s, 32K 72.7 -> 58.7 s
(32K is mostly attention, which keeps 4 columns). Translations: 0 empty, 21/130 small
wording changes. Long-context accuracy vs the host-attention reference: 32K cos 0.984
(was 0.631 before the conv_even fix).

## Rules for the caller (engine)

* Keep `configured` (the op the array is set up for) across runs.
* After any other xclbin ran (decode layer, lm_head), pass `configured = -1`: the NPU
  reloads the overlay and everything the host wrote is gone; a partial switch then hangs.
* Two switches per layer (dq -> attn -> dq) = ~0.25 ms per layer per prefill call.
* Keep `bd_owner` too (`hunyuan_overlay::bd_owner_after`), and pass it to the run
  generators: it says which dq plan's mem tile BDs are loaded (next section).

## Small-row plans (WP11, 2026-11-05)

**Why.** A dq round is 128 rows (4 core rows x 32 tokens), so every prompt up to 128
tokens paid for 128 rows: ~69 of ~92 ms of a 12-token prompt (WP0 profile, Krackan).
Fewer *active* rows alone does not help (each column still streams and dequantizes the
same weights). All 32 cores
have to stay busy on less work each.

**What.** Two more dequant_mm configurations, picked by the row count of the run
(`gen_dq_run`'s L): `dq32` for L = 32 and `dq64` for L = 64 ("split" plans). The
128-row plan `dq` is unchanged (its MLIR is byte-identical to before).

* In a split plan the two core pairs of a column take **two neighbouring 128-row
  weight lanes** of one 512-wide tile instead of the same lane. Inside a pair both
  cores still share one dequantized w tile, but take different tokens: 16 + 16
  (`dq32`, RTP `dq_m16` = 1) or 32 + 32 (`dq64`). So a round covers 2048 output columns
  instead of 1024, each weight is dequantized once per round instead of twice, and per
  block a core does the same `mm_aie` call as before, once (16 tokens) or twice.
* **The trick: one shim BD streams both lanes.** In the reordered weight layout
  (`reorder_cpy`, V = 2) a lane is two "halves" of `37 * Din` bytes (Q4_0: 36), so lane + 1 starts
  exactly 2 halves after lane. The weight gather `{blocks_per_row, 2, 4, blk}` becomes
  `{blocks_per_row, 4, 4, blk}`: per 256 input columns it sends lane's 4 blocks, then
  lane + 1's. The mem tile's weight relay splits this stream: S2MM 5 writes A0, B0, A1,
  B1 (each 18,944 B, Q4_0: 18,432 = one 256-column step of one lane, inside the old qw0 / qw1 slots),
  MM2S 2 sends A to core row 0, MM2S 3 sends B to core row 2. No new shim channel, no
  new arena space, 2 more mem tile locks.
* Core tile BDs and locks are the same as `dq`. The kernel change is small: with
  `dq_m16` it computes one 16-token slice per block (x0 / x1 then hold consecutive k
  slices) and sends half the y. The second slice is written out, not a loop over a
  runtime count, so the 32-token path compiles nearly as before.
* y: mem tile r still collects core row r of its half; its row now holds 4 pieces of
  128 columns ([t0 lane p][t0 lane p + 2][t2 lane p][t2 lane p + 2], p = r / 2), which
  the shim BD scatters with `{m, 2, 2, 128}`: 256-byte runs, whole 64-byte lines.
* When Dout is not a multiple of 2048 (kv, 1024) the last round has no second tile
  pair: columns 2, 3 / 6, 7 redo the first one and write it to y rows [L, 2L), so y
  needs 2L rows then (the engine's buffers have 128).
* **Results are bit-identical to the 128-row plan** (same math per row and the same
  per-element summation order): `run.exe --L 32 / 64` byte-compares every shape, and the
  engine's logits dumps are byte-identical to the WP1 build.

**Switch cost: `bd_owner`.** The three dq plans write the same mem tile BD registers
with different words. `extract_txn2.py` now marks a BD "shared" only if two ops write
it with *different* words (before: if two ops wrote it at all). Shared BDs are rewritten
by a switch only when another plan wrote them last (`bd_owner`), so a call that stays
on one plan switches dq <-> attn exactly as cheaply as before (+12 lock writes of ~1,400:
the new plans' extra mem tile locks are in the union every switch writes).

Program memory: `.text` 10,752 / 10,720 B (was 10,640 / 10,656), ~5.6 KB free. Stack
unchanged (2,176 B). L1 arena unchanged. (A template on the slice count instead of the
written-out second slice cost +900 B; a plain runtime loop made the 128-row path ~0.5%
slower at 504 tokens.)

Krackan, `run.exe -i 20`, median of 3 runs, us (L 128 = the unchanged 128-row plan):

| shape | L 32 | L 64 | L 128 | cos vs CPU at L 32 |
|---|---|---|---|---|
| qkv 2048 -> 3072 | 182 | 239 | 320 | 0.999955 |
| kv 2048 -> 1024 | 138 | 168 | 185 | 0.999962 |
| o 2048 -> 2048 | 139 | 168 | 247 | 0.999958 |
| up+gate 2048 -> 12288 (silu) | 356 | 543 | 1000 | 0.999896 |
| down 6144 -> 2048 | 217 | 293 | 500 | 0.999958 |

A 32-row run costs ~115 us fixed plus ~46 us per round (2048 columns x 2048 K), so the
single-round runs (q, kv, o) are mostly fixed cost now. Engine (`bench_prefill`, median of 3
alternated runs): 12 tokens 91.2 -> 55.4 ms, 20: 92.7 -> 55.0, 36: 94.3 -> 68.8; 68 tokens and
up unchanged. Engine profile at 12 tokens: GEMMs 72.5 -> 37.0 ms (36 tokens: 73.2 -> 49.0).

## Rows that change other rows (WP12, 2026-11-05)

The attention output of a real row can depend on rows the caller does not care about. Two
ways, both found with `probe_rows.exe` (attention alone, real rows byte-compared; every case
also re-runs the base case: 0 rows differ, so the runs are deterministic):

| L_begin | real rows L | q padding rows junk (not zero) | cache rows past L_begin + L junk |
|---|---|---|---|
| 1024 | 476 | rows 472-475 differ | rows 472-475 differ |
| 1024 | 480 | none | none |
| 0 | 12 | none | rows 8-11 differ |
| 512 | 188 | rows 185-186 | rows 184-187 |
| 32 | 5 (a translation turn) | none | rows 0-4 (all) |
| 32 | 40 | none | none |

1. **q padding rows, `calculate_y` (`kernel/attn/attn_kernel.h`).** After each 128-key block
   the running sums y are rescaled by c = exp(old max - new max). The code skips an 8-row
   group only when all 8 rows have c == 1, and the rescale itself is not exact for c == 1
   (y is split into a bf16 high and a bf16 low part, so it keeps only ~16 mantissa bits). So a
   padding row whose max moves forces a re-rounding of the real rows in its 8-row group. With
   one KV block (prompts from 0 up to 128 rows) c is 0 for everybody in the first block, so it
   never shows there. Zero q padding rows have max 0 in every block (c == 1), so they are
   harmless: the engine keeps them zero (`x_attn_in` and `q_rows`, see the engine README).
2. **Cache rows past the end, the emulated bf16 mmul in `_attn_fv`.** The kernel is built with
   `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`. In P.V the B side is V, and the emulation turns
   it into bfp16 blocks of 8 values along the reduction axis, i.e. **8 keys share one exponent**
   (`aie_api/detail/aie2p/mmul_bf16_bf16.hpp`: transpose, then `to_v64bfp16ebs8`). P is 0 for
   the masked keys, but a large stale V row in the same 8-key block still lowers the precision
   of the real V rows. So rows past the end of the prompt must be zero (or at least the same
   every time). `clear_context()` zeroes the cache; `restore()` does not (engine README, "Stale KV
   rows after restore"). The decode kernel reads a 16-key bf16 tail chunk past the current length
   and also packs it into 8-value bfp16 blocks (not measured alone; the translation order test in
   the engine README covers prefill and decode together).

For WP8 (attention rewrite): make the rescale per row (or exact for c == 1) so padding rows
cannot reach real rows, and keep in mind that any bfp16 V block spans 8 keys. Re-run
`probe_rows.exe` on a new kernel: with a row-independent kernel the "q padding" column is empty.
Build it in the kernel environment (`make probe`, it depends on the generated configs like
`make host`); outside it, compile `host/probe_rows.cpp` with the `run.exe` flags.

## x as bfp16 tiles (WP2, 2026-11-06)

**What.** The matmul input x now arrives as **bfp16 tiles made by the host**, not as bf16. The
kernel's A side reads them as they are; B (the dequantized weight tile) is unchanged.

**Why.** The kernel is built with `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`: a bf16
`mmul::mac` there first turns both operands into bfp16 blocks (8 values, one shared 8-bit
exponent, 8-bit mantissas), then runs the bfp16 MMUL. So every 8 x 8 block of x was converted
again inside the core for every weight tile it met, and x travelled at 16 bits per value. Made
once on the host, x costs 9 bits per value (9/16 of the bytes) and no core cycles. The host
blocks must equal the ones the core made, or the output changes.

**Layout** (`hunyuan_bfp16::_bfp16_rne_tile_group`): per 16 tokens, `[Din / 8][2 row blocks][72 B]`.
One 72-byte tile = 8 tokens x 8 k; each token is one 9-byte block (exponent byte + 8 two's
complement mantissas). That is the order `mm_aie` reads A in (k step, then the two 8-row
blocks), so a core's x buffer (16 tokens x 256 k = 4608 B) is one contiguous run of the host
buffer and the kernel reads it with one block stream (`block_vector_input_buffer_stream`,
whose pops are the fast way to fill a bfp16 register under Peano).

**Data path.**

* shim (`gen_dq_run`): per k tile (512) and 16-token group, the group's two k slices are 9216
  contiguous bytes, sent as 18 x 512 B (one dimension holds at most 1023 words).
* mem tile (`dq_tiles.py`): x buffer `[m / 16][2 slices][4608 B]`; send j picks slice j of each
  group, dims `[(m / 16, 9216 B), (9, 512 B), (512 B)]`.
* core: a plain 4608-byte BD per x buffer (before: 8192 B with a 3-D reshuffle into the mmul
  layout; the host does that now).
* All three plans (128-row `dq`, split `dq32` / `dq64`) use the same code; only m differs.

**The kernel's A side** (`mm_aie` in `kernel/dq/dequant_mm_{e,o}.cc`): pops X0, X1 (the two 8-row
blocks of a k step) from the stream; turns W0, W1 into bfp16 with the same three steps as the
emulation (8 x 8 transpose, bf16 -> fp32 accumulator, `to_vector<bfp16ebs8>`); then
`mac(X, op_transpose(W))` = the same `mac_8x8_8x8T` instruction, same accumulator order.

**How the conversion was pinned down (the identity probe).** `run.exe --identity` runs the o
shape with W = s I (q4_0 scale s, nibble 1 on the diagonal; since C4 as Q4_K: S = s, group scales 1, mins 0). Every output is then one product,
x_bfp16 * s, exact in fp32 and in bf16 (at most 8 significant bits). So the y of the *old*
kernel showed exactly which blocks the emulation made from x. The probe rows stress the corner
cases: wide exponent spread inside a block, mantissas that round up to 128, zeros and -0,
subnormals, large values. Result (1.3 M values, s = 1 and s = 2^100 so tiny values become
visible):

* **Rounding follows the core's rounding mode.** This kernel sets `conv_even`, and the
  conversion rounds to nearest, ties to even. The KV-cache packer next to it truncates; that
  matches the decode kernels only because they never set a rounding mode (default floor).
* Block exponent = the largest exponent of the 8. If a value then rounds up to +128 (does not
  fit int8) the block takes exponent + 1 and **all 8 are rounded again from the inputs**
  (-128 fits and stays).
* Subnormal bf16 inputs count as 0.

With that rule the host blocks matched all 1.3 M probe values, and the new kernel gives the
probe's bit-exact y and **every matmul output bit-identical to before** (`[hash]` lines, L 32,
64, 128, 512, all five shapes).

**Numbers** (Krackan, `run.exe -i 20`, median of 3 alternated runs, us, A = before):

| shape | L 32 A -> B | L 64 | L 128 | L 512 |
|---|---|---|---|---|
| qkv 2048 -> 3072 | 204 -> 177 | 251 -> 200 | 316 -> 258 | 948 -> 699 (-26%) |
| kv 2048 -> 1024 | 161 -> 131 | 175 -> 147 | 179 -> 157 | 396 -> 323 |
| o 2048 -> 2048 | 157 -> 132 | 174 -> 148 | 243 -> 207 | 656 -> 505 |
| up+gate 2048 -> 12288 | 382 -> 333 | 551 -> 432 | 1000 -> 780 | 3747 -> 2871 (-23%) |
| down 6144 -> 2048 | 240 -> 203 | 300 -> 232 | 499 -> 381 | 1665 -> 1274 (-24%) |

Engine (`FLM_DLL/detail/hunyuan_npu/README.md`, "x as bfp16 tiles"; the host now packs x, which
gives back part of it): `bench_prefill` 36 tokens 68.7 -> 63.9 ms, 132: 156.8 -> 131.8, 504: 295.1 ->
247.2, 1004: 607.0 -> 510.0; 12 / 20 tokens unchanged (~55 ms); 4K 2919 -> 2532 ms, 16K 20075 -> 18664.

**Room on the core.** `.text` 10,896 / 10,880 B (even / odd row; was 10,752 / 10,720), so about
5.4 KB of the 16 KB left. Stack unchanged (2,176 B, `tools/check_stack.py`). The L1 arena is
unchanged: the x0 / x1 slots are still 8 KB but x uses 4,608 B of each, so ~3.5 KB per slot is
free for later (a 32-token x buffer would need 9,216 B). The mem tile x buffers use
18 KB of their 64 KB slots.

**64-byte lines.** Only DDR *reads* changed (x, 512-byte runs). The NPU's writes (y) are as
before, and the host flushes whole 16-token groups (Din * 18 B, a multiple of 64).

## Q4_K weights (C4, 2026-11-12)

**What.** The model's body weights are Q4_K now (the converter, the engine and the decode layer
switched first). dequant_mm reads the **same** reordered Q4_K bytes the decode layer reads: one
weight buffer per layer, no second copy. Q4_0 is gone from this folder.

**The block** (`kernel/dq/q4_core.h`, `_Q4_K_bf16`): 32 rows x 256 columns in 4736 B, column
major: uint8 s[8 groups][32 rows], uint8 m[8][32], nibbles [256 columns][16 B] (low nibble = even
row, unsigned 0..15), bf16 S[32], bf16 M[32] (stored negated). w = S s q + M m.

**What stayed.** Everything around the dequant is the same: 8 columns, two 4-column halves, up|gate
in one run with the silu RTP, x as host-made bfp16 tiles, the 32 / 64-row plans, the K split of a
block between the two cores of a pair (the even core takes columns 0-127 = groups 0-3, the odd core
128-255 = groups 4-7; both write all 32 rows of the shared w tile).

**The dequant, per group of 32 columns (one core, 4 groups per block):**

* s, m (uint8, 32 rows) -> bf16 (exact, values <= 255).
* scale = S s, rounded to bf16 (the MAC needs a bf16 operand). M m goes into a fp32 accumulator
  exactly (8-bit x 8-bit product), and `base = M m - 128 scale`.
* Nibble -> bf16 without a convert: unpack to bytes and put 0x43 in the high byte. 0x43nn is the
  bf16 number 128 + nn (in [128, 256) one step is exactly 1). Hy-Memory's trick.
* `w = base + (128 + q) scale` is one MAC per 64 values, then rounded to bf16 with **round to
  nearest even** (the kernel sets `conv_even` at its start, as before; floor would bias every
  weight half a step low, which over 32 chained layers is a real error, see Hy-Memory's README).
* A load is 4 columns x 32 rows, so one scale vector `[scale, scale]` and one base accumulator
  `[base, base]` cover every 64-lane half; the shuffles into the w tile are the Q4_0 ones.

**First version was ~25% slower; why.** It built `concat(min, min)` as a vector of floats. Peano
compiled that into ~60 scalar 64-bit extract / push instructions, ~100 bundles of setup per group
(Q4_0: 6). Making the min tile as a 64-lane bf16 product straight into the accumulator
(`mul(concat(m, m), concat(M, M))`) cut the setup to ~25 bundles. Outputs are bit-identical
between the two (the `[hash]` lines match): the weight error is dominated by the bfp16 conversion
of w inside the mmul, not by rounding M m.

**Also tried, not kept** (no gain): (a) computing the 4 groups' tiles
first so their latency chains overlap: the inner loop then reloads them from the stack and gets
load stalls, and the stack frame grows to 2.3 KB; (b) one 16-step loop over the whole half block:
~40 bundles per step instead of ~36. Peano does not software-pipeline this loop in any of the
versions (Q4_0 included; `clang loop pipeline_initiation_interval` changes nothing), so a step is
one ~36-cycle dependency chain.

**Data path.** A whole 4736 B block is 1184 words, more than one BD dimension holds (1023), so the
shim still sends each block as two halves, now 2368 B (Q4_0: 2304). Mem tile weight buffers: 8
blocks = 37,888 B (was 36,864, still in the 64 KB slot). The core qw buffer grew to 4736 B, so
`y1` of the even core moved from 4608 to 4736 (same memory bank). In `hunyuan_overlay_configs.hpp`
only the weight BD lengths changed (0x480 -> 0x4A0 words).

**Build.** `.text` 10,848 / 10,880 B (even / odd row; was 10,896 / 10,880). Stack unchanged (deepest
chain 2,176 B of 4,096). RTP block unchanged.

**Correctness** (`run.exe`, CPU reference = the engine's `cpu_func::_dequant_mm` on the same Q4_K
bytes, random weights with 6-bit s / m): every shape and attention passes at L 32, 64, 128, 512,
through every switch direction; the 32 / 64-row plans are byte-identical to the 128-row plan;
`--identity` (W = 2^e I as Q4_K: S = 2^e, s = 1, m = M = 0) is exact for e = 0 and 100.

| shape | cos at L 32 | L 128 | L 512 | rel L2 |
|---|---|---|---|---|
| qkv 2048 -> 3072 | 0.999954 | 0.999929 | 0.999878 | 0.009 |
| kv 2048 -> 1024 | 0.999961 | 0.999951 | 0.999925 | 0.009 |
| o 2048 -> 2048 | 0.999957 | 0.999941 | 0.999894 | 0.009 |
| up+gate 2048 -> 12288 (silu) | 0.999930 | 0.999878 | 0.999824 | 0.010 |
| down 6144 -> 2048 | 0.999957 | 0.999936 | 0.999887 | 0.009 |

(Q4_0 had the same cos / rel L2 levels: the error is the bfp16 x and w of the mmul.)

**Speed** (Krackan, `run.exe -i 200`, us, the fastest run of 2 x 200 alternated with the Q4_0 build;
the CPU was fully loaded by other jobs, which delays the host wake-up after a run by up to ~1 ms,
so the fastest run is the fair number, and even then the unchanged attention moved by +-15%):

| shape | L 128 Q4_0 -> Q4_K | L 512 Q4_0 -> Q4_K |
|---|---|---|
| qkv 2048 -> 3072 | 242 -> 274 | 741 -> 776 (+5%) |
| kv 2048 -> 1024 | 145 -> 140 | 303 -> 320 |
| o 2048 -> 2048 | 191 -> 207 | 496 -> 570 |
| up+gate 2048 -> 12288 | 802 -> 935 | 2800 -> 3208 (+15%) |
| down 6144 -> 2048 | 392 -> 466 | 1242 -> 1408 (+13%) |

So dequant_mm is about 10-15% slower on the big shapes: per block a core now spends ~680 cycles on
the dequant instead of ~570 (the 4 x ~25-bundle group setup), next to ~640 on the mmul. The
small-row plans (L 32) are within noise (they are mostly fixed cost). `run.exe` now prints the
mean and the fastest run (`time_runs`).
