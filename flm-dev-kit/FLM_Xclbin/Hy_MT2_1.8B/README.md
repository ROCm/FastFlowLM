<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Hy-MT2-1.8B prefill attention — performance analysis and optimization plan

Scope: `attention_prefill/`. Everything here was derived from the scheduled object
`attention_prefill/build/kernel/attn.o(.lst)` and from measurements on the NPU
(`/dev/accel0`, npu2) using the randomized harness in `host/host.cpp` +
`host/attn_ref.hpp`.

Note this kernel is a near-verbatim copy of
`FLM_Xclbin/generic_prefill_attention/attention_DH_128_GQA_1_4`. Eight other models
in `FLM_Xclbin/` ship a byte-identical `attn_kernel.h` (`Qwen3_5`, `Qwen3_6`,
`Nanbeige`, `Qwen2_5`, the `generic_prefill_attention` variants, ...). Every fix
below is portable to all of them.

---

## 1. Baseline

`./run.exe -l <L> -i 100`, 16 q-heads, 4 kv-heads (GQA 1:4), `dh=128`, causal
(no sliding window), 4x4 cores = 2 CUs x 8 cores.

| L | NPU time | 128-key blocks | q-chunks | cosine | NMSE |
|------|-----------|------|------|----------|------------|
| 128 | 318.95 µs | 8 | 8 | 0.999127 | 0.00169 |
| 256 | 517.76 µs | 24 | 16 | — | — |
| 512 | 1032.75 µs | 80 | 32 | 0.998207 | 0.00347 |
| 1024 | 2919.01 µs | 288 | 64 | 0.997223 | 0.00548 |
| 2048 | 9979.68 µs | 1088 | 128 | — | — |

`blocks` counts (128-key block x head-round) pairs: with `n = L/128` q-chunks,
chunk `i` sweeps `i+1` kv blocks, and there are 8 head-rounds
(`Heads/num_cu = 16/2`), so `blocks = 8·n(n+1)/2` and `chunks = 8n`.

Least-squares fit over the five points:

```
t(µs)  =  214.5           fixed per invocation
       +    8.344 · blocks    per 128-key block
       +    4.71  · chunks    per q-chunk output epilogue
```

Predicts L=2048 as 9896 µs vs 9980 µs measured (**0.8% error**), L=512 within 4%.
Use this model to predict the payoff of any change before implementing it.

Where the time goes:

| | L=128 | L=1024 | L=2048 |
|---|---|---|---|
| fixed | 67% | 7% | 2% |
| per-block | 21% | 82% | 91% |
| per-chunk epilogue | 12% | 10% | 6% |

So: **short prompts are dominated by fixed dispatch cost, long prompts by the
per-block inner loop.** They need different fixes.

---

## 2. Static cycle budget per 128-key block

Bundle counts read out of `build/kernel/attn.o.lst` (issue slots between the
`ZLS_`/`ZLE_` software-pipeline labels) times trip counts, per core:

| stage | trips | cyc/iter | cycles | share |
|---|---|---|---|---|
| `calculate_y` (fp32 rescale of `y`) | 2x16 | **83** | ~2,700 | 20% |
| `apply_mask_and_get_max` x8 | 16 | 20 | ~2,760 | 20% |
| `apply_softmax` | 8x16 | 21 | ~2,700 | 20% |
| `_attn_fv` x8 | 8 | 36 | ~2,580 | 19% |
| `_attn_qk` x8 | 15 | 12 | ~1,800 | 13% |
| m broadcast + `reduce_max` (`attn.cc:70-73`, `:84-88`) | 16+16 | 9 / 19 | ~450 | 3% |
| `calculate_l` / `reorder_s` / `calculate_c` | | | ~635 | 5% |
| **total** | | | **~13,600** | |

The mmuls actually required are 512 (QK) + 512 (PV) = **1,024 cycles**, so the
kernel runs at **~7.5% of MAC peak** and **68% of core time is softmax-side
bookkeeping, not matmul**.

Sanity check: 8.344 µs/block at 1.8 GHz = ~15,000 cycles, against a 13,600-cycle
static model. The model accounts for ~90% of measured block time.

### 2.1 Root causes (all visible in the disassembly)

**(a) fp32 arithmetic is emulated.** AIE2p has no fp32 vector multiplier.
`aie::mul(CORRECT, Y)` in `calculate_y` (`attn_kernel.h:156`) expands to a
`VCONV.bf16.fp32` / `VMSC.f` / `VMUL.f` / `VADD.f` chain: 83 cycles for 64 floats,
~1.3 cycles *per element*. `calculate_l`'s epilogue has the same shape.

**(b) The softmax path is 16-lane and latency-bound.** The mask loop operates on
`wl3`/`wl8` — 256-bit halves, so half the 512-bit datapath idles. The
`apply_softmax` inner loop is 21 cycles containing 12 `NOPX` stalls:
`LDA.s16 -> VBCST -> VSUB -> VCONV -> VMUL -> VEXP2 -> VST` is a pure serial
dependency chain, one 16-lane exp per 21 cycles. It never pipelines because the
scalar `new_m_local[q]` reload sits in the middle of the chain.

**(c) `_attn_fv` is accumulator-spill bound.** 36 cycles for 8 MACs, body dominated
by `VLDA bm*` / `VST bm*`: all four 64-float C accumulators are reloaded from and
written back to the 8 KB `y` buffer on every `j` iteration — 32 memory ops per
8 MACs. Each of the 8 calls re-streams all of `y`.

**(d) `_attn_qk` starves its MAC slot.** The 12-cycle body uses the
shuffle/convert slot in 10 of 12 cycles (7 `VSHUFFLE` + 4 `VCONV`) but the MAC slot
in only 4. The culprits are `aie::transpose(K00, 8, 8)` (`attn_kernel.h:241`) and
the bf16 -> bfp16 packing forced by `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`
(`makefiles/common.mk`).

---

## 3. Open question: is the inner loop compute-bound or DMA-bound?

**This must be answered before investing in section 5.**

K and V are broadcast from `MT[3]` to all 16 cores over a *single* memtile MM2S
channel (`attn_aie.py:99-103`, `attn_tile.py` `block[7]`, `len = 2*lk_mt*d`).
Per 128-key block that is K+V = 2 x 128 x 128 bf16 = **64 KB through one channel**.
At the commonly quoted 4 B/cycle per memtile DMA channel and 1.8 GHz, that is
~16,400 cycles = **~9.1 µs/block**.

Measured is 8.344 µs/block. The static compute model says ~13,600 cycles
(~7.6 µs). **The two rooflines are within ~20% of each other**, so the current
measurement cannot distinguish them — the design happens to sit right at the
crossover.

This matters a lot: if the K/V broadcast binds, then sections 5 and 6 (compute
optimizations) buy at most ~1.2x on long prompts, and the real lever is section 7
(bandwidth amortization).

### Decision experiment (do this first)

Cheapest discriminator, kernel rebuild only, no redesign:

1. In `kernel/attn.cc`, keep every `_lock_acquire` / `_lock_release` and the loop
   structure intact, but stub out the compute — replace the `scale_div_aie` call
   with the `aie::store_v(o, ones)` line already sitting commented out at
   `attn.cc:110` and `:117`, and delete the `_attn_qk` / `_attn_fv` / softmax bodies.
2. Rebuild and re-run the L sweep. Re-fit the model.
3. If the fitted `per-block` coefficient stays near 8.3 µs, the K/V broadcast is
   the roofline -> go to section 7 first. If it collapses toward ~1-2 µs, the core
   is the roofline -> go to sections 5-6 first.

Alternative: the IRON trace hooks are already imported in `iron/attn_aie.py`
(`trace_utils`, `PortEvent`, `MemTilePortEvent`) and `visualize.py` exists.
Enabling a core-active-vs-stall trace on one core answers the same question with
more detail, at the cost of a bitstream rebuild.

---

## 4. Unconditional wins (independent of which roofline binds)

These attack the `214.5 µs` fixed term and the `4.71 µs`/chunk term, which are
pure overhead under either hypothesis. At L=128 they are 79% of runtime; at
L=1024 still 17%.

**4.1 — Output DMA granularity.** `o_ty` is 64 elements = **128 bytes**
(`attn_aie.py:49`). The epilogue at `attn.cc:111-120` therefore issues **32
separate output DMA BDs, each with a lock acquire/release round trip**, per
q-chunk per core, to move 4 KB. Widen `o_ty` to 1024 elements (2 KB) and restructure
the loop to 2 iterations. Costs 4 KB of core memory (there is ~31 KB free, see 5.4),
removes ~30 lock round-trips and 30 descriptor setups per chunk per core.

**4.2 — Serialized output drain.** The memtile output path token-chains
`o_prod_lock_0 -> 1 -> 2 -> 3` (`attn_tile.py:220-275` and `:370-425`), so core *k+1*'s
S2MM channel cannot start until core *k*'s quarter-buffer is done. The four cores
compute in parallel but drain serially. Check whether independent lock pairs per
quarter-buffer (they write disjoint offsets) allow the drains to overlap.

**4.3 — Per-invocation RTP writes.** `seq_gen.hpp:27-34` performs
`4 rows x 8 cols x 3 = 96` `rtp_write`s on **every** invocation, writing values
(`L_begin`, `Q_chunks`, `window_size`) that only change when the prompt shape
changes. Hoist them into a separate one-shot sequence.
Also: the loop covers `col 0..7`, but `attn_aie.py` instantiates `total_cols = 4`.
Half of those 96 writes target columns the design does not use — verify and drop
them.

**4.4 — Static-init guards.** `neg_inf` and `one` are function-local `static const`
(`attn.cc:38-39`). The object contains `_ZGVZ4attnE7neg_inf` / `_ZGVZ4attnE3one`
guard variables, so every call pays a guard check. Move them to namespace scope as
`constexpr`. Trivial, do it while you are in there.

---

## 5. Core compute plan (if section 3 says compute-bound)

Ordered by payoff / risk. Cycle targets are per 128-key block per core, against the
~13,600 baseline in section 2.

### 5.1 Kill the fp32 rescale — ~2,700 -> ~250 cycles (20%)

Round the running max to an integer in the log2 domain: `m~ = ceil(m · log2e)`.
Then `c = 2^(m~_prev - m~_new)` is exactly a power of two, and rescaling `y`
becomes an **int32 subtract on the fp32 exponent field** — one native vector op
instead of a 9-op emulation.

This is mathematically free: the softmax result `y/l` is invariant to the choice of
`m~` as long as `m~ >=` the true row max (no overflow) and not far above it (no
underflow). `ceil` of the true max keeps the largest `p` in `[0.5, 1]`.

Details to get right:
- Guard against `y == 0` (the exponent trick would corrupt a true zero):
  `select(0, result, y != 0)`.
- Clamp `k = m~_new - m~_prev` so a large jump saturates to 0 rather than wrapping.
- Special-case `block_idx == 0`, where `prev_m = -inf`, and skip the correction
  entirely (`y` is still zero there).
- `calculate_c` (`attn_kernel.h:55-66`) then collapses to an integer subtract, and
  `calculate_l`'s fp32 `mac_elem_16_accuracy_safe` epilogue becomes an exponent add.

**Lower-risk first cut**, if you want the win before committing to the exponent
representation: gate `calculate_y` on `m_new == m_prev` (then `c == 1` exactly).
One 16-lane vector compare skips the entire 2,700-cycle call on every block where
the row max did not move — common after the first couple of blocks.

### 5.2 Vectorize the softmax path — ~6,400 -> ~500 cycles (35%)

- **Fold `exp_scale` into the QK output scale.** `_attn_qk` already multiplies by
  `scale` at `attn_kernel.h:282-285`. Set `scale *= log2e` in `common.h` and delete
  the `aie::mul(Vec, exp_scale)` from `apply_softmax`. Free.
- **Rewrite `apply_softmax` on 32-lane bf16 vectors**, hoist the `new_m`
  broadcasts out of the loop into a precomputed vector (this is what breaks the
  serial chain), and unroll 4x to cover `VEXP2` latency. 2,700 -> ~200.
- **Split `apply_mask_and_get_max` into masked and unmasked variants.** Under
  causal-only masking (`window_size == MAX_L`), only the diagonal 128-block needs
  masking at all; interior blocks reduce to 8x `aie::max` on 512-bit vectors —
  320 -> ~10 cycles per tile. Vectorize the masked variant to 32-lane too. The
  fraction of interior blocks grows with L, so this scales the right way.
- **Drop the `m` scratch matrix.** `attn.cc:70-73` broadcasts 16 scalars into a
  16x16 buffer and `attn.cc:84-88` reduces it back with 16 `reduce_max`es
  (9 and 19 cycles/iter respectively). Keep a resident 16-lane running-max vector
  instead. ~450 -> ~40.
- `calculate_c` computes 16 scalars via 16 separate broadcast + `exp2` +
  extract-lane-0 round trips. It is one 16-lane `exp2`. ~220 -> ~15.
  (Subsumed by 5.1 if the exponent form lands.)

### 5.3 Matmul efficiency — ~4,400 -> ~2,200 cycles (16%)

- **Stop spilling the PV accumulators.** Enlarge `in_ty` from 1 to 4 k/v tiles
  (16 KB per ping/pong) and restructure `_attn_fv` so the `dh`-column block is the
  outer loop and the k-tile loop is inner. Accumulators then stay in `bm` registers
  across 4 tiles instead of round-tripping through `y` on every tile. Cuts `y`
  traffic 4x; target 36 -> ~12 cycles per 8 MACs.
- **Fold `calculate_y`'s correction into `_attn_fv`'s first accumulator load**,
  eliminating a separate 8 KB read/write pass over `y`.
- **Pre-transpose K** so `_attn_qk` can drop `aie::transpose`. Cheapest place is
  the shim->memtile access pattern in `seq_gen.hpp:94-102` (K already goes through
  a strided gather). K and V need different patterns, so give K its own memtile
  MM2S channel rather than sharing `in_buffer`. Frees the permute slot:
  12 -> ~7 cycles/iter.
- **Merge `reorder_s` into `_attn_qk`'s existing `interleave_zip`** so S is emitted
  directly in `_attn_fv`'s A-operand layout. Removes ~165 cycles.

### 5.4 Core memory budget

Current per-core usage, against 64 KB:

| buffer | size |
|---|---|
| `q_buffer` (32x128 bf16) | 8,192 B |
| `in_buffer_0/1` (16x128 bf16 each) | 8,192 B |
| `o_buffer_0/1` (64 bf16 each) | 256 B |
| `s_buffer` (16x128 bf16) | 4,096 B |
| `m_buffer` (16x16 bf16) | 512 B |
| `y_buffer` (16x128 fp32) | 8,192 B |
| 3x `L_ty` | 96 B |
| stack (`attn_tile.py:48`) | 3,072 B |
| **total** | **~32.4 KB** |

~31 KB free. The 5.3 `in_ty` growth (+24 KB) and the 4.1 `o_ty` growth (+4 KB)
both fit, landing around 60 KB — tight, so land them one at a time and watch the
`.map` files.

**Projected 5.1 + 5.2 + 5.3: ~13,600 -> ~3,200 cycles/block (~4.2x on the core).**
5.1 + 5.2 alone are ~2.4x, are confined to `kernel/attn_kernel.h` and
`kernel/attn.cc`, and need no IRON or host changes — start there.

---

## 6. Expected end-to-end effect

Applying the section-1 model, **assuming the core is the binding roofline**
(section 3 must confirm this):

| | now | +sec.4 | +sec.5 |
|---|---|---|---|
| fixed | 214.5 µs | ~150 µs | ~150 µs |
| per block | 8.34 µs | 8.34 µs | ~2.0 µs |
| per chunk | 4.71 µs | ~1.5 µs | ~1.5 µs |
| **L=128** | 319 µs | ~228 µs | ~178 µs |
| **L=1024** | 2919 µs | ~2500 µs | ~822 µs |
| **L=2048** | 9980 µs | ~9270 µs | ~2530 µs |

i.e. ~3.5-4x at long context, ~1.8x at short context. If instead the K/V broadcast
binds, the per-block term floors at ~9 µs and section 5 buys almost nothing — which
is exactly why section 3 comes first.

---

## 7. If the K/V broadcast is the roofline

The fix is arithmetic intensity, not core efficiency: do more query work per byte
of K/V streamed.

**7.1 — Exploit GQA 1:4.** Four q-heads share each kv head, but the current
schedule (`seq_gen.hpp:40-51`, `kv_head_offset = head / 2`) processes only 2 q-heads
(one per CU) against each K/V broadcast. Restructuring so all 4 q-heads sharing a
kv head are computed against one resident K/V stream **halves K/V bytes per unit of
work**. Requires each core to hold 2 heads' worth of Q and loop over heads inside
the kv-block loop.

**7.2 — Larger `lq` per core.** Each core owns only 16 query rows and re-streams all
of K/V per q-chunk. Doubling to `lq = 32` halves the number of q-chunks and
therefore halves total K/V re-streaming. Interacts with the section 5.3 memory
budget — evaluate 7.1 first, it is the cleaner win.

**7.3 — Second broadcast channel.** Split K and V onto two memtile MM2S channels
(they are already separate S2MM streams into `in_buffer`, see `attn_tile.py`
`block[1]`/`block[4]`). This pairs naturally with the K pre-transpose in 5.3, which
wants a separate K channel anyway.

---

## 8. Validation

`host/host.cpp` now generates random q/k/v and checks against the fp32 CPU
reference in `host/attn_ref.hpp` (`-r true`, the default), or replays the
safetensors dump with `-r false`. Current pass gate is
`cosine > 0.97 && NMSE < 6e-2`.

Two cautions before touching the max/rescale math:

- **Cosine similarity is a weak gate.** The section 5.1 exponent change alters
  rounding in a way cosine will happily hide. Add a max-absolute-relative-error
  check, and tighten the gate — the kernel currently achieves cosine 0.9972 / NMSE
  0.0055 at L=1024, so there is real headroom between "correct" and the 0.97 floor.
- **Watch NMSE drift with L.** It already grows 0.0017 -> 0.0055 from L=128 to
  L=1024, which is the expected bf16 flash-attention accumulation behaviour. Record
  the baseline curve now so a regression is distinguishable from that drift.

Note also that `aie::mmul<8,8,8,bf16>` on npu2 truncates both operands onto a
shared-exponent-per-8 (bfp16) grid — see `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`
in `makefiles/common.mk` and the `VCONV.bfp16ebs8.fp32` instructions in the
listing. Any accuracy work should account for this before chasing softmax rounding.

Re-extract the bundle counts from `build/kernel/attn.o.lst` after each step; the
numbers in section 2 are directly comparable. Re-fit the section 1 model after each
step so predicted and actual gains can be checked against each other.

---

## 9. Suggested order

1. Section 3 decision experiment. **Everything else branches on the result.**
2. Section 4 (unconditional overhead wins) — cheap, independent, helps short prompts most.
3. Section 5.1 + 5.2 (kernel-local, no IRON/host changes) — biggest core win per unit effort.
4. Section 5.3 (touches IRON + seq_gen) or section 7 (touches the schedule), depending on step 1.

---

## Build notes

1. **Use a July-or-newer Peano.** llvm-aie `21.0.0.2026061101` (June) miscompiles the decode
   layer: its `layer.xclbin` gives 40/40 wrong translations, 11 empty. The July llvm-aie
   `21.0.0.2026072001` gives 130/130 the same as shipped. Point `PEANO_INSTALL_DIR` at it.
   Cause not known. The four attention-kv cores use 96–98% of their 4 KB stack
   (`check_peano_stack.sh` warning), so a small code-gen change could push them over.
2. **Use an mlir-aie build that has `aie.helpers.dialects.ext`.** The pip wheel
   `mlir_aie 1.4.0` does not, so every IRON step fails with `ModuleNotFoundError`. A source
   build of mlir-aie (its `ironenv` plus `install/`) has it.
3. **A failed IRON step leaves an empty `.mlir`**, and `make` then thinks it is up to date and
   fails later in a confusing way. Delete empty `.mlir` files under `build/` before rebuilding.
4. The RTP addresses (core memory addresses the host writes settings into) are hard-coded in
   `FLM_DLL/detail/hunyuan_npu/hunyuan_npu_sequence.{hpp,cpp}`. If a kernel change moves a
   buffer, they move too. Re-check after kernel edits: decode side in
   `generic_decoding_layer/build/HUNYUAN_DENSE_1_8B/xclbins/rtp_addresses.txt`, prefill side with
   the `aie-opt --aie-assign-buffer-addresses` command written next to the constants.

### Baseline, 2026-11-02 (shipped = fresh build, same numbers)

Translation benchmark, 130 turns, system prompt pinned in the KV cache (32 tokens):

| | mean |
|---|---|
| decode speed | **49.1 tok/s** (~20.4 ms/token) |
| prefill per turn (5–9 new tokens) | **113 ms** |
| end to end per turn | 277 ms (8 tokens generated on average) |

Prefill sweep (`bench_prefill`, whole prompt, ms):

| tokens | 12 | 20 | 36 | 68 | 132 | 260 | 504 | 1004 |
|---|---|---|---|---|---|---|---|---|
| ms | 121 | 125 | 127 | 129 | 217 | 310 | 411 | 816 |

First things the numbers point at: every prompt up to 68 tokens costs ~120 ms, so short
prompts are nearly all fixed cost, and in the translation benchmark prefill is ~40% of each turn.

## Krackan baseline (WP0, 2026-11-05)

Measured on a Krackan test machine on
`hy_mt2` = IRON `5682aa708`, FastFlowLM `d5e186cb`. Every later WP compares against these numbers.
All speed numbers are medians of 3 separate runs with nobody else on the NPU.

### Translations and prefill sweep (G2, G3)

Translation benchmark: decode **55.26 tok/s**, prefill per turn **101.7 ms**, 0 empty, 21/130 text diffs (same as before).

Prefill sweep (`bench_prefill`, ms):

| tokens | 12 | 20 | 36 | 68 | 132 | 260 | 504 | 1004 |
|---|---|---|---|---|---|---|---|---|
| ms | 99.5 | 100.3 | 100.8 | 104.0 | 166.7 | 238.0 | 316.9 | 648.3 |

### Where prefill time goes (engine profile)

Engine built with `make PREFILL_PROFILE=1 BUILD_DIR=../../build/hunyuan_npu_prof` (finer timing points added in WP0,
profile build only), `test_long --decode 0` with every length 4 times, summed over the 32 layers by
`FastFlowLM/src/test/hunyuan_npu/prof_summary.py` (median of repeats 2-4). ms (% of the prompt). The printing makes
short prompts ~5% slower than the sweep above.

| step | 12 tok | 64 tok | 128 tok | 256 tok | 512 tok | 1024 tok | 4096 tok | 16384 tok |
|---|---|---|---|---|---|---|---|---|
| NPU gemm (q, kv, o, up+gate, down) | 75.9 (72%) | 77.3 (63%) | 75.5 (62%) | 124.7 (67%) | 224.7 (69%) | 449.1 (68%) | 1793.6 (58%) | 7377.0 (34%) |
| NPU attention (mha) | 12.2 (12%) | 12.9 (10%) | 12.7 (10%) | 16.2 (9%) | 27.0 (8%) | 78.6 (12%) | 814.5 (26%) | 11369.9 (53%) |
| CPU rms norms | 0.5 (0%) | 2.0 (2%) | 2.8 (2%) | 4.1 (2%) | 7.3 (2%) | 13.2 (2%) | 57.2 (2%) | 353.9 (2%) |
| CPU rope + qk norm | 0.5 (1%) | 2.2 (2%) | 3.2 (3%) | 4.9 (3%) | 8.5 (3%) | 15.2 (2%) | 59.0 (2%) | 391.5 (2%) |
| CPU glu multiply | 0.4 (0%) | 1.7 (1%) | 3.0 (2%) | 5.7 (3%) | 11.2 (3%) | 22.2 (3%) | 92.3 (3%) | 421.2 (2%) |
| CPU residual add + x memset | 0.7 (1%) | 1.6 (1%) | 2.0 (2%) | 2.9 (2%) | 4.9 (2%) | 8.9 (1%) | 38.4 (1%) | 240.1 (1%) |
| CPU fill_kv_cache | 0.2 (0%) | 0.7 (1%) | 1.1 (1%) | 1.8 (1%) | 3.2 (1%) | 4.5 (1%) | 17.7 (1%) | 82.7 (0%) |
| CPU _update_bfp16 | 0.0 (0%) | 1.9 (2%) | 2.5 (2%) | 5.0 (3%) | 8.7 (3%) | 20.6 (3%) | 81.6 (3%) | 365.1 (2%) |
| syncs (whole-BO) | 8.2 (8%) | 12.8 (10%) | 10.2 (8%) | 11.8 (6%) | 19.9 (6%) | 36.7 (6%) | 137.3 (4%) | 711.9 (3%) |
| sequence get + alloc | 0.0 (0%) | 0.0 (0%) | 0.0 (0%) | 0.0 (0%) | 0.0 (0%) | 0.0 (0%) | 0.0 (0%) | 0.1 (0%) |
| embedding | 0.1 (0%) | 0.0 (0%) | 0.1 (0%) | 0.1 (0%) | 0.1 (0%) | 0.3 (0%) | 1.2 (0%) | 5.8 (0%) |
| lm_head (incl. xclbin switch) | 5.9 (6%) | 7.7 (6%) | 6.3 (5%) | 5.9 (3%) | 5.7 (2%) | 5.6 (1%) | 5.6 (0%) | 6.3 (0%) |
| not covered by ticks | 1.4 (1%) | 1.9 (2%) | 1.8 (2%) | 1.8 (1%) | 2.9 (1%) | 3.9 (1%) | 16.2 (1%) | 134.1 (1%) |
| **total prefill ms** | 106.0 | 122.7 | 121.2 | 184.9 | 324.2 | 658.7 | 3114.5 | 21459.6 |
| repeats used | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

What the numbers say:

- **Short prompts pay for 128 rows.** 12, 64 and 128 tokens cost the same NPU time. GEMM time grows almost in line
  with the padded row count (`run.exe` below: one layer's GEMMs 2.1 ms at L 128, 7.2 ms at L 512), so the fixed part
  is small and a 12-token prompt would gain most from a smaller row plan.
- A 128-row prompt splits about as: GEMM kernels 69 ms, syncs 10 ms, CPU work 12 ms, attention kernel 7.5 ms,
  switches 7.4 ms (2 per layer, ~115 us each), lm_head 6 ms (includes switching to `layer.xclbin`), host dispatch
  ~5.5 ms (20-50 us per run over the kernel alone), overlay reload ~1.3 ms (layer 0's `q` run after the lm_head).
- Syncs are 6-10% up to 1K, the GLU multiply 2-3%, `_update_bfp16` 2-3% from 64 tokens on.
- **Long prompts:** attention grows from 8% at 512 to 26% at 4K and 53% at 16K.

### GEMMs and attention alone (`prefill_overlay/run.exe -i 20`, us)

| shape | L 128 | L 512 | cos vs CPU (L 512) |
|---|---|---|---|
| qkv 2048 -> 3072 | 313 | 992 | 0.99987 |
| kv 2048 -> 1024 | 176 | 396 | 0.99992 |
| o 2048 -> 2048 | 240 | 692 | 0.99989 |
| up+gate 2048 -> 12288 (silu) | 1007 | 3803 | 0.99969 |
| down 6144 -> 2048 | 501 | 1752 | 0.99989 |
| attention, L_begin 0 | 235 | 769 | 0.99960 |
| o + attention, switching every run | 705 | 1625 | (2 switches: ~115 / ~82 us each) |

The L 512 GEMMs are slower than the Strix Halo numbers in `prefill_overlay/README.md` (qkv 0.99 vs 0.75 ms,
up+gate 3.80 vs 2.83 ms); cause not checked (clock, PROCHOT or memory).

### Long-context accuracy (G4) vs the all-CPU reference

Reference: `make HOST_ATTN=1 HOST_MM=1 BUILD_DIR=../../build/hunyuan_npu_host`, `test_long --text long/locomo.txt
--decode 16 --dump`, all lengths (512-4K added in WP0; the rebuilt reference gives
bit-identical 8K dumps to the old ones).

```
  tokens  prefill cos  top1  top5  decode cos min      mean  top1 same
     512      0.99869   yes    5/5         0.99402   0.99611     15/16
    1024      0.99949   yes    5/5         0.98857   0.99546     14/16
    2048      0.99939   yes    5/5         0.99312   0.99493     13/16
    4096      0.99868   yes    4/5         0.98430   0.99271     16/16
    8192      0.99795    NO    5/5         0.97154   0.98856     10/16
   16384      0.99349   yes    5/5         0.97999   0.99015     13/16
   24576      0.98085   yes    5/5         0.96144   0.98258     14/16
```

Same engine run (`test_long`, prefill and 16 teacher-forced decode steps):

| tokens | 512 | 1K | 2K | 4K | 8K | 16K | 24K |
|---|---|---|---|---|---|---|---|
| prefill ms | 329 | 662 | 1412 | 3133 | 7613 | 21112 | 40449 |
| prefill tok/s | 1555 | 1547 | 1451 | 1307 | 1076 | 776 | 608 |
| decode tok/s | 54.8 | 53.3 | 51.5 | 47.1 | 40.7 | 32.3 | 26.8 |

### Decode (`decoding/bench_decode.exe --reps 20`, one runlist incl. lm_head, ms per token)

| ctx | 8 | 1K | 4K | 12K | 24K |
|---|---|---|---|---|---|
| ms | 17.10 | 17.86 | 20.12 | 26.22 | 35.45 |
| 32 layers alone | 14.91 | 15.70 | 17.96 | 24.08 | 33.34 |

The engine is ~1.0 ms per token slower than this at short context (55.26 tok/s = 18.1 ms) and ~1.9 ms at 24K.

### Room left on the overlay cores

- Program memory: `.text` 10,640 / 10,656 B of 16,384 on all 32 cores (~5.7 KB free). With Peano 22 it is smaller
  than the 11.3 KB in `prefill_overlay/README.md`.
- Stack: deepest call chain 2,176 B of 4,096 (`tools/check_stack.py`).
- L1 (64 KB): 4 KB stack + 53,376 B arena + 128 B RTP block + 141 B data = ~57.7 KB, ~7.6 KB free at the top. The
  GEMM's `x` tiles are 8 KB each (16 rows x 256); 64 tokens per core row needs a new arena layout (bfp16 A helps).
