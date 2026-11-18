<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# Phase 2: the vision tower's device backend

**Both halves are done.** Attention runs on `vision_attn.xclbin` and the 111
matmuls run on `vision_mm_{a,b}.xclbin`, so the tower reports `mm=npu attn=npu
bf16` — `image_cpu::backend()` prints it, and the mixed names (`npu-attn`,
`npu-mm`) remain so a half-configured run is legible in a log line rather than
being read as a finished one.

| file | what it is |
| --- | --- |
| `vision_attn_sequence.{hpp,cpp}` | `gen_attn_seq()` from `FLM_Xclbin/Qwen3_8/vision_attn/host/seq_gen.hpp`, descriptor for descriptor, with `head_perm` added |
| `vision_attn_op.hpp` | the `attn_op`: BO staging, padding, dispatch |
| `vision_mm_sequence.{hpp,cpp}` | `gen_mm_seq()` from `FLM_Xclbin/Qwen3_8/vision_mm/host/seq_gen.hpp`, plus the two designs' tile params and the per-build RTP addresses |
| `vision_mm_op.hpp` | the `mm_op`: design choice, weight BO, **operand pre-rounding**, dispatch |
| `vision_op_factory.hpp` | which backend each op gets |

Verified by `tools/verify_n15_vision_attn.cpp` (8 checks) and
`tools/verify_n16_vision_mm.cpp` (**19 checks, 0 failed** on hardware). Each
driver's own header carries its build line and reasoning; the short version is
below.

## The edits outside this folder

Two, one per phase, as the contract required. `qwen3_8mtp_npu.cpp`'s module 4
resolves `vision_attn.xclbin` and now also `vision_mm_a.xclbin` /
`vision_mm_b.xclbin`, and **throws naming whichever is missing**.
`image_cpu.cpp` did not change and neither did anything above it.

It used to honour `FLM_Q38_VISION_NPU=0` and `FLM_Q38_VISION_MM=0` and fall
back to the attention-only or the pure host factory. Both are gone with the
rest of the host arms. The A/B they served moved to `tools/verify_n14/n15/n16`,
which construct the factories directly and so can still put the real host path
on the off-arm (`ab-off-arm-is-not-the-host`) — the thing an env var was needed
for in the first place.

`activate.sh` needed three staging lines, and all three needed a **rename**:
`vision_attn`'s output is called `attn.xclbin`, same as
`gated_attention_prefill`'s, and **both** mm designs emit a file called
`mm.xclbin`. An unrenamed copy would put the language model's causal core where
the tower looks for its bidirectional one, or design B where design A belongs.

## What the move bought

### Attention — N15 check 9, both arms at the engine's `-O3`

| | device | host (16 threads) | |
| --- | --- | --- | --- |
| L = 256 | 1.62 ms | 4.48 ms | 2.8x |
| L = 1200 | 7.92 ms | 41.68 ms | 5.3x |
| per image, 27 blocks | 214 ms | 1125 ms | |

### Matmuls — N16 check 10, same conditions

| tag | K | N | device | host | | x |
| --- | --- | --- | --- | --- | --- | --- |
| `patch_embed` | 1536 | 1152 | 2.20 ms | 5.88 ms | 2.67x | x1 |
| `block.qkv` | 1152 | 3456 | 10.77 ms | 13.30 ms | 1.24x | x27 |
| `block.proj` | 1152 | 1152 | 2.34 ms | 4.49 ms | 1.92x | x27 |
| `block.fc1` | 1152 | 4608 | 5.14 ms | 22.55 ms | 4.39x | x27 |
| `block.fc2` | 4608 | 1152 | 4.49 ms | 23.23 ms | 5.17x | x27 |
| `merger.fc1` | 4608 | 4608 | 5.58 ms | 39.46 ms | 7.07x | x1 |
| `merger.fc2` | 4608 | 5120 | 5.99 ms | 40.71 ms | 6.79x | x1 |

| per image (summed) | device | host | |
| --- | --- | --- | --- |
| L = 1200 | 479–628 ms | ~1810 ms | 2.9–3.8x |
| L = 256 | 142–154 ms | ~334 ms | 2.2–2.4x |

**Do not quote that table as the win.** It uses `bench_min` — warm BOs, min of
10 — and the tower dispatches each of the 111 ops **once** per image, with a
cold sequence and a cold `x` upload. It overstates by roughly 1.8x, and the
per-op device column is noisy run to run (`merger.fc1` read 5.6 ms and 20.0 ms
on two consecutive runs).

The honest number is check 10's **in situ** arm, which drives the real
`image_cpu` so every host-side norm, rope, gelu epilogue and gather is in both
arms. Two runs at L=1200:

| backend | encode | vs full device |
| --- | --- | --- |
| `npu` (both) | 1633 / 1942 ms | — |
| `npu-attn` (mm on host) | 2573 / 2602 ms | **mm move buys 660–940 ms** |
| `cpu` (both on host) | 3118 / 3324 ms | attn move buys a further ~550–720 ms |

`qkv` is the weakest per-op ratio and it is the one shape whose host arm is
already well blocked; it is still a win at every size measured, so `run()` has
no length threshold.

**What was left was host glue — and phase 3 took it.** Those 1633–1942 ms were
~0.5 s of device matmul, ~0.2 s of device attention and ~1 s of the tower's own
fp32 host work: the norms, the rope, the `gelu_erf` epilogue, the merger gather,
and the fp32<->bf16 conversions at all 111 seam crossings. Three changes —
vectorising and fusing that conversion, threading the stages that were not, and
sizing the OpenMP team so it does not spin on every core — took **encode at
L=1200 from 1662 ms to 599 ms**, and the tower is now **device bound**: 513 ms
of the 599 is the two dispatch brackets and 91 ms is everything else.
See `REPLACEMENTWORK.md` §N-VISION-HOST for the stage table, the measurements
and the bit-identity gate; `image/vision_par.hpp` for the thread-count rule;
`image/npu/vision_bfp16.hpp` for the fused conversion.

A phase 4 would have to be kernel work, not host work.

### `make test` cannot confirm this on its own

End to end with `panda.png`, one prefill sample per arm:

| turn | mm=npu | mm=cpu |
| --- | --- | --- |
| 1 | 8.43 s | 8.55 s |
| 2 | 7.80 s | 8.07 s |
| 3 | 8.02 s | 8.04 s |

The run-to-run spread *within* the on-arm is 0.63 s — the same size as the
whole effect — so a single prefill sample cannot resolve a 0.7 s encode win.
Measure the move with check 10's in-situ arm, not with `make test`.

### The move costs accuracy, and it flips one token

Turns 1 and 2 are **identical** across all three backends, including the image
description itself, which is the turn the tower actually decides. Turn 3 (`"Is
Alibaba a good company despite that it trained you?"`, an opinion question 300+
tokens deep with the image embeddings sitting in context) opens `"Actually,"`
on `mm=npu` and `"I'm"` on both other arms — the same correction either way,
but a different first token.

This is **not** noise, and it is not a wash. `image_embeds` against the HF
reference, all three backends at the dump's grid:

| tower | relL2 | cos |
| --- | --- | --- |
| `cpu` | 0.00984 | 0.9999519 |
| `npu-attn` | 0.02864 | 0.9995901 |
| `npu` | **0.04008** | 0.9991992 |

Each device move adds the bfp16 mmul floor at more op boundaries and the error
grows monotonically; the mm move is a 1.4x degradation over attention-only and
4x over the pure host tower. **Pre-rounding already bought back 3x of it**
(0.12111 without) — without that, this would not be shippable at all.

So the honest trade is ~0.7 s per image against cos 0.9996 -> 0.9992 on the
rows the language model reads, which is enough to move a near-tie argmax deep
in a context. That was the case for keeping `FLM_Q38_VISION_MM=0` as a knob;
the decision since has been to ship one configuration, so the knob is gone and
the accuracy ladder above is the record of what it cost. Re-run it with
`tools/verify_n16_vision_mm`, which builds both factories itself.

**Build the host arm at `-O2` and it reads 3x slower**, which turns the
attention ratios into 6.5x and 13.7x and made an early estimate claim the
device *lost* at L=256. It does not; the crossover is below the smallest image
this checkpoint produces. (`host-vs-device-ab-needs-shipping-flags`)

## The accuracy fix the mm port needed: pre-round **both** operands

This is the part of phase 2 that was not in the plan, and it is the difference
between a usable tower and an unusable one.

`makefiles/common.mk:28` builds with `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`,
so `aie::mmul<8,8,8,bf16>` floors **each block of 8 consecutive k** onto a
shared-exponent, 7-mantissa-bit grid, **truncating toward -inf**
(`aie-mmul-bf16-truncates-to-bfp16`). Truncation is a biased error, so it does
not cancel across a long k; it accumulates.

`preround_bfp16_inplace()` re-quantises onto that same grid with
**round-to-nearest**, on the host, so the device's floor becomes a no-op. It is
applied to the weights in `bind()` (once) and **to `x` in `run()`** (every call).

| | before | after | |
| --- | --- | --- | --- |
| `patch_embed` relL2 | 0.11940 | 0.00653 | 18x |
| `merger.fc2` relL2 | 0.05909 | 0.00995 | 6x |
| tower `image_embeds` | 0.12111 / cos 0.9929 | **0.04008 / cos 0.9992** | |

**Pre-rounding `x` matters more here than pre-rounding `W`, which contradicts
the Gemma4 note in that memory.** `merger.fc2`'s band 3 sat at relL2 0.119
against a reference that models only the weight floor and 0.032 against one
that models both; the weight grid explained none of the gap. Each weight row
projects onto the x-error vector differently, so the damage is per-band and a
whole-tensor score hides it.

`FLM_Q38_VISION_MM_PREROUND=0` turns it off — it is the A/B, and N16 check 5
requires the verdict to **flip** with it.

**The pre-round must not use libm.** It runs over ~265M elements per
1200-patch image. A first version called `std::log2`/`std::ldexp` per group of
8 and cost **4.0 ms of `patch_embed`'s 10 ms**; deriving the exponent from the
bf16 bit pattern instead costs **0.20 ms**, a 20x difference, and is what keeps
this free rather than eating the port's whole margin.

The bias is **not** pre-rounded: it never enters the mmul, it seeds the
accumulator (`bias-prologue-spans-k-chunks`).

## The bind sites

111 bound weights, ~954 MB, which is why `bind()` is separate from `run()` —
and why each op gets its own weight BO (`xrt-bo-4gb-addressing-limit`: the 4 GB
cap is per-BO, not aggregate).

`image_cpu::image_cpu()` makes one `mm_op` per tag. D=1152, D_ff=4608,
patch_dim=1536, merger_in=4608, out_hidden=5120:

| tag | K | N | act | M (per image, L=1200) |
| --- | --- | --- | --- | --- |
| `patch_embed` | 1536 | 1152 | none | 1200 |
| `blockN.qkv` | 1152 | 3456 | none | 1200 |
| `blockN.proj` | 1152 | 1152 | none | 1200 |
| `blockN.fc1` | 1152 | 4608 | **gelu** (tanh) | 1200 |
| `blockN.fc2` | 4608 | 1152 | none | 1200 |
| `merger.fc1` | 4608 | 4608 | **gelu_erf** | 300 |
| `merger.fc2` | 4608 | 5120 | none | 300 |

`w` is `[N][K]` row-major bf16 (nn.Linear order, no transpose anywhere) and
points into the mmap'd `vision_weight.q4nx`, which outlives the op.

Design choice is a rule, not a table: **A** when `N % 384 == 0`, else **B**
when `N % 1024 == 0`, else `bind()` throws naming the geometry. Every K and N
above already divides its design's tile, so `bind()` is a straight memcpy with
no weight padding; only **M** is padded (A: 256, B: 128). N16 check 2 asserts
both the placement and the refusals.

### Traps that were real, for the matmuls

**`vision_mm`'s `rtp_act` has no `gelu_erf`.** `merger.fc1` asks for it. It
binds `none` and keeps the epilogue on the host; substituting `gelu_tanh` is
forbidden. The two differ by ~1e-3 absolute, which is invisible in an
end-to-end cosine and lands on the last rows the language model reads — that is
the entire reason it is a separate enumerator. N16 check 7 pins it: against the
HF `merger_fc1` tap, erf scores relL2 0.008264 and tanh 0.013418, so the check
is a real separation and not a tautology. Adding an `rtp_act = 3` would mean
rebuilding both bitstreams and re-extracting both RTP addresses for one
dispatch out of 111; the epilogue is 1.38M `erff` on 16 threads, ~1-2 ms.

**One `rtp_act` per round, not per dispatch** (`rtp-act-is-per-round-not-per-dispatch`).
The activation belongs to `bind()` for this reason, and each op owns a private
`npu_app` so no two bound weights ever share a sequence.

**A fused matmul hides its narrow tensors** (`fused-matmul-cosine-hides-narrow-tensors`).
`qkv` is `[Q|K|V]` in one 3456-wide output; N16 check 3 breaks relative L2 out
per third — q 0.00446, k 0.00570, **v 0.01021** — and per N band at `tile_N`
granularity, because the worst band (0.01094) is 2.3x the aggregate (0.00479).

**A rotation control's offset must not be a symmetry of the tensor it
rotates.** `vision_mm`'s own harness rotates the reference weight's rows by
exactly `N/2`. For `merger.fc1` that is degenerate — the weight has a genuine
period of `N/2`, so the "broken" reference scores 0.098 while every other shift
scores ~1.33, and the control looks like a device failure. N16 requires
collapse at **two** shifts, `N/2+1` and `N/3+1`, and gates on the weaker.

**Nothing is gated on cosine, and the gate is not tight.** Check 6's fused-gelu
agreement is 0.0246 against a 0.050 bar, which is loose on purpose: the two
arms do not share an intermediate (the fused path applies gelu to the fp32
accumulator, the host arm after a bf16 rounding) and gelu is steep where its
output is small. The claim is carried by the **40x separation** from the
unapplied arm (0.99500), not by the agreement's tightness
(`score-an-elementwise-op-with-a-control`).

**The pad rows are not inert** here either (`kv-rows-above-cursor-are-not-inert`).
`run()` zeroes rows `[M, M_pad)` on **every** call; N16 check 8 runs M=300
after M=1200, twice, and requires bit-identity with a fresh op — and requires
the second dispatch to equal the first (`a-prologue-bd-breaks-the-second-dispatch`).

**`merger.fc1` runs 512 padded rows for 300 real ones** — design A's tile_M is
256, and B is unavailable because `4608 % 1024 != 0`. 1.7x on 21.7 GFLOP: a
known, bounded waste, not a bug.

## Traps that were real, for the attention

**`head_perm` is supported, not refused.** The seam permits a device backend to
`throw`, but then N14's rotated-head control could never run against hardware.
Permuting the k/v DDR base offset in the sequence costs nothing. N15 checks it
twice: rotated-vs-straight must **collapse** (relL2 1.448) and rotated-vs-
rotated must **agree** (0.0043) — without the second half, a device that
silently dropped `head_perm` would pass the first for the wrong reason.

**The padding rows are not inert** (`kv-rows-above-cursor-are-not-inert`). The
kernel's mask covers the **key** tail, but the QK mmul still reads those rows
and the v tail is not masked at all, so a previous, longer image's rows would
be multiplied before being masked. `run()` zeroes rows `[L, L_pad)` of q, k and
v on **every call**, not just at allocation, and N15 check 8 runs L=200 after
L=1200 and requires the result to be **bit-identical** to a fresh op's.

**The captured dump is one round.** L=256 pads to 512, which is a single
iteration of the round loop, so the BD ping-pong and the advancing q offset are
untested by it. N15 check 7 tiles the real rows out to L=1200 (three rounds)
with jitter that keeps the softmax diffuse — 86.5 effective keys of 1200, not
the ~1 that independent Gaussians give (`random-kv-cache-cannot-test-attention`).

**Nothing is gated on cosine.** The vision softmax is diffuse enough that the
output sits near an average of `v` however the scores are tilted: check 4's
deliberately broken run scores relL2 1.448 at **cos 0.019**, but N14 already
showed the other direction — dropping the rotary entirely still scores cos
0.9975. The gate is relative L2, per head. (`per-row-cosine-reports-the-norm`)

**~0.03 relL2 is the floor, not a bug.** The kernel builds with
`-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`, so both operands of both the QK
and the PV products are floored onto a shared-exponent-per-8 grid — 7 mantissa
bits. `vision_attn`'s own harness reports 0.0329 at L=256 against its fp32
reference, which is what says the gap is the mmul. (`aie-mmul-bf16-truncates-to-bfp16`)

**The RTP address is per-build.** `vision_attn` 57344; `vision_mm` **A 24576,
B 7936**. A stale one does not fault — it writes into some other buffer and the
dispatch hangs in `qds_device::wait()`. N15 check 1 and N16 check 1 diff the
header constants against `build/rtp_addr.txt` *first*, so a rebuild is a
readable FAIL and not a two-minute hang. (`rtp-addresses-are-per-build`)

## Where N14 fits now

`tools/verify_n14_vision.cpp` is unchanged and still host-only — it links no
XRT. Its check 12 (block 0's attention rebuilt from `norm1`, with the rotated-
head and no-rope controls) is reproduced in N15 against the device, plus the
rotated-vs-rotated half N14 has no need for.

N16 check 9 closes the loop the other way: it drives the **real `image_cpu`**
through the device factory and scores N14's stage taps against
`vision_ref.safetensors` with hardware underneath both ops —

| tap | relL2 | cos |
| --- | --- | --- |
| `patch_embed` | 0.00622 | 0.9999807 |
| `block_00.attn_proj` | 0.00775 | 0.9999714 |
| `block_26.out` (27 deep) | 0.11352 | 0.9935379 |
| `merger_fc1` | 0.06934 | 0.9976008 |
| `image_embeds` | **0.04008** | **0.9991992** |

The rise to 0.114 at block 26 and the fall back to 0.040 at the output is the
expected shape: error accumulates through 27 residual blocks and the merger's
`fc2` projects most of it out. N14 remains the tower's *stage-by-stage*
host reference.
