<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# lm_head_npu_bin — the Qwen3.8-27B vocabulary projection, `L = 1`

`logits[VOCAB] = rms_norm(x[D], norm_w[D]) · Wᵀ`, with the final RMSNorm fused
into the kernel. One row at a time. Q8_0 weights: 32×256 int8 plus 256 bf16
scales per block, 8704 B for 8192 weights — **17/16 bytes per weight**, so the
Qwen3.8-27B table (248320 × 5120) is **1.35 GB**.

All 8 columns × 4 cores. Each round covers `M_PER_ROUND = 32 × 32 = 1024`
output rows — 128 per column, 32 per core — so `VOCAB` is padded up to a
multiple of 1024 (248320 → 248832) and the tail rows hold a replicated copy of
block 0. `LOOKAHEAD = 3` rounds of BDs are in flight at once.

The weight stream is the entire cost. A shim tile has two MM2S channels, so
each column's 128 rows arrive as two half-streams of 64, and that is the whole
reason `reorder_cpy_q80()` exists: it interleaves the source's `[nrb][ncb]`
block order into `(group, half, cb, i)` so each pair of cores reads a
contiguous run.

## Layout of this directory

| | |
|---|---|
| `iron/q4k_mm.py` | the design — memtile/core DMA chains, packet ids, locks |
| `iron/lm_head_aie.py` | the array construction |
| `kernel/qmm.cc` | the core kernel; `rms_norm(x, x, x + MODEL_DIM)`, eps `1e-6` |
| `host/host.cpp` | the standalone driver — `generate_seq()` at :174, `reorder_cpy_q80()` at :285 |

The input BO is `2·D` bf16: the hidden row, then the norm weight. The kernel
norms **in place**, so both halves must be rewritten every dispatch — a caller
that writes only the first half projects through whatever the previous call
left there, and the result is still fluent.

## This design is consumed by the DLL

`FLM_DLL/detail/qwen3_8mtp_npu/common/npu/` holds a **second copy** of the two
host-side pieces, ported to read the engine's own types:

| here | there |
|---|---|
| `generate_seq()` | `lm_head_sequence.{hpp,cpp}` — `q38npu::gen_lm_head_seq()` |
| `reorder_cpy_q80()` | `lm_head_weights.hpp` — the same permutation as a closed form |

Both have a hard contract with `iron/q4k_mm.py`: packet ids (`X_PKT_ID = 0`,
`W_PKT_ID = 1`, and W's id set on column 0 only), which channel each stream
enters on, and the `LOOKAHEAD`-deep BD rotation. **A change to the design must
be mirrored in both copies.** Nothing enforces this — a drifted sequence hangs
in `wait()` or reads silently wrong bytes.

`lm_head_weights.hpp`'s closed form is checked against this file's
running-pointer loop nest for every block (checked when it was ported);
that check was the only thing standing between the two derivations.

### Status in the engine: verified, off by default

19/19 on the port-time lm_head check, 8/8 argmax against the reference dump, and
byte-identical drafts and acceptance rate through the engine. It ships behind
`FLM_Q38_LM_HEAD=1` rather than on, because it is **not a speedup**: the
projection measures 25.3 ms here against 25.9 ms on 32 host threads — both at
the DRAM wall on 1.35 GB — and a draft step that uses it makes two NPU
dispatches instead of one, which costs ~4.9 ms/step on top. See
`FLM_DLL/detail/qwen3_8mtp_npu/REPLACEMENTWORK.md` §N-MTP-LMHEAD.

### The `L = 8` sibling

`../lm_head_8` is the batched version, wanted for the speculation verify leg.
It scores 8/8 and re-dispatches correctly; steady state is ~31 ms for all eight
rows. It once returned NaN on the **second** dispatch — a one-shot prologue BD
filled `x` exactly once per xclbin load — which is fixed, and the post-mortem is
worth reading because both dispatches reported `COMPLETED` throughout. See
`../lm_head_8/README.md` and REPLACEMENTWORK.md §N-VERIFY-LMHEAD.

It takes the same **raw** `x` this design does. The core there never holds a
whole row, so it normalises differently — the sum of squares accumulates across
round 0's k loop and `model.norm.weight` arrives as a ninth lane of the `x`
stream — but the interface is identical, so the two cannot be swapped by
mistake.

## Build and run

```bash
source ../../setup.sh      # FLM_Xclbin/setup.sh
make bitstream             # NOT `make all`
./run.exe                  # the standalone driver
```

`run.exe` takes no environment switches. It dispatches once per reference row,
so it exercises repeated dispatch by construction — which is why the defect the
`L = 8` sibling had could not hide here.

The xclbin lands at `build/QWEN3_8_27B/xclbins/lm_head.xclbin` and stages into
the runtime under its own name — unlike the attention designs, which are all
built as `layer.xclbin` and renamed.
