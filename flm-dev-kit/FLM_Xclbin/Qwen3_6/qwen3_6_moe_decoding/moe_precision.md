<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# Why the attention projections must stay at Q8_0

Switching weights from Q8_0 to Q4_K cost more accuracy than expected. This note
explains why, and argues for a specific fix: **keep Q4_K for the 256 routed
experts, but put the attention projections back to Q8_0.** That costs about
**+2.9% model size** and recovers most of the loss.

It assumes no prior knowledge of Mixture-of-Experts.

---

## 1. What a MoE layer actually does

A normal transformer FFN layer has one feed-forward network. Every token goes
through it.

A **Mixture-of-Experts** layer has many small feed-forward networks — in
Qwen3.6-35B-A3B, **256 of them**, called *experts* — and each token only goes
through **8** of them. Which 8 is decided per token, at runtime, by a tiny
neural network called the **router**.

That is the whole trick: the model has 256 experts' worth of *parameters* but
only pays for 8 experts' worth of *compute* per token. "35B-A3B" means 35
billion total parameters, ~3 billion active per token.

The router is a single matrix multiply. Given the token's hidden state `x`
(2048 numbers, produced by the attention block), it produces one score per
expert:

```
z = W_router @ x          W_router is 256 x 2048   ->   z is 256 scores
```

Then three steps:

1. **Select**: take the 8 experts with the highest scores.
2. **Weight**: softmax those 8 scores into weights that sum to 1.
3. **Combine**: run those 8 experts and add their outputs, weighted.

```
output = sum over the 8 chosen experts of  weight_e * Expert_e(x)
```

### A concrete example

Say for one token the router produces these top scores (out of all 256):

| rank | expert | score `z` | softmax weight |
|-----:|-------:|----------:|---------------:|
| 1 | #137 | 9.20 | 0.42 |
| 2 | #4   | 8.55 | 0.22 |
| 3 | #201 | 8.10 | 0.14 |
| 4 | #88  | 7.60 | 0.086 |
| 5 | #19  | 7.20 | 0.057 |
| 6 | #244 | 6.90 | 0.043 |
| 7 | #63  | 6.71 | 0.035 |
| 8 | #150 | 6.68 | 0.034 |
| — | — | — | — |
| 9 | #92  | **6.66** | *(not chosen)* |
| 10 | #7  | 6.31 | *(not chosen)* |

Experts #137 through #150 run. Experts #92, #7, and the other 246 do not.

Look at the boundary. Expert #150 scored **6.68** and got in. Expert #92 scored
**6.66** and did not. The gap is **0.02**.

That 0.02 gap is the entire subject of this note.

---

## 2. Why this is not like ordinary quantization error

When you quantize an ordinary matrix multiply, the error is **smooth**. If a
weight is off by 1%, the output is off by about 1%. Halve the error, halve the
damage. Errors from different weights partly cancel. This is why 4-bit
quantization works at all, and it is the assumption behind every "Q4 costs you
about this much perplexity" rule of thumb.

The router's top-8 selection is **not smooth**. It is a step function. Feed it a
score of 6.68 and expert #150 runs at full strength; feed it 6.65 and expert
#150 contributes *nothing at all*, and a completely different expert — a
different 2048x512x3 block of weights, trained to do a different thing — runs
instead.

So the router acts as an **amplifier that converts small continuous noise into
large discrete errors**. A 0.3% wobble in `z` is invisible in a normal matmul.
In the table above, 0.3% of 6.68 is 0.02 — exactly the gap between running
expert #150 and running expert #92.

Two properties make it worse:

- **Ties are common.** With 256 scores drawn from a similar distribution, the
  8th and 9th are usually very close together. Near-ties at the cut line are the
  normal case, not a rare one.
- **The error does not average out.** A wrong expert is a wrong expert; running
  it many times does not converge to the right answer.

This is why the loss from Q8_0 to Q4_K was worse than predicted. The prediction
was made as if the error were smooth. In an MoE model it is not.

---

## 3. Where the noise comes from

Note what `x` is in `z = W_router @ x`: it is the **output of the attention
block**. If the attention projections are noisy, `x` is noisy, and the router's
scores are already wrong before the router does any arithmetic of its own.

`models/qwen3.6-35b-a3b.h` has a single switch:

```cpp
#define QUANT_TYPE __Q4_K__
```

and `kernels/proj_main.cc` applies it to **every** projection it runs. So a
change intended for the 256 routed experts also moved q, k, v, o_proj,
attn_gate, the linear-attention qkv and out_proj, and the shared expert from
Q8_0 to Q4_K.

How much noise does that add? For a group-quantized format with grid step `t`,
the per-weight error is `t/sqrt(12)` against the rms weight in the group. A dot
product inherits this roughly 1:1 — both the signal and the error grow like
`sqrt(K)`, so the errors do **not** average away over the 2048-long reduction:

| format | levels | per-weight relative error |
|---|---|---|
| Q8_0 | 256 | **0.57%** |
| Q4_K | 16  | **6.28%** |

About **10x more noise on the router's input**. Simulated effect on which
experts get chosen (20k tokens, 256 experts, top-8, against an exact reference):

| attention projections | error in `x` | experts wrong per token | tokens with any swap |
|---|---|---|---|
| Q8_0 *(reference commit `20793897`)* | 0.57% | 0.10 of 8 | 9.9% |
| Q4_K *(current HEAD)* | 6.28% | **0.93 of 8** | **72.5%** |

Roughly one of the eight experts is the wrong expert, on most tokens.

---

## 4. How bad is one wrong expert?

Less bad than "1 in 8 is wrong" sounds — but the tail is what hurts.

The expert that gets displaced is almost always the *least important* one. Look
back at the example table: the swap happens at the boundary, rank 8 versus rank
9, and rank 8 only carries a softmax weight of **0.034**. The top expert carries
0.42, twelve times more, and it is nowhere near the cut line.

Measuring the true softmax mass that lands on wrong experts:

| attention projections | experts wrong | routing mass on wrong experts (mean) | (p99) | median true rank of a dropped expert |
|---|---|---|---|---|
| Q8_0 | 0.10 | 0.19% | 4.2% | 8 of 8 |
| Q4_K | 0.93 | **2.3%** | **15.8%** | 7 of 8 |

On a typical token about **2.3%** of the FFN output comes from the wrong place.
That is survivable. But on 1 token in 100, **16%** of it is misrouted — and that
is where visibly broken generations come from. A model does not need to be wrong
often to feel unreliable.

### And misrouting is only half of it

Everything above is about the router. But the same `QUANT_TYPE` flip put 6% noise
on *every* attention projection, at *every* layer, for *every* token. That is a
direct 6% perturbation of the **residual stream** — the model's main information
channel — on top of the routing damage, which only affects the FFN branch.

So the routing statistics understate the problem. They are the part that is easy
to count. The router is the smoke detector, not the fire.

---

## 5. The fix, and what it costs

**Restore per-phase quantization dispatch: Q8_0 for the projections that are not
routed experts, Q4_K for the routed experts.**

The size argument is the whole reason this is an easy call. There are 256 copies
of each expert matrix and exactly one copy of each attention matrix, so the
routed experts are **~96.5% of the projection weights**. Everything else — all
of attention, all of linear attention, the shared expert — is the remaining
~3.5%. Moving only that 3.5% from 4.625 bpw to 8.5 bpw works out to:

> **+2.9% model size.**

For that you get:

- the 6% noise off the residual stream at every layer, and
- routing back from 0.93 to 0.10 wrong experts per token.

This is **not** "going back to Q8_0". The 35B of expert weights — the part that
actually determines whether the model fits — stays at Q4_K exactly as it is
today. This is going back to what commit `20793897` already did: `proj_main.cc`
carried both `mvm_Q8<false>` and `mvm_Q4<true>` and dispatched per call site,
with `uint8*` weight buffers shared by both. The machinery was removed, not
outgrown.

Work involved: `kernels/proj_main.cc` (per-call dispatch), `iron/proj_tile.py`
(block sizing), and the DLL weight descriptors.

### "Can't we just encode Q4_K better instead?"

Checked, and no. FLM re-fits llama.cpp's `(fp16 d, uint6 s)` into
`(bf16 S', uint8 s')`, and that re-fit adds 1.6e-4 relative error against an
intrinsic 4-bit floor of ~1.4e-3 — about 1.3% in quadrature. Raising `search`
from 3 to 15 in `_refit_one_side` improves the total by **0.2%** while tripling
worst-case group error. `search=3` is already at the right knee.

There is no 10x hiding in the encoder. The 10x is the difference between 16
levels and 256 levels, and the only way to get it is to spend the bits.

### Cheapest way to confirm before committing

Set `QUANT_TYPE` to `__Q8_0__` globally and convert a model that way. It will
not fit the size budget as a product, but it separates "the attention path is
noisy" from "the experts are noisy" in a single run — and it needs no code
changes at all, since both kernels are still in the tree.

---

## Appendix — provenance of the numbers

- The **experts-wrong** and **routing-mass** tables are simulations against an
  exact float64 reference. Router scores were drawn as 256 Gaussians with sigma
  4, matching the observed scale of a bf16 `x` dotted with a bf16 2048-wide
  router row.
- The **0.57% / 6.28%** per-weight figures are an *analytic model* of the
  group-quantization grid (`step/sqrt(12)` against the rms weight in a group),
  not a measurement of the real tensors. The absolute values could be off by a
  factor. The ~10x ratio between the two formats is robust, and the ratio is
  what drives the conclusion.
- The **Q4_K re-fit** figures come from running the real `_refit_one_side` from
  `FLM_Q4NX_Converter/q4nx/gguf_tensor.py` over 200k synthetic super-blocks with
  llama.cpp-like `(fp16 d, uint6 s)` metadata, swept over `search` in 0..31.
- The **+2.9% model size** figure assumes `MODEL_DIM=2048`,
  `INTERMEDIATE_SIZE=512`, 256 experts, and the LAYER_PERIOD=4 attention /
  linear-attention mix from `models/qwen3.6-35b-a3b.h`. It is insensitive to the
  layer count and to the expert intermediate size within a factor of 2, because
  the experts are multiplied by 256 and dominate regardless.
