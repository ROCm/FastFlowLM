<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# Quantization analysis

## Some annotations

Weights and their metadata:

- Basic dequant equation: $w = s\,q + m$
- $w$: the dequantized weight that $q$, $s$, $m$ together represent
- $q_i$: the quantized weight (4 bits for example) at index $i$ in a group, $q_i \in [0, 15]$ for uint4. A **group** is the 32 values that share one scale/min
- $s$: scale for a group
- $m$: minimum for a group
- $q^j_i$: for Q4_K, 8 groups form a **super-block** of 256 values; $j \in [0, 7]$ is the group id in the super-block, $i \in [0, 31]$ remains the index inside one group
- $s_j / m_j$: scale/min of group $j$ in a super-block
- $S / M$: scale/min shared by all groups inside a super-block, so that $s_j$ and $m_j$ only need 6 bits each
- $s'_j / m'_j$ and $S' / M'$: the re-fitted metadata FLM stores instead — uint8 group scale/min and BF16 super-block scale/min, chosen at convert time so that $S' s'_j \approx S s_j$ and $M' m'_j \approx M m_j$

Activations and results:

- $x_i$: the input activation at index $i$ in a group
- $x^j_i$: the input activation at index $i$ in group $j$ of a super-block
- $y$: result of one reduce. The per-group / per-super-block equations below are written for a single block; the full-$K$ form is given at the end of each FLM section
- $K$: the reduction depth, i.e. the row length of the weight matrix. It spans $K/32$ groups for Q4_1 and $K/256$ super-blocks for Q4_K
- $g$: group index over the whole row, $g \in [0, K/32)$. A superscript $g$ selects that group's operands: $s^g$, $m^g$, $q^g_i$, $x^g_i$
- $b$: super-block index over the whole row, $b \in [0, K/256)$. $Y_b$ denotes the float32 value the kernel accumulates for super-block $b$, i.e. everything inside the outermost $\operatorname{BF16}(\cdot)$ of the Q4_K equations

llama.cpp re-quantizes $x$ before the dot product, so a bar marks everything on the activation side:

- $\bar{s} / \bar{m} / \bar{q}$: scale/min/quant of $x$. Indices carry over unchanged, so $\bar{q}^j_i$ pairs with $x^j_i$
- $\bar{\Sigma}$: **Q8_1 only.** The block sum $\bar{s}\sum_i \bar{q}_i \approx \sum_i x_i$ that llama.cpp precomputes and stores next to the quantized activations (the `s` field of `block_q8_1`)
- $\bar{B}_k$: **Q8_K only.** The partial sum $\sum_{i = 0}^{15} \bar{q}_{16k + i}$ over the $k$-th run of 16 activations, $k \in [0, 15]$, stored as int16 (the `bsums` field of `block_q8_K`)

## Note

- In all equations, unless a conversion is written explicitly, float32 is used to receive the intermediate data.
- If it is reduce, all in float32.
- $\operatorname{BF16}(z)$ and $\operatorname{FP16}(z)$ mean "round $z$ to the nearest value of that format"; $\operatorname{round}(z)$ means "round to the nearest integer". A conversion that is provably exact is still written out when it marks a real step in the datapath.
- In all FLM equations $q$ is unpacked from uint4 and converted to BF16 before it reaches the MAC, hence the explicit $\operatorname{BF16}(q)$. The conversion itself is exact — BF16 keeps an 8-bit significand, so it represents every integer up to $256$ and a uint4 (or a uint6 $s_j / m_j$, or a uint8 $s'_j / m'_j$) survives it unchanged.
- BF16 rounding of a non-integer costs up to $2^{-8}$ relative; FP16 keeps an 11-bit significand and costs up to $2^{-11}$. Round-to-nearest halves the worst case, and the relative step shrinks across a binade, so BF16's *rms* relative error is $1.66\cdot10^{-3}$, not $2^{-8} = 3.9\cdot10^{-3}$.

## Q4_1

- $x$: BF16
- $y$: BF16
- $q$: uint4
- $s$: FP16
- $m$: FP16

### llama_cpp

- llama.cpp never multiplies against $x$ directly: the activation block is re-quantized to Q8_1 first, so $\bar{q}_i$ is int8 and $\bar{s}$ is FP16. Q8_1 is symmetric, so $\bar{m} = 0$ — there is no min on the activation side.
- Q8_1 also stores $\bar{\Sigma} \approx \sum_i x_i$, computed once at quantize time, so the $m$ term costs a single multiply instead of a second reduction.
- FP16 $\to$ FP32 widening is exact, so $s$, $m$, $\bar{s}$, $\bar{\Sigma}$ carry no rounding marker; the only lossy steps are the $x \to$ Q8_1 rounding, the FP16 rounding of $\bar{s}$ / $\bar{\Sigma}$, and the final cast.
- $\sum q_i \bar{q}_i$ is an **exact** int32 accumulation ($\le 15 \times 127 \times 32 = 60960$); everything after it is float32, cast once at the end.

Quantization of the activation block:

$$
\begin{aligned}
\bar{s} &= \operatorname{FP16}\!\left( \frac{\max_i \lvert x_i \rvert}{127} \right), \qquad
\bar{q}_i = \operatorname{round}\!\left( \frac{x_i}{\bar{s}} \right) \in \left[ -127,\, 127 \right] \\[6pt]
\bar{\Sigma} &= \operatorname{FP16}\!\left( \bar{s} \sum_{i = 0}^{31} \bar{q}_i \right) \;\approx\; \sum_{i = 0}^{31} x_i
\end{aligned}
$$

Dot product:

$$
\begin{aligned}
y &= \operatorname{BF16}\!\left( \sum_{i = 0}^{31} \left( s q_i + m \right) x_i \right),
     \qquad x_i \approx \bar{s}\, \bar{q}_i \\[6pt]
  &= \operatorname{BF16}\!\Biggl(
        s\, \bar{s} \sum_{i = 0}^{31} q_i \bar{q}_i
      + m\, \bar{\Sigma}
     \Biggr)
\end{aligned}
$$

### FLM

This is the kernel that ships today ([`q4_core.h`](../FLM_Xclbin/Qwen3_5/qwen3_5_decoding/kernels/q4_core.h), `_Q41_x_bf16`).

- $s$ and $m$ are stored FP16 in the GGUF, so FLM rounds both to BF16 before use — hence the explicit $\operatorname{BF16}(\cdot)$ on them below.
- All summation are done in float32 before final cast.
- $x$ is never re-quantized: it stays BF16 all the way into the MAC, so there is no bar notation on this side.

$$
\begin{aligned}
y &= \operatorname{BF16}\!\left( \sum_{i = 0}^{31} \left( \operatorname{BF16}(s)\, \operatorname{BF16}(q_i) + \operatorname{BF16}(m) \right) x_i \right) \\[6pt]
  &= \operatorname{BF16}\!\Biggl(
        \operatorname{BF16}(s) \cdot \operatorname{BF16}\!\left( \sum_{i = 0}^{15} \operatorname{BF16}(q_i)\, x_i \right)
      + \operatorname{BF16}(s) \cdot \operatorname{BF16}\!\left( \sum_{i = 16}^{31} \operatorname{BF16}(q_i)\, x_i \right)
      + \operatorname{BF16}(m) \cdot \operatorname{BF16}\!\left( \sum_{i = 0}^{31} x_i \right)
     \Biggr)
\end{aligned}
$$

The split into two half-group reduces is the `temp_acc_0` / `temp_acc_1` pair in `q4_core.h`; it exists for the pipelining dependency chain, not for arithmetic. The two accumulators can be combined in the **accumulator** domain (`aie::add` on `accum<accfloat>`) and cast once, which removes one `aie::mac` and one `to_vector<bf16>()` per group. On Q4_1 that is accuracy-neutral — both halves are multiplied by the same $\operatorname{BF16}(s)$ and folded in float32 anyway, so merging trades two casts of half-magnitude sums for one of the full sum ($\mathrm{ENOB}^\ast$ $3.6685$ either way). On Q4_K it is worth $+0.004$ bits as well, because there the fold is a *second* cast; see the proposal below.

Over a full row the float32 accumulator is zeroed once per output element, carried across all $K/32$ groups, and cast to BF16 once at the very end — so the outer $\operatorname{BF16}$ moves outside $\sum_g$ and **nothing rounds between groups**:

$$
y = \operatorname{BF16}\Biggl( \sum_{g = 0}^{K/32 - 1} \biggl[
        \operatorname{BF16}(s^g) \cdot \operatorname{BF16}\!\left( \sum_{i = 0}^{15} \operatorname{BF16}(q^g_i)\, x^g_i \right)
      + \operatorname{BF16}(s^g) \cdot \operatorname{BF16}\!\left( \sum_{i = 16}^{31} \operatorname{BF16}(q^g_i)\, x^g_i \right)
      + \operatorname{BF16}(m^g) \cdot \operatorname{BF16}\!\left( \sum_{i = 0}^{31} x^g_i \right)
     \biggr] \Biggr)
$$

The single-group equation above is therefore the $K = 32$ case. For any larger $K$ it over-counts the error by charging one final cast per group where the hardware charges one per output element.

## Q4_K

- $x$: BF16
- $y$: BF16
- $q$: uint4
- $S$: FP16
- $M$: FP16
- $s_j$: uint6
- $m_j$: uint6

**Sign convention.** $m_j$ is *unsigned*, and Q4_K dequantizes as

$$
w^j_i = S s_j\, q^j_i - M m_j
$$

i.e. the min is a magnitude that is **subtracted**, unlike Q4_1's signed $+m$. Every Q4_K equation below keeps that minus.

### llama.cpp

- Super-block is 256 values = 8 groups of 32, so $j \in [0, 7]$ and $i \in [0, 31]$. The 8 six-bit $s_j$ and 8 six-bit $m_j$ are packed into 12 bytes.
- Q4_K pairs with **Q8_K**, not Q8_1. Q8_K covers the whole 256-value super-block with one scale $\bar{s}$, and that scale is stored **FP32** (unlike Q8_1's FP16 $\bar{s}$).
- Instead of one block sum, Q8_K stores the 16 partial sums $\bar{B}_k$ as int16. Group $j$ of 32 consumes two of them, $\bar{B}_{2j} + \bar{B}_{2j + 1}$.
- Both reductions stay in **exact integer** arithmetic: the products $q^j_i \bar{q}^j_i$ accumulate through int16 into int32, and the min term is an int32 dot of `bsums` against $m_j$. The only floating point work is the two scalar multiplies $S \bar{s}$ and $M \bar{s}$ plus the FP32 accumulation across super-blocks.

Quantization of the activation super-block:

$$
\begin{aligned}
\bar{s} &= \frac{x_{i^\ast}}{-128}, \qquad i^\ast = \arg\max_i \lvert x_i \rvert
   \qquad \text{(the signed extremum, so } \bar{s} \text{ takes its sign)} \\[6pt]
\bar{q}_i &= \min\!\left( 127,\, \operatorname{round}\!\left( \frac{x_i}{\bar{s}} \right) \right) \\[6pt]
\bar{B}_k &= \sum_{i = 0}^{15} \bar{q}_{16k + i} \quad \in \text{int16}, \qquad k \in \left[ 0,\, 15 \right]
\end{aligned}
$$

Dot product over one super-block:

$$
\begin{aligned}
y &= \operatorname{BF16}\!\left(
        \sum_{j = 0}^{7} \sum_{i = 0}^{31}
        \left( S s_j\, q^j_i - M m_j \right) x^j_i
     \right),
     \qquad x^j_i \approx \bar{s}\, \bar{q}^j_i \\[6pt]
  &= \operatorname{BF16}\!\Biggl(
        S \bar{s} \sum_{j = 0}^{7} s_j \sum_{i = 0}^{31} q^j_i \bar{q}^j_i
      - M \bar{s} \sum_{j = 0}^{7} m_j \left( \bar{B}_{2j} + \bar{B}_{2j + 1} \right)
     \Biggr)
\end{aligned}
$$

### FLM

- Since NPU only supports 8 bit and 4 bit, supporting 6 bit is hard. Also, converting FP16 to BF16 also loss 3 bit precision (BF16 has more range, so the loss is entirely on mantenssa). Therefore, we can do this: 1. assign 2 more bits to local scale/min, make it uint8; 2. Use BF16 to super-block scale.
- In computing, we keep the *reduce first, scale after* order: reduce $q$ against $x$ inside the group, multiply the group sum by $s'_j$, and apply $S'$ once at the end of the super-block. The super-block scale is never folded into the group scales offline.

$s'_j / m'_j$ are uint8 and $S' / M'$ are BF16, both produced offline by the re-fit of the next subsection — so the format the kernel reads is already the format the kernel wants. The whole 32-element group is reduced in one float32 accumulator and cast to BF16 exactly once, where it meets $s'_j$:

$$
\begin{aligned}
y &= \operatorname{BF16}\!\left(
        \sum_{j = 0}^{7} \sum_{i = 0}^{31}
        \left( S' s'_j\, \operatorname{BF16}(q^j_i) - M' m'_j \right) x^j_i
     \right) \\[6pt]
  &= \operatorname{BF16}\!\Biggl(
        S' \sum_{j = 0}^{7} \operatorname{BF16}(s'_j) \cdot
          \operatorname{BF16}\!\left( \sum_{i = 0}^{31} \operatorname{BF16}(q^j_i)\, x^j_i \right) \\[4pt]
    &\qquad\qquad
      - M' \sum_{j = 0}^{7} \operatorname{BF16}(m'_j) \cdot
        \operatorname{BF16}\!\left( \sum_{i = 0}^{31} x^j_i \right)
     \Biggr)
\end{aligned}
$$

**Every multiplier operand is BF16-exact, and the whole metadata path carries no run-time rounding at all:**

- $\operatorname{BF16}(q^j_i)$ is a uint4 $\le 15$, exact.
- $s'_j$ and $m'_j$ are uint8, i.e. integers $\le 255$, and BF16 still holds every integer up to $256$ exactly — so $\operatorname{BF16}(s'_j) = s'_j$ remains exact, exactly as it was for uint6. Widening from 6 to 8 bits costs nothing here; it would have cost something at 9 bits.
- $S'$ and $M'$ are *stored* BF16, so no $\operatorname{BF16}(\cdot)$ appears on them at all. Reading Q4_K natively would have to round $S$ and $M$ from FP16, twice per super-block; this format does not.
- The scale product stays exact in float32 too: $s'_j \times \operatorname{BF16}(\cdot)$ is $8 + 8 = 16$ significand bits, well inside FP32's 24.
- So the **only** run-time weight-side loss is the one $\operatorname{BF16}(\cdot)$ on the group sum — 8 per super-block, against the 24 a two-half reduce with a separate fold would take.

What it pays instead is the convert-time fit error $S' s'_j - S s_j$, bounded by construction, plus 4 bytes of extra storage per super-block: 8 B of uint8 scales + 8 B of uint8 mins + 4 B for $S', M'$ = 20 B per super-block against Q4_K's 16 B, i.e. 148 B vs 144 B, $4.625$ bpw vs $4.5$ bpw. The other win is mechanical rather than numerical: uint8 is a native NPU element width, so the 12-byte 6-bit unpack disappears from the kernel.

#### Re-fitting (uint6, FP16) into (uint8, BF16)

The pair $(S, s_j)$ is only ever used as the product, so the quantity to preserve is the *effective* per-group scale

$$
t_j = S\, s_j, \qquad j \in \left[ 0,\, 7 \right]
$$

and the fit is: choose one BF16 $S'$ and eight uint8 $s'_j$ minimizing the error in $S' s'_j \approx t_j$. Run the same procedure independently on $(M, m_j)$ to get $(M', m'_j)$; nothing below is specific to the scale side.

The one useful degree of freedom is that $s'_j$ has 2 bits more than it needs to hold $s_j$. Spending them as a plain $\times 4$ — $S' = \operatorname{BF16}(S/4)$, $s'_j = 4 s_j$ — buys nothing, since $S/4$ is the same significand as $S$ and takes the same $2^{-8}$ hit: that is native Q4_K with extra storage. The 2 bits are worth something only if they are used to *absorb* the error $S'$ makes, which means fitting $s'_j$ to $S'$ after $S'$ is fixed, not before.

```text
Input : S  (FP16),  s[0..7]  (uint6)
Output: S' (BF16),  s'[0..7] (uint8)

1.  t[j]  <- float32(S) * float32(s[j])        # exact: 11 + 6 = 17 <= 24 significand bits
2.  tmax  <- max_j |t[j]|
3.  if tmax == 0:  return S' = 0, s'[j] = 0    # dead super-block
4.  sigma <- sign(S)                           # s[j] unsigned, so all t[j] share S's sign
5.  S'    <- BF16_toward_inf( tmax / 255 )     # round the magnitude UP, never down
6.  S'    <- sigma * S'
7.  for j: s'[j] <- round_to_nearest( t[j] / S' )
8.  for j: s'[j] <- clamp( s'[j], 0, 255 )     # defensive; step 5 already guarantees it
```

- Step 5 rounds away from zero on purpose. Rounding to nearest could land just below $t_{\max}/255$, which would push $s'_{\arg\max}$ past $255$ and force a clamp — a clamp is a much larger error than the $2^{-8}$ given up by rounding up. The cost of rounding up is that the largest $s'_j$ comes out at $254$ or so instead of $255$.
- Step 7 is where the 2 extra bits pay off. $S'$ carries a relative error $\delta$, $\left| \delta \right| \le 2^{-8}$, against the ideal $t_{\max}/255$; the quotient $t_j / S'$ absorbs $-\delta$, and since $t_j/S'$ runs up to $255$, that correction is up to $255 \cdot 2^{-8} \approx 1$ integer step — representable, because the grid is 4× finer than the $s_j$ grid it came from. Rounding $S$ to BF16 at run time gives the same $\delta$ nowhere to go, so it lands on the result.
- **Fit error.** Only step 7 rounds, so $\left| S' s'_j - t_j \right| \le \left| S' \right| / 2$ for every $j$, uniformly, and $\left| S' \right| \le S \cdot 63/255 < S/4$ puts that at $S/8$ — **one eighth of a step of the grid the k-quant search itself worked on**. Measured: $0.1243$ steps, so the bound is tight and never violated.
- It does **not** follow that the re-fit is free. Comparing that bound against $\left| t_j \right| \cdot 2^{-8}$ is a worst-case-against-worst-case comparison, and BF16 does much better than its bound in RMS ($1.66 \cdot 10^{-3}$, not $3.9 \cdot 10^{-3}$). A uint8 grid pinned at $t_{\max}/255$ has the same top-end resolution ($1/510$ against $1/512$) and a *flat* absolute step below it, where BF16's shrinks. The measurement below puts the net gain of the base pseudocode at ~10%, not at elimination.
- What is genuinely gone is the *run-time* rounding: the error moves to convert time, where it is computed once in float32 and can be optimized.
- **The local search is part of the proposal, not an option.** BF16 has only 256 values per octave, so a search over the handful of admissible $S'$ at and just above $\operatorname{BF16}_{\uparrow}(t_{\max}/255)$ costs nothing offline. Score each candidate by $\sum_j \left( S' s'_j - t_j \right)^2$ and keep the best; a slightly larger $S'$ often aligns better with several $t_j$ at once than the smallest admissible one. Three candidates is the knee (`--search 3`), and it is what turns the re-fit from distribution-sensitive into a steady win — see the measured sweep.
- If $S$ can be negative in a given model file, step 4 keeps $s'_j$ unsigned by carrying the sign on $S'$, so the uint8 storage assumption holds. If $S$ is known non-negative, steps 4 and 6 drop out.

#### Over a full row

Same outer structure as Q4_1: one float32 accumulator per output element, spanning every super-block of the row, cast to BF16 exactly once. Writing $Y_b$ for the float32 body above — everything inside its outermost $\operatorname{BF16}(\cdot)$, evaluated with super-block $b$'s operands $S'^b, M'^b, s'^b_j, m'^b_j, q^{b,j}_i, x^{b,j}_i$:

$$
y = \operatorname{BF16}\Biggl( \sum_{b = 0}^{K/256 - 1} Y_b \Biggr)
$$

So the per-super-block equation is the $K = 256$ case, and every rounding count below is **per super-block**, scaling linearly with $K/256$: 8 group-sum casts, 8 activation-sum casts $\operatorname{BF16}(\sum_i x^j_i)$ (shared across all $M$ output rows), and zero metadata roundings.

## Measurements

Every equation above is simulated element for element in [`enob_flm_vs_llamacpp.py`](enob_flm_vs_llamacpp.py) — each explicit $\operatorname{BF16}(\cdot)$ applied where this document writes it, everything else float32, against ports of llama.cpp's own reference encoders (`make_qkx2_quants`, `quantize_row_q4_K_ref`, `quantize_row_q8_K_ref`, …). Numbers below are that script's output at `M=512 K=4096 T=64`, gaussian $w$ and $x$, seed 1234.

Two error terms are reported separately, then composed:

- $\varepsilon_W = \lVert W_{\mathrm{eff}} - w \rVert / \lVert w \rVert$ — the **format**. What an exact accumulator fed exact activations would have multiplied.
- $\varepsilon_{\mathrm{path}} = \lVert y - W_{\mathrm{eff}} x \rVert / \lVert w x \rVert$ — the **datapath**, as a paired difference against this same pipeline's own effective weights, so the format error cancels exactly instead of being subtracted in quadrature.
- $\mathrm{ENOB} = -\log_2 \varepsilon$, and $\mathrm{ENOB}^*$ composes the two in quadrature. Rank on $\mathrm{ENOB}^*$: it is built from $MK = 2^{21}$ weights and a paired difference, where the end-to-end $\mathrm{ENOB}_{\mathrm{tot}}$ averages only $MT$ dot products and carries $\sim 0.006$ bits of sampling noise.

The FLM rows are measured with a per-super-block BF16 cast of $Y_b$ switched **on**, which is the conservative reading of the equations above; `q4_core.h` in fact carries a single `accum` across the whole row. `--no-blk-cast` models that and is worth another $+0.0004$ bits.

### Headline

```text
  key  scenario                          bpw    ENOB_W   ENOB_path   ENOB_tot   ENOB*
  --------------------------------------------------------------------------------------
  W16  BF16 weights (ceiling)          16.000    9.234      9.231      8.736    8.732
  FK   Q4_K  FLM  proposed uint8+BF16   4.625    3.808      7.464      3.803    3.804
  LK   Q4_K  llama.cpp   Q8_K acts      4.500    3.810      7.151      3.801    3.803
  L1   Q4_1  llama.cpp   Q8_1 acts      5.000    3.676      7.464      3.670    3.672
  F1   Q4_1  FLM         BF16 acts      5.000    3.672      7.454      3.669    3.669
  RQ   Q4_K->deq->Q4_1  FLM (today)     5.000    3.568      7.459      3.558    3.564
```

**The format decides everything; the datapath decides almost nothing.** $\varepsilon_W \approx 7 \cdot 10^{-2}$ against datapath terms of $\sim 5 \cdot 10^{-3}$, and $5\cdot10^{-3}$ in quadrature against $7\cdot10^{-2}$ moves the total by $0.0037$ bits. What *is* worth a quarter of a bit is Q4_K over Q4_1 ($3.80$ vs $3.67$) and, most of all, not re-quantizing. The proposed format lands **above llama.cpp's own Q4_K** at $0.125$ bpw more, and the margin widens on outlier activations.

### llama.cpp vs FLM: the trade, quantified

The two vendors pay in different places and land nearly even.

```text
  Activation side -- what llama.cpp pays to get exact integer MACs
    Q8_1  (32-value block, FP16 scale) : eps_x = 5.364e-03 ( 7.54 bits)   9.000 bit/value
    Q8_K  (256-value block, FP32 scale): eps_x = 6.877e-03 ( 7.18 bits)   9.125 bit/value
    BF16  (what FLM feeds the MAC)     : eps_x = 1.661e-03 ( 9.23 bits)  16.000 bit/value

  Metadata side -- what FLM pays to avoid FP16
    Q4_1  BF16(d), BF16(m)          eps_W 7.823e-02 -> 7.844e-02   isolated 5.593e-03
    Q4_K  BF16(S), BF16(M) x uint6  eps_W 7.131e-02 -> 7.149e-02   isolated 5.095e-03
    Q4_K  uint8 + BF16 re-fit       eps_W 7.131e-02 -> 7.138e-02   isolated 3.205e-03
```

llama.cpp's whole reduction is exact integer arithmetic — its $\varepsilon_{\mathrm{path}}$ is *nothing but* the Q8 activation quantization plus the final cast, and the prediction confirms it to 1% (`LK` predicted $7.075\cdot10^{-3}$, measured $7.038\cdot10^{-3}$). But it buys that exactness with the worse activations: an int8 block quant lands at $7.2$–$7.5$ bits where BF16 is at $9.2$. FLM keeps the better activations and pays BF16 roundings later. The two very nearly cancel.

**The tie breaks in FLM's favour on real activations.** Gaussian $x$ is the best case for a block int8 quantizer. With 1% of values at $8\times$ (`--xdist outlier`, which is what LLM activations actually look like):

```text
    LK   ENOB_path 7.15 -> 5.73,   ENOB* 3.803 -> 3.761
    FK   ENOB_path 7.51 -> 7.56,   ENOB* 3.804 -> 3.804   (flat)
```

Q8_K takes the bigger hit because one scale spans 256 values, so a single outlier coarsens the whole super-block. BF16 is scale-free and does not care.

### The converter path today is the largest single loss in the table

`RQ` is what `q4nx/gguf_tensor.py` does now — dequantize Q4_K, re-quantize to Q4_1 (the `#TODO: FIXME: A direct conversion from q4_k to q4_1` branch). It is the worst row, and it costs *more* bits than the alternatives:

```text
  RQ  3.564 bits at 5.000 bpw   ->   FK 3.804 bits at 4.625 bpw   (+0.240 bits, -0.375 bpw)
```

ENOB understates it, because stacking two round-offs damages the *tail*, not the mean:

```text
  kurtosis   2.26 (LK) -> 3.21 (RQ)          max|e|  3.149e-01 -> 4.668e-01  (1.48x)
  p99.9      1.849e-01 -> 3.047e-01 (1.65x)  worst row  7.434e-02 -> 9.321e-02  (1.25x)
```

The worst output channel degrades faster than the average one. And even with no Q4_K kernel at all, quantizing the *original* weights straight to Q4_1 beats it ($7.824\cdot10^{-2}$ vs $8.435\cdot10^{-2}$) at the same 5.0 bpw — so the converter should never see the dequantized grid.

### What each rounding actually costs

Every datapath rounding switched off, then exactly one switched back on. A plain BF16 rounding is $1.661\cdot10^{-3}$; the last column is how much worse the *same* rounding is in place.

```text
  F1    (no roundings at all: 2.200e-07 -- FP32 accumulation only)
        rounding                    isolated eps   ENOB   x BF16
        BF16(half sums)   2/group     3.852e-03    8.02    2.32x
        BF16(sum_i x)     1/group     3.510e-03    8.15    2.11x
        BF16(y_block)     1/block     1.647e-03    9.25    0.99x
        BF16(y) final     1 total     1.673e-03    9.22    1.01x
        all on (the real path)        5.703e-03    7.45

  FK    (no roundings at all: 1.976e-07 -- FP32 accumulation only)
        rounding                    isolated eps   ENOB   x BF16
        BF16(group sum)   1/group     3.803e-03    8.04    2.29x
        BF16(sum_i x)     1/group     3.483e-03    8.17    2.10x
        BF16(y_block)     1/block     1.660e-03    9.23    1.00x
        BF16(y) final     1 total     1.667e-03    9.23    1.00x
        all on (the real path)        5.662e-03    7.46
```

**Every rounding taken *before* the min is subtracted is amplified $\sim 2.3\times$**, and the reason is cancellation: the scale path and the min path are individually far larger than the result that survives.

```text
  key   rms(scale path)  rms(y_block)   cancellation ratio
  F1          13.1395        5.6651              2.32
  FK          36.6157       15.9479              2.30
```

`BF16(y_block)` is the only rounding that sits *after* the cancellation, and it alone comes in at $1.0\times$. This is the single most useful number in the study: it says **the roundings to attack are the ones inside the group, not the ones at the end.**

### The re-fit, measured

Against the exact FP32 $t_j = S s_j$ that llama.cpp's own dequantizer would compute, using the base pseudocode (`--search 0`), which is what the $\lvert S'\rvert/2 \le S/8$ bound covers:

```text
  side   grid    max|err|/P   rms(err)/rms(t)   overflow   s'max  s'mean
  scale  uint8     0.12426         1.464e-03          0      255   203.1
  min    uint8     0.12426         1.535e-03          0      255   187.6
```

The $S/8$ bound holds exactly ($0.1243 \le 0.125$) and `overflow` is zero, so rounding $S'$ up in step 5 does keep step 8's clamp from ever firing. But against the naive alternative — $\operatorname{BF16}(S)$ times the native uint6 $s_j$:

```text
  scale  naive BF16(S) rms 4.5741e-04  ->  re-fit 4.1127e-04   (0.899x)
  min    naive BF16(M) rms 3.4549e-03  ->  re-fit 3.2362e-03   (0.937x)
```

Only ~10% better — for the reason given in the re-fit section. Worse, **the base pseudocode is distribution-sensitive**, because $S'$ is pinned at the smallest admissible value. Writing $\rho = \mathrm{rms}_j(t_j)/t_{\max}$ for how spread the eight group scales are inside a super-block:

```text
     rho bin        n      naive rms    re-fit rms   ratio
     [0.00,0.70)     326   3.9141e-04   5.1189e-04   1.308
     [0.70,0.80)    3086   4.3575e-04   4.3298e-04   0.994
     [0.80,0.90)    4449   4.7076e-04   3.9055e-04   0.830
     [0.90,1.01)     331   5.2761e-04   3.6053e-04   0.683

     w~gauss   rho 0.810   -0.034%   (helps)
     w~laplace rho 0.737   -0.003%   (a wash)
     w~outlier rho 0.605   +0.413%   (HURTS)
```

Heavier-tailed weights spread the per-group scales further, one group dominates, and the other seven sit low on a grid pinned to $t_{\max}$. Real LLM weights are the heavy-tailed case.

**This is why the local search is in the proposal.** It is worth far more than the uint8 grid itself, and it is free — convert time only, identical kernel, identical footprint:

```text
     search   eps_W          of the naive -> LK metadata gap recovered
     0        7.146174e-02    13.6%
     1        7.141078e-02    42.1%
     2        7.138110e-02    58.7%
     3        7.137895e-02    59.9%   <- the knee, and what --search defaults to
     7        7.136682e-02    66.7%
     15       7.135977e-02    70.7%
     (naive)  7.148599e-02     0.0%
     (LK)     7.130736e-02   100.0%   exact FP16 x uint6 metadata
```

and it removes the distribution sensitivity entirely: $-0.150\%$ / $-0.140\%$ / $-0.137\%$ on gauss / laplace / outlier, where `search=0` swings $-0.034\%$ / $-0.003\%$ / $+0.413\%$. **It is the difference between a safe win and a coin flip.**

### Where the casts are allowed to go

Same 20 B format and same metadata throughout; only the reduce structure moves. "Two halves" is what earlier drafts of the Q4_K equation wrote and what `q4_core.h` does today for Q4_1; "FP32 group" is the proposal — span all 32 in the accumulator and cast once. Casts are weight-side per super-block; the 8 activation-sum casts are shared across all $M$ rows and dropped.

```text
  variant                 eps_W       eps_path     ENOB*   casts/sb
  two halves, search 0    7.146e-02  7.058e-03    3.7997       24
  two halves, search 3    7.138e-02  7.045e-03    3.8014       24
  FP32 group, search 0    7.146e-02  5.653e-03    3.8022        8
  FP32 group, search 3    7.138e-02  5.662e-03    3.8038        8
  + FP32 across blocks    7.138e-02  5.413e-03    3.8042        8
  (llama.cpp Q4_K)        7.131e-02  7.038e-03    3.8028        -
```

The FP32 group reduce is both cheaper and more accurate: it deletes 16 of the 24 casts, and every cast it deletes sat before the min subtraction, where the cancellation table above says it cost $2.3\times$ its nominal price. The search is free at run time. Together they put the proposal past llama.cpp's own Q4_K on $\mathrm{ENOB}^*$, on gaussian activations, at 4.625 bpw.

The same structural change applied to the Q4_1 kernel that ships today is accuracy-neutral but still cheaper:

```text
    q4_core.h as written   eps_path 5.703e-03   ENOB* 3.6685
    merged half reduce     eps_path 5.727e-03   ENOB* 3.6685
```

### Op counts

Weight-path multiplies per super-block per output element, weights reused across $T$ activation columns; activation sums are shared across all $M$ rows and dropped.

```text
    datapath                                  mult   casts   bpw
    two halves, each MAC'd against s'_j        282      16   4.625
    two halves + BF16 fold, one s'_j MAC       274      24   4.625
    FP32 group reduce, one cast (proposed)     274       8   4.625
```

Against the first layout — which is what `q4_core.h` does — the group reduce also removes one `aie::mac` per group, 8 per super-block. Against the second it is free. Either way it is the cheapest of the three *and* the most accurate.

## Proposal: Q4_K on the NPU

Machine model, which is what the AIE actually offers: $c \mathrel{+}= a \cdot b$ with $a, b$ BF16 and $c$ a float32 accumulator; no FP16 operand, no cheap FP32 multiply, no int8 $\times$ int4 MAC. Two rules fall out of the measurements and they decide the whole design.

> **Rule 1 — never put an inexact value on a multiplier input.** $q$ (uint4), $s'_j$ / $m'_j$ (uint8), $S'$ / $M'$ (stored BF16) and $x$ (BF16 input) are all exact operands. $s'_j q$ is *not* — it needs 12 bits, so the datapath never forms it as a float.
>
> **Rule 2 — every accumulator you feed back into a multiply must be squeezed to BF16 first, and the ones inside the group cost $2.3\times$ their nominal price.** So minimize *accumulator → operand hand-offs*, and care much more about the ones before the min subtraction than the one after.

**Store uint8 $s'_j / m'_j$ + BF16 $S' / M'$ (20 B per super-block, 4.625 bpw), re-fitted offline with `--search 3`, and reduce each 32-element group in one float32 accumulator with a single cast.**

### The kernel, per super-block per output row

| step | op | cost | rounding |
|---|---|---|---|
| unpack uint4 $q$ → BF16 | — | — | exact |
| MAC $q \cdot x$ into one float32 accum per group | 256 MAC | — | none |
| $\operatorname{BF16}(\text{group sum})$ | — | 8 casts | **the only run-time weight-side loss** |
| MAC $s'_j \times$ that → `acc_s` | 8 MAC | — | $s'_j$ uint8, exact |
| MAC $m'_j \times \operatorname{BF16}(\sum_i x^j_i)$ → `acc_m` | 8 MAC | — | $x$-sums shared across all $M$ rows |
| $S' \cdot \texttt{acc\_s} - M' \cdot \texttt{acc\_m}$ | 2 | — | $S'$, $M'$ stored BF16, exact |
| float32 across super-blocks, one BF16 at end of row | — | 1 cast | after the cancellation, so $1.0\times$ |

**274 multiplies, 8 casts, zero metadata roundings, zero inexact operands, 4.625 bpw, no 6-bit unpack.**

### What each alternative costs you

| versus | you give up | you gain |
|---|---|---|
| **native Q4_K storage** (16 B/sb, 4.500 bpw) | $0.125$ bpw | the 12-byte 6-bit unpack, the 2 per-super-block metadata roundings, and — with the search — a strictly better $\varepsilon_W$ as well |
| **folding $S, M$ into 8 BF16 pairs offline** (32 B/sb, 5.000 bpw) | nothing numerically ($0.003$ bits) | $0.375$ bpw — 8.1% of all weight DMA, and decode MVM is memory-bound |
| **llama.cpp Q4_K** | nothing on gaussian activations | $\sim 0.04$ bits on outlier activations, where Q8_K's one int8 scale per 256 values loses $1.4$ bits of $\mathrm{ENOB}_{\mathrm{path}}$ and BF16 loses none |
| **`RQ`, what the converter does today** | nothing | $+0.240$ bits *and* $0.375$ bpw, plus a tail that is $1.65\times$ better at p99.9 |

The 20 B format is datapath-agnostic. An int8 $\times$ int4 $\to$ int16 MAC, if one ever exists, would let the group scale be folded into $q$ in the integer domain — the whole per-group scale path leaves the inner loop and the 8 group casts collapse to 1 — without changing a byte of what the converter emits. Committing to the format now costs nothing either way.

### Priority

The datapath choices above are worth $0.004$ bits. Ranked by what they are actually worth:

1. **Stop re-quantizing** — a direct Q4_K path is $+0.240$ bits *and* $-0.375$ bpw against `RQ` today, and it fixes a tail that is $1.65\times$ worse at p99.9. This is 50× larger than every other decision in this document.
2. **Merge the half-group reduce** — one fewer `aie::mac` and one fewer `to_vector<bf16>()` per group. Accuracy-neutral on the Q4_1 kernel that already ships, $+0.004$ bits on Q4_K where it also deletes the fold cast.
3. **The uint8 + BF16 format, built with the search** — buys the 6-bit unpack away for $0.125$ bpw. Its case is the kernel, not the ENOB; the search is what keeps the ENOB from going backwards on heavy-tailed weights.
4. Everything else is inside the noise.
