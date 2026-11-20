<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# Worked example: why the two extra bits have to be *re-fitted*, not padded

Companion to [`quant.md`](quant.md) §"Re-fitting (uint6, FP16) into (uint8, BF16)".
Every number below is produced by [`refit_example.py`](refit_example.py).

## The question

Q4_K stores a super-block scale $S$ in FP16 and eight group scales $s_j$ in uint6.
The NPU wants BF16 and uint8. Two ways to get there:

| | $S'$ | $s'_j$ |
|---|---|---|
| **naive** — cast and pad | $\operatorname{BF16}(S/4)$ | $4 s_j$ |
| **re-fit** — spend the bits | $\operatorname{BF16}_\uparrow(t_{\max}/255)$, searched | $\operatorname{round}(t_j / S')$ |

Both produce 20 B per super-block and both feed the identical kernel. The only
question is which reconstructs the *effective* scale $t_j = S s_j$ more accurately.

**Why the effective scale is the whole story.** The dequantized weight is
$w^j_i = t_j\, q^j_i$ with $q$ a uint4 that every scheme reproduces exactly, so
the relative error on $w$ equals the relative error on $t$, element for element.
No dot product, no activations, no kernel — comparing $S' s'_j$ against $S s_j$
*is* comparing the reconstructed weights against the originals.

## Why the naive $\times 4$ is a no-op

$S/4$ has the identical significand to $S$ — dividing by 4 only decrements the
exponent. So $\operatorname{BF16}(S/4) = \operatorname{BF16}(S)/4$, and

$$S' s'_j = \frac{\operatorname{BF16}(S)}{4} \cdot 4 s_j = \operatorname{BF16}(S)\, s_j$$

The uint8 field holds $4 s_j$, whose low two bits are **always zero**. The naive
scheme's error is therefore *exactly* the BF16 rounding of $S$, and nothing else:

$$S' s'_j - t_j = \delta \cdot t_j, \qquad \delta = \frac{\operatorname{BF16}(S) - S}{S}$$

That is native Q4_K accuracy at 4.625 bpw instead of 4.5 — strictly worse than
just reading Q4_K.

---

## Example 1 — worst-case $S$

$S = 1 + 2^{-8} = \texttt{1.0000000}\underline{\texttt{1}}\texttt{000}_2$, which sits exactly
halfway between two BF16 values, so ties-to-even drops it to $1.0$ and
$\delta = -2^{-8}$. Group scales `s = [63, 58, 51, 47, 44, 40, 35, 29]`.

```
t_j = [63.246094 58.226562 51.199219 47.183594 44.171875 40.156250 35.136719 29.113281]
```

**naive** — $S' = 0.25$, $s'_j = 4 s_j$:

| $s_j$ | 63 | 58 | 51 | 47 | 44 | 40 | 35 | 29 |
|---|---|---|---|---|---|---|---|---|
| $s'_j$ | 252 | 232 | 204 | 188 | 176 | 160 | 140 | 116 |
| error | −0.2461 | −0.2266 | −0.1992 | −0.1836 | −0.1719 | −0.1563 | −0.1367 | −0.1133 |

Every group is short by the same **−0.389%**. Relative rms $3.891\cdot10^{-3}$.

**re-fit** — $t_{\max}/255 = 0.2480239 \to S' = 0.248046875$:

| $s_j$ | 63 | 58 | 51 | 47 | 44 | 40 | 35 | 29 |
|---|---|---|---|---|---|---|---|---|
| $4 s_j$ | 252 | 232 | 204 | 188 | 176 | 160 | 140 | 116 |
| $s'_j$ | **255** | **235** | **206** | **190** | **178** | **162** | **142** | **117** |
| Δ | +3 | +3 | +2 | +2 | +2 | +2 | +2 | +1 |
| error | +0.0059 | +0.0645 | −0.1016 | −0.0547 | −0.0195 | +0.0273 | +0.0859 | −0.0918 |

Relative rms $1.388\cdot10^{-3}$ — **2.8× better**, worst case 2.4× better.

The Δ row *is* the two extra bits doing work. $S'$ landed $\approx 0.9\%$ below the
ideal $t_{\max}/255$; the quotient $t_j/S'$ absorbs that, and because it runs up
to 255 the correction reaches $255 \cdot 2^{-8} \approx 1$ integer step — exactly the
range of Δ above its $4 s_j$ baseline. The naive scheme's low two bits are pinned
at zero, so it has nowhere to put the same correction.

```
                 rel rms      max|err|
  naive         3.891e-03      0.2461
  re-fit        1.388e-03      0.1016     <= |S'|/2 = 0.1240 <= S/8 = 0.1255  OK
  ratio            0.357x       0.413x
```

## Example 2 — BF16-exact $S$: the case the search has to not lose

$S = 1.0625 = \texttt{1.0001}_2$ is already a BF16 value, so $\delta = 0$ and the
naive scheme is **exact**. A re-fit pinned at $t_{\max}/255$ would *introduce*
$1.488\cdot10^{-3}$ of error where there was none. The search is what prevents that —
candidate $k{=}1$ is the naive solution itself:

```
  k=0  S'=0.263671875   s'=[254 234 206 189 177 161 141 117]   SSE=4.434586e-02
  k=1  S'=0.265625      s'=[252 232 204 188 176 160 140 116]   SSE=0.000000e+00   <-- picked
  k=2  S'=0.267578125   s'=[250 230 203 187 175 159 139 115]   SSE=4.515839e-02
  k=3  S'=0.269531250   s'=[248 229 201 185 173 158 138 114]   SSE=5.346680e-02
```

Result: **exact, 0 error**, same as naive. This is the general reason the search is
mandatory rather than optional — when $s_{\max} = 63$, the naive $S' = \operatorname{BF16}(S)/4$
sits $\tfrac{255}{252} - 1 = 1.19\%$ above $t_{\max}/255$, i.e. $0.0119/2^{-8} \approx 3$ BF16 steps, so
`--search 3` has it in range and the fit can never come out worse than the padding.

## Example 3 — heavy-tailed group scales

`s = [63, 19, 14, 11, 9, 7, 6, 4]` ($\rho = 0.39$), $S$ with $\delta = -0.72 \cdot 2^{-8}$.
This is the regime the ρ table in `quant.md` flags: one group dominates, the other
seven sit low on a grid pinned to $t_{\max}$, so a $S'$ pinned at $k{=}0$ *loses* to
naive ($3.024\cdot10^{-3}$ vs $2.801\cdot10^{-3}$). The search moves one step up and wins:

```
  k=0  S'=0.005401611   s'=[254 77 56 44 36 28 24 16]   SSE=2.096314e-05
  k=1  S'=0.005432129   s'=[253 76 56 44 36 28 24 16]   SSE=5.722046e-06   <-- picked
  k=2  S'=0.005462646   s'=[251 76 56 44 36 28 24 16]   SSE=5.819835e-06
  k=3  S'=0.005493164   s'=[250 75 56 44 36 28 24 16]   SSE=2.108887e-05
```

```
                 rel rms      max|err|
  naive         2.801e-03    3.845e-03
  re-fit        1.580e-03    1.587e-03    <= S/8 = 2.724e-03  OK
  ratio            0.564x       0.413x
```

Note what the winning candidate does: it keeps $s'_j = 4 s_j$ for the six small
groups (where the naive answer was already the best available on this grid) and
spends its freedom on the one group that matters, $253$ instead of $252$. The
re-fit is not uniformly finer — it is *selectively* finer where the error is.

---

## The general picture

The two schemes have completely different error mechanisms, and that is why one
number does not summarize the comparison.

**Naive error = $\delta$, and nothing else.** Over all 1024 FP16 mantissas:

```
  exact (delta = 0): 128/1024 = 12.5%
  rms |delta| = 1.6194e-03     max |delta| = 3.8911e-03   (2^-8 = 3.9062e-03)
```

It is **bimodal**: free one super-block in eight, a full $2^{-8}$ gain error in the
worst ones. And within a super-block all eight groups are wrong by the *same
relative amount in the same direction*, so nothing averages out across $j$.

**Re-fit error = grid rounding, and nothing else.** Uniform on $[-S'/2, S'/2]$ with
$S' = t_{\max}/255$, so rms $= S'/(2\sqrt3)$. Against $\mathrm{rms}_j(t_j) = \rho\, t_{\max}$:

$$\varepsilon_{\text{re-fit}} \;\approx\; \frac{1}{255 \cdot 2\sqrt3 \cdot \rho} \;=\; \frac{1.132\cdot10^{-3}}{\rho}$$

Flat, sign-mixed, independent of $\delta$, and bounded by $S/8$ every time.
Example 1 gives $\rho = 0.748 \to$ predicted $1.51\cdot10^{-3}$ against measured
$1.39\cdot10^{-3}$. **This formula is what generates the ρ sensitivity in `quant.md`:**
pinned at $k{=}0$ the crossover is $\rho = 1.132/1.619 = 0.70$, matching the measured
`[0.70, 0.80)` bin sitting at ratio $0.994$. The search is what removes the $1/\rho$
term, and with it the distribution sensitivity.

Averaged over 200k random super-blocks, $S$ drawn as random FP16, spread swept:

```
  rho     naive        re-fit       ratio
  0.466   1.5009e-03   1.1397e-03   0.759
  0.542   1.4957e-03   1.1888e-03   0.795
  0.641   1.4974e-03   1.1684e-03   0.780
  0.747   1.5042e-03   1.0843e-03   0.721
  0.893   1.5042e-03   9.0306e-04   0.600
```

The naive column is flat at $\approx 1.50\cdot10^{-3}$ — it cannot depend on $\rho$, since
its error is $\delta$ regardless of what the $s_j$ look like. The re-fit is
**better at every spread**, by 20–40%, and it never goes backwards on the
heavy-tailed blocks where real LLM weights live.

## Summary

| | naive: cast + pad | re-fit + search |
|---|---|---|
| error source | $\operatorname{BF16}(S)$ rounding | uint8 grid rounding |
| relative rms | $1.62\cdot10^{-3}$ | $0.90$–$1.19\cdot10^{-3}$ |
| worst case | $2^{-8} = 3.91\cdot10^{-3}$ | $S/8$, hard bound, never violated |
| distribution | bimodal: 12.5% free, tail at $2^{-8}$ | flat |
| within a super-block | all 8 groups share one sign and magnitude | sign-mixed |
| depends on $\rho$ | no | no (the search removes it) |
| storage / kernel | identical | identical |
| convert cost | none | 4 candidate fits per super-block, offline |

The two extra bits are worth something only if they are allowed to *move*. Padded
with zeros they are 0.125 bpw spent to reproduce Q4_K exactly; re-fitted against a
searched $S'$ they buy a $0.60$–$0.80\times$ reduction in weight error, uniformly across
weight distributions, for zero run-time cost.
