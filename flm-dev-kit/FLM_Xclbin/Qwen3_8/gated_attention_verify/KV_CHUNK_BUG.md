<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# The KV replay chunk-count bug

Found 2026-09-17 while diagnosing why the Qwen3.8 NPU attention offload
(`FLM_DLL/detail/qwen3_8mtp_npu`) prefills exactly but collapses MTP
speculation.

**Status: fixed 2026-09-17, verified on hardware.** Both kernels now walk the
pass-uniform chunk count and mask; the RTP payload shrank from the `L_TOKENS`
per-token key counts to a single int. See §6 for what was actually built (it
is not either of the two options §6.3 originally offered) and §7 for the
evidence. The xclbin was rebuilt; the RTP addresses did not move.

The change is a **protocol change between host and kernel**. An old xclbin
driven by the new host reads `L_max` out of `L[0]`, which used to hold
`ctx + 1`, and a new xclbin driven by an old host reads the same slot the same
wrong way. Both directions are silent. Any deployed copy of this bitstream has
to be refreshed together with its host.

---

## 1. The bug in one paragraph

The host streams the KV context to the attention CUs as one BD whose
**iteration dimension carries the token repeat**, so every one of the
`L_TOKENS = 8` tokens in a pass receives the same number of chunks:

```
chunks = ceil((context_offset + L_TOKENS) / L_CHUNK)      // L_CHUNK = 16
```

The kernels, however, consume a number of chunks derived from **their own**
key count, `L[t] = context_offset + t + 1`:

```c
int i = L[token];
do { ... i -= 16; } while (i > 0);          // ceil(L[token] / 16) chunks
```

Those two counts only agree when every token in the pass needs the same
number of chunks. When they disagree, the chunks a short token does not read
are **not discarded at the token boundary** — they stay queued in the memtile
and the next token consumes them in place of the chunks it wanted. The pass
comes back wrong.

`iron/attn.py:368` states the contract exactly, and it is the contract being
violated:

> the host's transfer length decides how many chunks arrive and the kernels'
> `L[t]` decides how many they consume. Those two counts have to agree; see
> the assert in the host.

There was no such assert in the host, and the counts did not agree. The assert
exists now, in `_replay_kv_cache`, and both sides derive the count the same
way.

## 2. When it fires

Mismatch condition:

```
ceil((ctx + 1) / 16) != ceil((ctx + 8) / 16)      <=>      ctx % 16 >= 9
```

so **7 cursors in every 16 are corrupt** — 44%.

Worked example at `ctx = 9`: the host sends `8 x ceil(17/16) = 16` chunks.
The kernels consume `ceil(10/16) + ... + ceil(16/16) + ceil(17/16)`
` = 7 x 1 + 2 = 9`. Seven surplus chunks are left queued.

**This is why prefill is exact and only speculation breaks.** Prefill advances
the cursor by exactly `L_TOKENS = 8` per tile, so it only ever dispatches at
`ctx % 16` in `{0, 8}` — always in the safe band. A speculative verify batch
resumes at whatever the accepted-token count left behind, which is arbitrary.

## 3. The measurement

`/tmp/probe_cursor.cpp` (kept; rebuild instructions in its header comment).
One `L = 1` NPU pass at every cursor `0..47`, each on a cache freshly mirrored
from the CPU reference so nothing can carry over, scored against the CPU's own
`L = 1` pass at the same cursor.

```
 c | sent consumed | h2 cos      k cos       v cos
 0 |  1     1     | 0.9999987   0.9999789   0.9999912
 8 |  1     1     | 0.9999912   0.9999669   0.9999893
 9 |  2     1     | 0.9999921   0.9999656   0.9999913   <-- mismatch
10 |  2     1     | 0.9992543   0.9999660   0.9999913   <-- mismatch
11 |  2     1     | 0.9978929   0.9999707   0.9999923   <-- mismatch
14 |  2     1     | 0.9987134   0.9999628   0.9999904   <-- mismatch
15 |  2     1     | 0.9980466   0.9999683   0.9999909   <-- mismatch
16 |  2     2     | 0.9999616   0.9999673   0.9999918
24 |  2     2     | 0.9999230   0.9999640   0.9999898
25 |  3     2     | 0.9999360   0.9999659   0.9999906   <-- mismatch
27 |  3     2     | 0.9963143   0.9999692   0.9999918   <-- mismatch
32 |  3     3     | 0.9994255   0.9999678   0.9999915
40 |  3     3     | 0.9994038   0.9999626   0.9999904
41 |  4     3     | 0.9977968   0.9999712   0.9999924   <-- mismatch
46 |  4     3     | 0.9942025   0.9999726   0.9999921   <-- mismatch
```

Two things to note:

- **`k` and `v` are clean at 0.99996 / 0.99999 at every cursor.** The append is
  correct; the damage is entirely in the replay/softmax read.
- The safe-band baseline decays with depth on its own (0.999999 at c < 9,
  0.99994 at c = 32..40) — that is the known
  growth of prefill-attention error with the number of KV blocks. The mismatch rows sit
  an order of magnitude below their own band's baseline, which is the signal.

The first cursor of each mismatch band (9, 25) reads clean because its own
token 0 still takes chunk 0 first; it is the *following* passes that inherit
the surplus.

## 4. End-to-end cost

`tools/verify_qwen3_8mtp`, same model and prompt, pure-CPU HEAD (d013a4aa) vs
the 16-layer NPU attention offload:

|                     | CPU ref      | NPU attention |
|---------------------|--------------|---------------|
| layer_63_out cos    | 0.97511617   | 0.97384195    |
| M11 next_token      | 760 = ref    | 760 = ref     |
| M9 verify argmax    | 8/8, acc 1   | **7/8, acc 7 (ref 1)** |
| M13 mtp draft       | **7/7**      | **1/7**       |
| M19 hit rate        | 39.3%        | 3.6%          |
| M17 output-neutral  | 16/16        | **diverged at 2** |
| overall             | 10 pass      | 8 pass, 2 fail |

`make test`: CPU answers "The capital of France is **Paris**." (decode 4.6 s,
MTP 64.3%); NPU answers "Paris, France.\n2025-09-06." (decode 30.9 s, MTP
14.3%). The ~6.7x slowdown is almost entirely extra verify cycles caused by the
collapsed hit rate.

M9 over-accepting (7 where the reference accepts 1) while matching only 7/8
argmaxes was the tell: rows of the batch not being independent, rather than a
weight or dequant error, which would lower cosine uniformly and *reduce*
acceptance.

## 5. Why the host cannot fix this alone — two dead ends, both tried

**Dead end 1: per-token transfer lengths.** Replacing the single repeated BD
with `L_TOKENS` separate `npu_dma_memcpy_nd` calls, each of
`ceil((ctx + i + 1) / 16)` chunks, makes the counts agree exactly. It
**times out** (`dispatch state 8`). The token repeat has to live in the
iteration dimension `size[0]` — see the note at
`qwen3_8mtp_npu_attn_sequence.cpp:203-214` — and a repeated BD by definition replays
*one* descriptor, so a per-token length is not expressible in that form.

**Dead end 2: the same, with alternating BD ids and `npu_dma_wait` between
them** (the shape `_move_weights` uses for its rounds). Also **times out**: the
wait on that channel blocks against core progress, and the cores are waiting on
q, which is streamed later in the sequence. Deadlock.

The host replay in `qwen3_8mtp_npu_attn_sequence.cpp::_replay_kv_cache` has
been **left in its original working form** (single BD, `size[0] = L_TOKENS`
repeat, uniform `chunks`), with the comment above it updated to record the
constraint. Nothing in the DLL is currently broken by this investigation.

*Update, with the fix in:* the replay is still in that form and is still
correct — it was never the wrong side. What did have to change in the DLL is
`_write_attn_L` in `verify/npu/attn_sequence.cpp`, which was still writing the
eight-word per-token burst. It loads the very xclbin rebuilt here, so leaving
it would have fed `L_max = ctx + 1` to the new kernel. It now writes the
single `L_max` word, and carries a note that it has to be kept in step with
`kernels/attn_qk.cc`.

## 6. The fix

The surplus is the kernels' to absorb. Both attention kernels must walk **all
`chunks`** — a count derived from the pass cursor, identical for every token —
and mask the keys past their own `L[t]`, instead of stopping early.

**What was built.** The sections below are the original prescription and are
kept as written. Two things came out differently:

- The RTP payload is **one int, not `L_TOKENS`**. Once every token walks the
  same `n_chunks`, the per-token array is redundant: the host sends only
  `L_max = ctx + L_TOKENS` and the qk kernel recovers its own count as
  `keys = L_max - (L_TOKENS - 1) + token`. `L_ty` in `iron/layer_aie.py` is
  now `(1,)`, and `_write_attn_L` writes a single word per core instead of an
  eight-word burst. The kv kernel needs no key count at all — the mask is
  entirely the qk core's — so its `static int token` is gone.
- The RTP addresses **did not move** (`qk 57344`, `kv 36864`), across both the
  8-word and the 1-word rebuild. §6.3's warning still stands in general; it
  just did not bite here. `q38npu::attn_rtp_addresses` needed no edit.

The 1-word build was checked to be bit-identical to the 8-word one at all 48
cursors before the array was removed.

### 6.1 `kernels/attn_qk.cc`

Current (the chunk count comes from this token's key count):

```c
const int keys = L[token];
token = (token + 1 == L_TOKENS) ? 0 : token + 1;
_lock_release(l_cons_lock);
int i = keys;
do {
    _lock_acquire(k_cons_lock);
    _lock_acquire(s_prod_lock);
    is_ksc_ping = !is_ksc_ping;
    ...
    bool is_first = (i == keys);
    aie::mask<16> mask = aie::le(idx, (i < 16) ? i : 16);
    _attn_qk<DH / 8, GQA_R, GQA_S, GQA_T>(q, k, s, m, c_local, mask, is_first);
    ...
    i -= 16;
} while (i > 0);
```

Wanted: loop `n_chunks` times, where `n_chunks` is the *pass* count, and let
the mask kill the chunks past `keys`:

```c
const int keys = L[token];
const int n_chunks = <pass chunk count>;        // see 6.3
int remaining = keys;
for (int j = 0; j < n_chunks; j++) {
    _lock_acquire(k_cons_lock);
    _lock_acquire(s_prod_lock);
    ...
    bool is_first = (j == 0);
    int valid = remaining >= 16 ? 16 : (remaining > 0 ? remaining : 0);
    aie::mask<16> mask = aie::le(idx, valid);   // valid == 0 -> all lanes off
    _attn_qk<...>(q, k, s, m, c_local, mask, is_first);
    ...
    remaining -= 16;
}
```

A fully masked chunk must be a no-op, and by inspection of `update()` it is:
`masked_vm` is all `-inf`, so `vmax = max(-inf, *m) = *m` leaves the running
max alone; `Vec` is zeroed by `aie::select((bf16)0, Vec, mask)`; and
`correct = *m - vmax = 0`, so `exp(correct) = 1` rescales the accumulator by
one. **Verify this on hardware rather than trusting the reading** — the
`aie::le(idx, 0)` case with `idx = 1..16` is the one to confirm.

**Confirmed on hardware. A fully masked chunk is exactly inert.** The control
is `ATTN_TAIL_POISON=<value> ATTN_TAIL_FROM=<row>` in `host/host.cpp`: it
zeroes the cache, copies the golden rows below `from`, and fills everything
from `from` up to the padded chunk boundary with an absurd value. At
`ctx = 41` the pass spans four chunks and only token 7 reaches chunk 3, so
tokens 0..6 mask it entirely.

| poisoned range | cosine | reading |
| --- | --- | --- |
| none | 0.999992 | baseline |
| `[56, 64)` | 0.999992, bit-identical | the mask contributes exact zeros |
| `[52, 64)` | 0.999591 | |
| `[49, 64)` | 0.999591 | |

Only rows 0..6 degrade in the last two, and only token 7 in any of them. The
`[52,64)` and `[49,64)` drop is **not** a mask leak: both spill into the 8-row
`aie::mmul` block `[48, 56)`, which holds token 7's real key at row 48.
`aie::mmul<8,8,8,bf16>` truncates its operands to bfp16 — one shared exponent
per 8 rows — so a 1000.0 anywhere in that block wrecks the mantissa of the
real row beside it. That is an artifact of a deliberately absurd input, not of
the mask. Poison confined to a block boundary is invisible, which is the
answer to the question above.

### 6.2 `kernels/attn_kv.cc`

Same change. Its loop is

```c
int i = L[token];
do { ... i -= 16; } while (i > 0);
```

and it must become a `for j < n_chunks`. Its `s` input is zero for a masked
chunk (the qk core zeroes it), so `calculate_l` adds nothing and `attn_fv`
accumulates zero — but confirm, and check the `c` correction path the same way.

Note both kernels carry `static bool is_ksc_ping` / `is_svc_ping` /
`static int token` **across dispatches**. Once the counts agree these stay in
lockstep with the memtile's free-running chain; do **not** reset them to a
fixed value at token 0, because the memtile's parity at pass start depends on
the total chunks of the previous pass and will not be `false`.

### 6.3 Getting `n_chunks` to the kernels — the host change

*Superseded — a third option was taken. See "What was built" at the head of
§6: `L_ty` shrank to `(1,)` and the host writes `L_max = ctx + L_TOKENS`
directly, so neither of the two below applies. The RTP-churn warning at the
end of this section is still the standing rule.*

`n_chunks = ceil((ctx + L_TOKENS) / 16)`, which equals `ceil(L[L_TOKENS-1]/16)`
because the host writes `L[t] = ctx + t + 1`. Two options:

- **Cheap, no signature change:** the kernel derives it itself,
  `const int n_chunks = (L[L_TOKENS - 1] + 15) >> 4;`. `L_ty` is already
  `(L_TOKENS,) int32` and every core receives the whole array. No host change,
  no RTP address churn.
- **Explicit:** grow `L_ty` to `(L_TOKENS + 1,)` in `iron/layer_aie.py:50` and
  have `_write_attn_L` write the count into the extra slot. Clearer, but it
  moves the RTP addresses — see below.

Either way, after rebuilding, **re-read `rtp_addresses.txt` and update the
hard-coded `q38npu::attn_rtp_addresses`** in
`FLM_DLL/detail/qwen3_8mtp_npu/verify/npu/attn_sequence.hpp` (the DLL tree was
reorganised; it used to be `qwen3_8mtp_npu_attn_weights.hpp`). A stale RTP
address does not fault — it hangs in the wait for 60 s and then returns
`ERT_CMD_STATE_TIMEOUT`.

## 7. How to know it is fixed

1. `/tmp/probe_cursor` — every cursor `0..47` must sit at its band's baseline,
   with no order-of-magnitude drop at `c % 16 >= 9`. This is the direct test.

   **Done, passes.** Every cursor `0..47` now lands in `0.99997..0.999995`
   with no band structure at all. The `k` and `v` cache cosines were unchanged
   throughout, before and after — the append was never at fault. Cursors 63,
   105, 121 and 200 are clean too. `ctx >= 249` trips this harness's
   `MAX_L = 256` assert, which is the harness, not the bug.

   The bug was also reproduced *inside this design* first, which it had not
   been: the original diagnosis was a `/tmp` probe against the DLL. This
   design's own `LC` test values `{8, 14, 30, 40, 56}` all happen to fall in
   the safe band, which is why its suite never caught this. `LC` = 33, 34 and
   39 (cursors 25, 26, 31) scored `0.9996`–`0.9998` against a safe-band
   baseline of `0.99998`–`0.99999`.

   The test is still sensitive after the fix: `ATTN_POISON` at `ctx = 41`,
   which moves one row in each of chunks 0, 1 and 2, drops it from `0.999992`
   to `~0.99986`.
2. The engine's attention-layer check (since removed) — the "decode reproduces the verify batch" check was
   tightened this session to be **per row** with a `0.9999999` gate (it was an
   aggregate `0.999`, which is why it passed at `0.99952867` while the batch was
   leaking). It currently reports:
   `[0:1.0000000 1:1.0000000 2:0.9999999 3:0.9999999 4:0.9978015 5:0.9981356 6:0.9999587 7:0.9991350]`
   All eight rows must come back at `1.0` to seven places.

   **Done, passes:**
   `[0:1.0000000 1:1.0000000 2:0.9999999 3:0.9999999 4:1.0000000 5:1.0000000 6:0.9999999 7:1.0000000]`,
   worst row 2 at `0.99999991`. Whole suite 35 passed, 0 failed. Build it with
   `verify/npu/attn_sequence.cpp`, not the flat filename in its header comment.
3. `tools/verify_qwen3_8mtp` — M9, M13, M17, M19 must return to the CPU
   reference's numbers in the table above (10 passed, 0 failed).

   **Done, passes.** The engine was switched back to `npu_attention` after 1
   and 2 came back clean (`load_weights()` now picks it whenever there is a
   device), and the driver re-run against the 16 offloaded layers:

   |                    | broken NPU | CPU ref | NPU, fixed |
   |--------------------|------------|---------|------------|
   | M9 verify argmax   | 7/8, acc 7 | 8/8, acc 1 | **8/8, acc 1** |
   | M13 mtp draft      | 1/7        | 7/7     | **7/7**    |
   | M19 hit rate       | 3.6%       | 39.3%   | **39.3%**  |
   | M17 output-neutral | diverged at 2 | 16/16 | **16/16** |
   | overall            | 8 pass, 2 fail | 10 pass | **10 passed, 0 failed** |

   `layer_63_out` cos 0.97384195 and M11 next_token 760 = ref, both unchanged
   from the broken run — as expected, since neither goes through a resumed
   cursor.
4. `make test` — the answer must match the CPU's "The capital of France is
   **Paris**." and the MTP hit rate must return to ~64%. Note wall-clock swings
   +/-10% run to run; never read a single run as a speed result.

   **Done, passes.** "The capital of France is **Paris**." — identical to the
   CPU, where the broken offload produced "Paris, France.\n2025-09-06.". MTP
   hit rate 64.3% (9/14), decode 5.494 s, prefill 3.061 s, against the broken
   run's 14.3% and 30.9 s.

All four gates pass: the cursor sweep, the DLL's own NPU attention layer, the
milestone driver and the runtime. The offload is on by default.

There is a second, much smaller and unrelated problem to look at afterwards:
per verify row the NPU is 240.6 ms against the CPU's 208 ms.
