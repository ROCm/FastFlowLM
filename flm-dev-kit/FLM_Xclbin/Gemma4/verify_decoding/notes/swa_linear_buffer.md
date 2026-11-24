<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# SWA linear window buffer (plan S4.1 re-scoped by the D6 change), engine side

Code: `FLM_DLL/detail/gemma4e_dflash/` (header `gemma4e_dflash_swa.hpp`).

## Contract (what the 8-row verify host path uses)
- Each non-skip SWA layer buffer holds `kSwaBufferRows = 2048` rows per plane (K plane then V plane, plane stride `2048 * SWA_DK`); skip layers use the owner's buffer.
- The engine keeps one `swa_base` (the same for all SWA layers); the row of position `p` is `p - swa_base`.
- Append: decode writes row `ctx - base`; a verify batch writes rows `ctx - base .. ctx + 7 - base`.
- Replay (single token): the last `L_local = min(L, 512)` keys, L = keys including the new token: ONE contiguous range starting at row `(L - L_local) - base`, padded to a 16-row multiple (rows past the end are ZERO).
  This is exactly the ascending-position key order of the old ring replay: same 16-key chunks, decode kernels unchanged, `L_local` RTP unchanged.
- Replay (8-row verify xclbin, attn8 contract): `S = floor16(max(0, ctx - 511))`, range `[S - base, ctx + 8 - base)`.
- Compaction: when `L - base + 32 > 2048` (`swa_needs_compaction`) the host moves the rows of positions `[floor16(max(0, L-512)), L)` to the front of every SWA plane, zeroes the stale rows behind them and sets `base = floor16(max(0, L-512))`. Cost 2.0-4.5 ms (20 layers x 2 planes, DDR memmove), once per about 1490 tokens (first at ctx 2017, then 3521, ...).
  Compaction happens only between steps. A verify batch compacts BEFORE it runs (`dflash_verify`: `swa_needs_compaction(ctx0 + n + 1)` -> `swa_compact(ctx0)` + `set_context_length(ctx0)` because the instruction sequences bake the row offsets in) and never inside it: a commit may roll the context back to ctx0+1 and the next window needs positions down to ctx0-510.
- Rows past the context must be ZERO in every layer buffer (global and SWA): the attention kernels read whole 16-row chunks and stale finite rows change the rounding. Commit zeroes the rejected rows (`ctx0+n_keep .. ctx0+n`), compaction/restore/prefill zero what they leave behind. `FLM_DFLASH_CHECK_SWA=1` scans for violations after every commit.
- Saved state / checkpoint: the last `min(L,512)` rows, oldest first, K rows then V rows, each plane 512 rows (`swa_window_elems()` = 512 * (SWA_DK + SWA_DV)); `write_state`/`restore` lay them back at `base = floor16(max(0, L-512))` and zero everything else up to the old high-water mark.
- Prefill: `gemma4e_prefill_shape` carries `swa_base_old/new/zero_hi`; the decode buffer gets the rows with position >= new base (old window rows are relocated when the base moves); the prefill scratch cache stays absolute-position.
- Stress build: `-DFLM_SWA_BUFFER_ROWS=640` (via `FLM_OVERRIDE_FLAGS`) compacts every ~81 tokens; `static_assert(kSwaBufferRows >= 576)`.

## Bugs found by the rigs (both fixed)
1. Compaction inside a verify batch + commit rollback: `first_row < 0` assertion in `_move_kv_cache` (window needed positions already dropped).
2. After a pre-verify compaction the instruction sequences generated for the previous `set_context_length` still had the old base baked in: the first verify row was appended at the old row and the replay used the old base (found by `FLM_DFLASH_CHECK_SWA`: a stale non-zero row behind the context; streams diverged 15-35 tokens later).

## Test results (all with the real Bolt package, greedy)
See ledger.md rows of 2026-10-29 (S4.1): compare_tags 5/5; long_compare (3010-token story, 3978-token prefill + 900 decode) identical; engine rig 4600 tokens vs the unmodified base engine identical (2 compactions); state round trip + checkpoint/restore identical; spec rig chat 12x128, long 3x128, restore identical (AL 3.09 / 2.35 unchanged); stress 640-row build: 4600-token stream identical to base engine with 42 compactions, state identical, spec chat/long/1200-token identical with compactions inside speculative runs; prefix cache hit with 2964 cached tokens identical; flm bench plain decode unchanged (13.08 vs 13.10 tok/s at 1k, 12.15 vs 12.18 at 4k, 9.49 vs 9.48 at 16k, 7.34 vs 7.35 at 32k).
