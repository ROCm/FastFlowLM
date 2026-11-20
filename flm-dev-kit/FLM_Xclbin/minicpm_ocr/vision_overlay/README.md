<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# vision_overlay: MiniCPM-OCR vision mm + vision attention in one xclbin

MiniCPM-OCR's vision tower runs every block as `mm_a -> attention -> mm_a` on two different xclbins, so each
image pays 2 hardware-context switches per block, 27 blocks: ~55 switches, ~3 ms each, ~175 ms per image
(measured 2026-11-04 with an attn/proj run-twice probe: mm -> attn 2.7-3.6 ms, attn -> mm 2.6-3.4 ms). This folder
puts both ops in one xclbin; the host switches between them by writing registers, the method of
`FLM_Xclbin/Qwen3_5_2B/vision_overlay` (on main) and `Hy_mem_30BA3B/prefill_overlay`.

Design B of `vision_mm` (only `merger.fc2`, once per image) stays its own xclbin.

## Build

Peano only. Same environment as `../README.md` ("Peano"), then:

```
make            # standalone designs (../vision_mm design A, ../vision_attn), layout, base xclbin, switch tables
make pm         # program size of one core (limit 16384)
make engine     # copy host/*.hpp into FLM_DLL/detail/minicpm_ocr_npu/image/npu
```

`build/xclbins/vision_overlay.xclbin` + `host/op_configs.hpp`. **After any rebuild, run `make engine` and rebuild the
engine**: the switch tables must match the xclbin and the standalone designs they came from.

## How it works

* **Base xclbin** (`iron/overlay_aie.py`): all 32 cores run one program (`kernel/fused.cc`: both kernels, unchanged,
  `#include`d from `../vision_mm/kernel/mm.cc` and `../vision_attn/kernel/attn.cc` in their own namespaces, plus a
  dispatcher), a 3 KB stack, one buffer area ("arena") per core, and core lock 10 as the go lock. No routes and no
  DMA chains.
* **Switch tables** (`host/op_configs.hpp`): `aiecc --aie-generate-txn` on the two *standalone* designs, turned into
  per-op register lists (routes, shim mux, BDs, lock values, channel starts) by `tools/extract_txn2.py` (copied
  unchanged from the Qwen3.5-2B overlay).
* **Each run** (`host/vision_overlay.hpp`): the engine's own sequence (`gen_vision_mm_seq` / `gen_vision_attn_seq`,
  unchanged) with a prefix in front: the switch, only if the array is set up for the other op; then, every run, the
  op number and call count to all 32 cores (words at 0xE020) and the go lock set to 1.

### Choices and tricks

* **Every buffer at its standalone address.** Peano's stack frames are small (attention < 1.2 KB, mm < 1 KB), so both
  ops fit the mm design's 3 KB stack, and then both designs' buffers can stay exactly where their standalone builds
  put them (they overlap, which is fine: only one op runs at a time, and each run rewrites its own RTP words).
  Two things follow. The standalone designs' register dumps are already right for the overlay, so no IRON DMA code
  was ported. And bank placement, so speed, is the standalone one.
  `tools/gen_layout.py` reads the addresses from the standalone builds (and checks all 32 cores agree), so a
  standalone rebuild that moves a buffer moves the overlay with it. `tools/check_arena.py` checks the base build:
  every arena at 0x0C00, the program's own data (attention's 896 B of arrays) after it.

  | core memory | use |
  |---|---|
  | 0x0000-0x0C00 | stack |
  | 0x0C00-0xC880 | mm A buffers (RTP 0x6000) |
  | 0x2000-0xE020 | attention buffers (RTP 0xE000) |
  | 0xE020-0xE028 | dispatcher words: op, calls |
  | 0xE040-0xE400 | program data |

* **BDs are rewritten at every switch.** Both standalone designs number their BDs from 0, and on the odd-column mem
  tiles attention alone uses 22 of the 24 BDs the even channels can use, so the two ops cannot each have their own.
  A switch writes the target op's BDs (2,304 words for mm, 2,816 for attention) on top of routes, locks and starts.
  The core BDs (6 per op, 16 available) could be given separate numbers to cut ~1,100 words if switches ever matter.
* **A prefix every run, not only at a switch.** The standalone kernels have no go lock: a standalone core just calls
  its kernel forever and each call starts when data arrives. Here the core must know which op to run before it
  waits on data, so every run writes the op and the call count and then sets lock 10. Call count: mm, `N / tile_N`
  (one `mm()` call per output strip); attention, `segments x heads / 4` (one `attn()` call per head group).
  The op's RTP words are written after the go lock (in the op's own sequence); that is safe because neither kernel
  reads them before its first input arrives.
* **The rounding mode is reset before mm.** `attn()` sets round-to-nearest-even and leaves it set; `mm()` never sets
  one, so standalone it runs in the hardware default (floor). Without `aie::set_rounding(floor)` in the dispatcher
  the encode differed from standalone by relL2 ~8% (only ~2% of the outputs equal); with it, bit-identical.
* **No ping/pong reset after a switch.** A switch restarts every DMA chain at its first (ping) buffer. Under Peano
  both kernels keep their ping/pong flags as locals that start at ping on every call (each call flips them an even
  number of times, a `static_assert` in each kernel), so they agree with the restarted chains without help.

### In the engine (`FLM_DLL/detail/minicpm_ocr_npu`)

Used when `vision_overlay.xclbin` is in the model's xclbin folder (with `vision_mm_b.xclbin`);
`FLM_MINICPM_OCR_VISION_OVERLAY=0` keeps the three-xclbin path (the A/B arm). `vision_attn.xclbin` and
`vision_mm_a.xclbin` are then not loaded.

* **The switch follows the real state.** `overlay_state` (`image/npu/vision_overlay_state.hpp`) holds the op the
  array is set up for. Each overlay run takes its prefix from it and updates it; a design-B run or a failed run
  resets it, so the next run switches fully. `patch_embed`, the first op of every encode, always switches fully
  (the language model's xclbins may have run since). A fixed prefix per op would be wrong: with views of two sizes
  (720p, 1080p) attention runs twice in a row, and its second run follows attention, not mm.
* **One app per (op, predecessor).** A changed sequence makes `npu_app` rebuild its ELF, so each prefix gets its
  own app; in practice every op always sees the same predecessor, so each op builds one.

## Results (2026-11-04, Strix Halo test machine)

Encode output bit-identical to the shipped Chess three-xclbin path at all three sizes, including the attention ->
attention case. `verify_minicpm_ocr_vision --bench-only`, best warm encode, 3 interleaved rounds:

| image | Chess, 3 xclbins (shipped) | Peano, 3 xclbins | Peano overlay |
|---|---|---|---|
| icon (1 view 32x32, 256 tokens) | 453-471 ms | 402-422 ms | **277-283 ms** |
| 720p (7 views) | 2,062-2,333 ms | 2,040-2,211 ms | **1,862-1,935 ms** |
| 1080p (9 views) | 2,650-2,805 ms | 2,663-2,700 ms | **2,391-2,590 ms** |

Device time for the icon: attention 158 -> 64-70 ms, mm ~250 -> 163-178 ms (the context switches gone).
Core program 12,736 B (+ 896 B data).

End to end (FastFlowLM `test_minicpm_ocr_npu`, 960x720 test image, 5 views): every turn's text identical with the
overlay on and off; vision 2,033 / 1,573 ms against 2,260 / 2,091 ms (two rounds; single app runs are noisy).

## Limits

* Another process using the NPU in the middle of an encode loses the host-written configuration; the next runs wait
  for DMA that never comes. Same as the Qwen3.5-2B overlay and the engine's decode.
* `PROFILE`-style cycle counting is not available (Peano has no `get_cycles()`).
* CI builds only `make bitstream`; the engine keeps its own copies of `host/op_configs.hpp` and
  `host/overlay_layout.hpp`, and `make bitstream` fails (`check_engine`) when this build's copies differ from them.
* Everything here lives in `namespace minicpm_ocr_overlay` (`vision_overlay` is an alias). The Qwen3.5-2B flash engine
  has an overlay with the same names, and FastFlowLM loads both engines into one process: on Linux the switch tables
  (`inline` variables) and the statics in `gen_switch()` are `STB_GNU_UNIQUE`, one copy per process for both libraries,
  whatever `-Bsymbolic-functions` says. With the old global names the Qwen3.5-2B flash overlay ran MiniCPM-OCR's tables
  and hung on every image (FastFlowLM PR #983, Linux XRT and HRX; Windows DLLs are not affected). Check with
  `nm -D --defined-only <lib> | grep ' u '` that no per-engine table is shared with another engine library.
