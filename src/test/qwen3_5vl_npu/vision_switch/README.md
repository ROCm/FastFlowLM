# Qwen3.5-2B vision encoder: how much do xclbin switches cost?

The vision encoder uses two xclbins, `vision_mm.xclbin` and `vision_attn.xclbin`. Each of the 24 layers runs
qkv (mm) -> attention -> o, fc1, fc2 (mm), so every layer switches the NPU twice (mm -> attn, attn -> mm).
This folder measures what those switches cost, to decide whether an overlay xclbin is worth building. The
overlay idea is the one in `FLM_Xclbin/Hy_mem_30BA3B/prefill_overlay` on IRON branch `hy_mem_30A3B`: one
xclbin holds both ops and the host rewrites routes/BDs to switch, ~0.12 ms per switch on Strix Halo.

## Build and run

```bash
cd /scratch/michyu/FastFlowLM/src/test/qwen3_5vl_npu/vision_switch
make -j 12          # timed engine copy + vision_bench (output in src/build/test/vision_switch)
make test           # 4 image sizes x 3 runs, normal then VISION_PROBE=1, ~2 min; prints the summary
make test SIZES="640x480" ITERS=1    # quick check
```

Check `fuser -v /dev/accel/accel0` first: another session may share the NPU and make single runs slow.

## What is in here

| file | what |
|---|---|
| `engine/` | Snapshot (2026-11-02) of the 2B engine copy `FastFlowLM_IRON/FLM_Xclbin/Qwen3_5_2B/engine` (IRON `ba3fc198b` + the prefill session's diff + lm_head in layer.xclbin), with timers added to `qwen3_5vl_image.cpp` by `patch_timers.py`. Nothing else changed. |
| `engine/vision_timer.hpp` | Adds up the time between `vt.mark()` calls per stage; prints a table to stderr at the end of `encode()`. |
| `vision_bench.cpp` | Loads qwen3.5:2b, runs "Describe this image in one sentence." with one image, 24 tokens, greedy. Prints TTFT and the answer. Text-only run first as a baseline. |
| `make_image.py` | Makes a test PNG (gradient + red ellipse + blue box) of a given size. |
| `summarize.py` | Picks the median run per size and mode from the log and prints each stage. |
| `patch_timers.py` | The exact edits made to `qwen3_5vl_image.cpp` (to re-apply on a newer engine copy). |

`make test` runs the bench through `LD_LIBRARY_PATH=<build>/engine`, so it uses the timed `.so`; nothing in
`src/lib/xrt` or the IRON tree is touched.

## How the switch cost is measured

`VISION_PROBE=1` runs the qkv, attention and o calls a **second time right away**, with the same inputs
(the outputs are just written again with the same values, so the answer does not change). The first call
pays for a switch if one is needed, the second never does:

- `attn_run - attn_run_again` = switch mm -> attn
- `o_run - o_run_again` = switch attn -> mm
- `qkv_run` vs `qkv_run_again` is the control: qkv comes right after the previous layer's fc2 (same xclbin),
  so the two should be equal, and they are (within 2-5 %).

## Results (2026-11-02, xsjnuckrck04, Krackan, Performance power mode, NPU not shared during the run)

Image size -> vision tokens L (smart_resize to multiples of 32) -> padded L used by the kernels.

| image | L (padded) | switch mm->attn | switch attn->mm | switches per image | encode | switch share of encode | TTFT | switch share of TTFT |
|---|---|---|---|---|---|---|---|---|
| 256x256 | 256 (512) | 3.47 ms | 2.86 ms | **152 ms** | 257 ms | **59 %** | 0.673 s | 23 % |
| 640x480 | 1200 (1536) | 3.46 ms | 2.94 ms | **153 ms** | 433 ms | **35 %** | 1.028 s | 15 % |
| 1024x768 | 3072 (3072) | 3.47 ms | 2.91 ms | **153 ms** | 781 ms | 20 % | 1.763 s | 8.7 % |
| 1920x1080 | 8160 (8192) | 3.42 ms | 2.88 ms | **151 ms** | 2428 ms | 6.2 % | 4.49 s | 3.4 % |

Text-only TTFT for the same prompt is 0.37 s. All answers were right ("A red oval sits on a gradient
background ...") and were the same with and without the probe.

**The switch cost does not depend on the image: ~6.35 ms per layer, ~152 ms per image.** That is 2.5x the
2.5 ms guessed from the text prefill.

### Where the time goes at 640x480 (L 1200, encode 433 ms)

| part | ms | share |
|---|---|---|
| xclbin switches (48) | 153 | 35 % |
| mm work (qkv 29, o 12, fc1 43, fc2 30, merger 4) | 118 | 27 % |
| before the layer loop (`pre_layers`) | 62 | 14 % |
| attention work (1.98 ms/layer) | 48 | 11 % |
| host work in layers (layer norms, bias + RoPE, residuals) | 21 | 5 % |
| buffer syncs (weights ~8 ms of it) | 21 | 5 % |

## What this means for the overlay

If an overlay switch costs about what it does on Strix Halo (~0.12 ms; not measured on Krackan), it saves
about **145 ms per image**, whatever the size:

| image | encode now -> with overlay | TTFT now -> with overlay |
|---|---|---|
| 256x256 | 257 -> ~110 ms (-57 %) | 0.67 -> ~0.53 s |
| 640x480 | 433 -> ~290 ms (-34 %) | 1.03 -> ~0.88 s |
| 1024x768 | 781 -> ~635 ms (-19 %) | 1.76 -> ~1.62 s |
| 1920x1080 | 2428 -> ~2280 ms (-6 %) | 4.49 -> ~4.35 s |

Running attention on the CPU for small images is not a good substitute: it only wins at L <= ~512. At
L 1200 the NPU attention takes 2 ms per layer, and the CPU would need tens of ms.

## Other things seen

- **Padding wastes mm work on small images.** L 256 is padded to 512 (2x the mm work) and 1200 to 1536
  (+28 %). At L 256 about 23 ms of the 47 ms of mm is padding.
- **`pre_layers` is large and not yet split up**: 19 / 62 / 91 / 214 ms. It covers patch embedding, position
  embeddings, RoPE tables, making the 7 instruction sequences, and allocating + zeroing + syncing every
  buffer, all redone on each `encode()`.
- **Big images are attention bound.** At L 8160 attention is 49.8 ms per layer, 1.19 s = 49 % of the encode.
- Re-syncing the weights every layer costs only ~8 ms per image, so it is not worth fixing first.
- The first call of each image (layer 0 qkv) also pays a switch from the text `layer.xclbin`. An
  overlay can't remove that one.

## The vision overlay (2026-11-02)

`FastFlowLM_IRON/FLM_Xclbin/Qwen3_5_2B/vision_overlay` puts vision mm and vision attention in one xclbin, and the
host switches between them in ~0.11-0.13 ms instead of ~3.2 ms (design and tricks: its README). The engine copy
here uses it when `VISION_OVERLAY=<path to vision_overlay.xclbin>` is set; without it nothing changes.

```bash
make compare            # both paths, 4 sizes x 3 runs; prints "bit-identical" per size and the answers
make compare SIZES="640x480" ITERS=1
```

Engine copy changes for the overlay (`engine/qwen3_5vl_image.{hpp,cpp}`, `engine/Makefile`):

- `VISION_OVERLAY` set: one `register_xclbin` (the overlay) for all 7 apps, plus `qkv_first_projection_app`.
- After the usual sequences are generated, `vision_overlay::prepend` puts a switch in front: layer 0 qkv gets
  a full switch to mm (the text `layer.xclbin` ran before), attention gets mm -> attention, o_proj gets
  attention -> mm. qkv (layers 1-23), fc1, fc2 and the merger need none.
- `VISION_PROBE` repeats are skipped with the overlay (re-running a sequence that holds a switch is not valid).
- `VISION_DUMP=<prefix>` writes the encoder output to `<prefix>_L<L>.bin`; `VISION_DUMP_STAGES=<prefix>`
  writes qkv / attention / o / fc2 outputs of layers 0-1 (`VISION_DUMP_LAYER=N` adds layer N,
  `VISION_DUMP_ALL_LAYERS=1` adds fc2 of every layer). Used to find a one-row difference at layer 3 (below).
- `engine/Makefile`: `-I` to the overlay's `host/` folder.

### Results (xsjnuckrck04, Krackan, Performance mode, NPU not shared; median of 3)

| image | L | encode two xclbins | encode overlay | saved | TTFT two xclbins | TTFT overlay | output |
|---|---|---|---|---|---|---|---|
| 256x256 | 256 | 256.0 ms | 112.1 ms | 144 ms (-56 %) | 0.711 s | 0.529 s | bit-identical |
| 640x480 | 1200 | 437.9 ms | 277.1 ms | 161 ms (-37 %) | 1.024 s | 0.871 s | bit-identical |
| 1024x768 | 3072 | 803.7 ms | 584.3 ms | 219 ms (-27 %) | 1.792 s | 1.594 s | bit-identical |
| 1920x1080 | 8160 | 2430.6 ms | 2180.2 ms | 250 ms (-10 %) | 4.475 s | 4.232 s | bit-identical |

All answers are the same word for word. Per layer, attention now takes 505 / 1,978 / 6,976 / 47,792 us and
o_proj 384 / 575 / 893 / 2,304 us (with their switch).

Why more than the ~150 ms of switches at large L: attention itself got faster. The standalone kernel added the
row sums (`l += sum`) and computed `1/l` with software float (`float32_add`, `float32_div`); at L 8160 that is
65,536 software adds per core per layer. To fit both kernels in program memory these became a small exact
integer add and a 128-entry table (vision_overlay README).

**The one-row difference.** The first overlay build added the row sums with a vector fp32 add. Outputs were
bit-identical at L 256 and 1200, but not at 3072 / 8160 (final cos 0.99 / 0.976). Stage dumps showed layers 0-2
identical and, at layer 3, one row of one head (row 2040, head 5) about 1 % smaller: the vector add differed
from software float by one fp32 step there, `l` sat on a bf16 boundary, and the floor conversion flipped
`1/l` by one bf16 step. The encoder then spread that tiny change over 20 more layers (final rel 0.14). An exact
integer round-to-nearest-even add fixed it (checked on the host against the CPU's float add: 20.9 M pairs, 0
mismatches except overflow to infinity, which cannot happen here).
