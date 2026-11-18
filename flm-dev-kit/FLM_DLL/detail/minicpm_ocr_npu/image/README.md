<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# MiniCPM-V-4.7-1B vision tower

The SigLIP tower that turns an image's views into the rows the language model
reads in place of `<|image_pad|>`. A port of Qwen3.8-27B's tower
(`qwen3_8mtp_npu/image/`, read its two READMEs first): the geometry is
identical -- 27 blocks, D 1152, 16 heads x 72, MLP 4304 padded to 4608 -- so its
two bitstreams serve this one **unchanged**:

```
FLM_Xclbin/Qwen3_8/vision_mm     design A (m=64,k=384,n=48), B (m=32,k=384,n=128)
FLM_Xclbin/Qwen3_8/vision_attn   head_dim 72, full bidirectional, 512-row rounds
```

Every matmul (115 of them) and every full-view attention run on the array. The
host does the O(L*D) glue and one attention: the vit_merger's, which is 4x4 over
2x2 windows and has no business on a 512-row kernel.

## Layout

| file | what it is |
| --- | --- |
| `vision_model_def.hpp` | `minicpm_ocr_vision_desc`: geometry from the packaged config's `MINICPM_OCR_*` keys |
| `vision_weights.hpp` | mmap'd view of `vision_weight.q4nx`, shape-checked (4-D patch embed) |
| `vision_ops.hpp/.cpp` | **the seam** (`mm_op`, `attn_op`, `op_factory`) and the host ops -- Qwen3.8's, plus `prepare()`, a row stride, and `run_views()` |
| `vision_glue.hpp/.cpp` | layernorm, the bucketized position lookup, 2x2 regroup + mean, the windowed attention |
| `vision_tower.hpp/.cpp` | the tower's control flow (Qwen3.8's `image_cpu`, renamed) |
| `mrope.hpp/.cpp` | the language model's **canvas** m-rope for image prompts |
| `vision_common.hpp` | bf16, mmap safetensors, error stats, fp32<->bf16 SIMD (a copy, namespaced) |
| `vision_par.hpp`, `vision_profile.hpp` | thread count rule, `make VISION_PROFILE=1` stage table |
| `npu/` | the device ops, both sequence generators, the bfp16 pre-round, the factory |

## What is MiniCPM's own

* **Views, not images.** The processor slices a picture into a source view plus
  up to 9 slices, each its own grid (a 1280x720 image: one 24x44 view and six
  28x36). Matmuls batch every view; attention is block-diagonal over views.
  `run_views()` puts consecutive same-length views in one `vision_attn` dispatch,
  one segment each (`gen_vision_attn_seq`'s `n_seg`), and `npu_vision_attn_op`
  keeps one app per (length, count) so alternating shapes do not rebuild ELFs.
* **Raster order.** Patches arrive raster within a view, so the two 2x2 merges
  are real gathers (`regroup_2x2`), not Qwen3.8's free reinterpretation.
* **Position table:** a SigLIP NaViT *bucketize* into a 70x70 table, not a
  bilinear resample. Ties follow torch's float32 `arange`, reproduced exactly
  (the position table is a bucketize).
* **Patch K = 588** (3x14x14), not a multiple of 384: the device op zero-pads K
  to 768 in both operands and pre-rounds the padded rows.
* **The vit_merger after block 6:** LN -> qkv -> 2x2 windowed attention (host) ->
  proj -> +x -> 2x2 merge -> LN(4608) -> fc1 4608->17280 (gelu_tanh) -> fc2 ->
  + mean of the 4. Blocks 7..26 then run on a quarter of the patches.
* **Projector:** post_layernorm (`merger.norm`), 2x2 merge, LN(4608), fc1
  (erf gelu, host epilogue), fc2 on design B (N 1024).
* **No vision RoPE.**

## One deviation from Qwen3.8: the tanh gelu is on the host

Scored op by op on the reference's own inputs, every device matmul without an
activation lands at ~2.7x the bf16 host op's error; the two with the fused device
gelu landed at 11x (block fc1) and 19x (vit_merger fc1). So fc1 binds `RTP_NONE`
and the gelu runs in the widen every dispatch already does (`widen_gelu_tanh_par`),
at no measurable cost. `FLM_MINICPM_OCR_VISION_DEVICE_GELU=1` is the A/B
(the fused device GELU was the dominant device error).

## Verification

Reference: `models_simple/capture_minicpm_ocr_vision.py` -> `vision_ref.safetensors`
(960x720 photo, 5 views of 28x36, every stage tapped, plus the HF prompt, its
canvas positions and HF fp32's greedy answer).

```
# tower, stage by stage (from this directory's parent)
g++-13 -O3 -march=native -fopenmp -std=c++20 -ffast-math -I. -I../../include -I../q4nx \
  -I/opt/xilinx/xrt/include tools/verify_minicpm_ocr_vision.cpp image/vision_tower.cpp \
  image/vision_ops.cpp image/vision_glue.cpp image/mrope.cpp image/npu/vision_mm_sequence.cpp \
  image/npu/vision_attn_sequence.cpp -L../../../../src/lib/xrt -L/opt/xilinx/xrt/lib \
  -lxrt_coreutil -laiebu -luuid -lpthread -o /tmp/verify_minicpm_ocr_vision
FLM_MINICPM_OCR_VISION_FP32=1 /tmp/verify_minicpm_ocr_vision --backend cpu   # logic
/tmp/verify_minicpm_ocr_vision --backend cpu                                # bf16 floor
/tmp/verify_minicpm_ocr_vision --backend npu [--bench]                      # the array
# engine end to end on the reference prompt (host_ref/)
make -C host_ref && run_vision_e2e <model_dir> <ref> <exec_path> <model_name>
```

| | `image_embeds` cos | relL2 |
| --- | --- | --- |
| host, fp32 | 1.0000000 (all 50 stages exact) | 0.00000 |
| host, bf16 | 0.9998254 | 0.01869 |
| array (`mm=npu attn=npu`) | 0.9968140 | 0.07981 |

Canvas positions: 0 of 1047 differ from HF. Controls that collapse: transposed
position walk, one attention segment across views (relL2 0.047 vs 0.002 -- cosine
barely moves, 0.9989), full-view instead of windowed vit_merger attention,
shifted rows; the projector's erf beats tanh in fp32.

**Task score.** `run_vision_e2e`: the engine's answer on the reference prompt is a
correct description on the array tower and on the exact host tower alike; both
diverge from HF fp32 at token 1, which is the Q4 language model, not the tower.
Through `flm run` (app branch `minicpm-s7b-app`), 9/9 image questions answered
correctly against HF fp32's 9/9 (1, 5 and 10 views; counts, colours, objects).

## Cost (720p = 7 views, 7104 patches, 444 rows)

| | ms |
| --- | --- |
| device `vision_mm` (~3.4 TFLOP, ~5.9 TFLOPS) | 576 |
| device `vision_attn` | 328 |
| host (narrow/pre-round, syncs, widen+gelu, glue) | ~200 |
| **encode, warm** | **~1100-1130** |

The tower is array-bound. Attention is compute-bound at ~0.22 ms per 512x512
round, and its 512-row granularity wastes ~4x on post-merge views (~256 rows
padded to 512) and ~2x on a 1056-row source view; that is a kernel property.
`downsample_mode: "4x"` is not a speed lever: it skips the cheap vit_merger and
runs blocks 7..26 on 4x the patches.

## Run failures on the overlay (2026-11-10)

`vision_overlay.xclbin` keeps the attention and the design-A matmuls on one xclbin. The host
switches between them by writing routes, BDs and locks in front of a run, and only what differs
from the op set up last (`overlay_state::current`). If the driver reloads the context between two
runs (Krackan: a context idle at 3 timer ticks; Strix Halo: whole-NPU autosuspend after 5 s; Windows:
not known), that host-written setup is gone. The next run then waits until the driver's timeout
(state 8: ~6-8 s on Krackan, 60 s on Strix Halo) and resets the array. The 2026-11-10 Windows stress
test hit this (`block12.proj` ... `dispatch state 8`). The same problem in Hy-MT2 was fixed the same way (WP12).

**What the ops do now** (`overlay_state::run_checked`, used by `vision_mm_op` on design A and by
`vision_attn_op` on the overlay): if a run does not complete, or XRT throws while waiting, forget
the setup (`OP_NONE`) and run the same op again from a full switch. Normal runs take the same
path, so results are byte-identical.

**Why up to three tries, with pauses.** Measured on Strix Halo with a passthrough stand-in
(one real 60 s timeout per test): right after a
timeout, the **first** next run on the same context fails at once (state 5, 0.2 ms); the second and
all later runs complete with correct data. After a 2 s pause, or on a new hardware context, the first
run already works. A new `xrt::kernel` on the same context does not help. So:

| try | pause before it | tried when |
| --- | --- | --- |
| 1 | 0 | always, after the failed run |
| 2 | 100 ms | try 1 failed fast (not a timeout) |
| 3 | 1 s | try 2 failed fast |

A second **timeout** stops it: the op throws as before. A context that keeps timing out is truly
stuck; sending it more work is what came before the 0x139 bugcheck in the Windows test. One failure
costs the driver's timeout once; the tries themselves take milliseconds.

Off the overlay (the three-xclbin path, design B) nothing changes: a separate xclbin has no host-written
setup to lose.

**Test hooks** (overlay runs counted from 1 per engine, comma-separated lists):
`FLM_MINICPM_OCR_VISION_FAKE_FAIL_AT=<n,...>` treats run n as failed after it ran (the run-again
path without a real timeout); `FLM_MINICPM_OCR_VISION_IDLE_BEFORE=<n,...>` waits 7 s before run n, so
on Strix Halo the NPU autosuspends and run n really times out.

Checked with `tools/verify_minicpm_ocr_vision --bench-only` (icon, 720p, 1080p, `VISION_SAVE`), against the
build before this change: normal runs byte-identical, speed within noise; 20 faked failures (attention,
proj, fc1, patch_embed) byte-identical; 2 REAL timeouts via `IDLE_BEFORE=1165,1844` (block20.proj in
the last 720p run, attention L 1024 x 8 in the last 1080p run): each ran 7 s idle + ~60 s timeout, then
try 1 completed at once, and all outputs were byte-identical (Strix Halo, amdxdna 7.0.0-rc1).

## Knobs

`FLM_MINICPM_OCR_VISION=0` (no tower: images rejected), `FLM_MINICPM_OCR_VISION_FP32=1`
(host ops in fp32, the precision control), `FLM_MINICPM_OCR_VISION_MM_PREROUND=0`,
`FLM_MINICPM_OCR_VISION_DEVICE_GELU=1`, `FLM_MINICPM_OCR_VISION_BACKEND=cpu` (diagnostic
host tower in the engine, for an answer-level A/B), `FLM_MINICPM_OCR_VISION_THREADS`,
`FLM_MINICPM_OCR_VISION_PROFILE=1` (engine prints vision / prefill split),
`FLM_MINICPM_OCR_VISION_OVERLAY=0` (three-xclbin path), and the two test hooks in "Run failures".
