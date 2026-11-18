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
| `vision_model_def.hpp` | `minicpm_v_vision_desc`: geometry from the packaged config's `MINICPM_V_*` keys |
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
  (notes/tricks/2026-10-05-minicpm-v-position-table-is-bucketize.md).
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
at no measurable cost. `FLM_MINICPM_V_VISION_DEVICE_GELU=1` is the A/B.
(notes/tricks/2026-10-05-vision-mm-fused-gelu-is-the-dominant-device-error.md)

## Verification

Reference: `models_simple/capture_minicpm_v_vision.py` -> `vision_ref.safetensors`
(960x720 photo, 5 views of 28x36, every stage tapped, plus the HF prompt, its
canvas positions and HF fp32's greedy answer).

```
# tower, stage by stage (from this directory's parent)
g++-13 -O3 -march=native -fopenmp -std=c++20 -ffast-math -I. -I../../include -I../q4nx \
  -I/opt/xilinx/xrt/include tools/verify_minicpm_v_vision.cpp image/vision_tower.cpp \
  image/vision_ops.cpp image/vision_glue.cpp image/mrope.cpp image/npu/vision_mm_sequence.cpp \
  image/npu/vision_attn_sequence.cpp -L../../external_libs -L/opt/xilinx/xrt/lib \
  -lxrt_coreutil -laiebu -luuid -lpthread -o /tmp/verify_minicpm_v_vision
FLM_MINICPM_V_VISION_FP32=1 /tmp/verify_minicpm_v_vision --backend cpu   # logic
/tmp/verify_minicpm_v_vision --backend cpu                                # bf16 floor
/tmp/verify_minicpm_v_vision --backend npu [--bench]                      # the array
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

## The streamed encoder (host work hidden behind the device)

With the NPU backend the tower no longer runs one op at a time. Inside a block
everything from the attention output to the next block's qkv is row-independent
(proj, residual, LN, fc1 + gelu, fc2, residual, LN, qkv), so `vision_tower.cpp`
pushes that stretch through the array as chunks of ~1024 rows, three in flight:
a worker thread (`npu/vision_lane.hpp`) runs the dispatches in order while the
calling thread does the narrow / widen / LN / residual of other chunks. Each
in-flight chunk owns a staging slot (x and y BOs), so nothing is shared.

The activations also stay in bf16 where they already are bf16: the qkv output
goes to the attention BOs and the attention output to the proj input without a
widen-to-fp32 / narrow-back round trip, and fc1's gelu feeds fc2's x in one pass
(`npu/vision_bfp16.hpp`). Every fused stage is an exact re-expression of the
two-pass one (same roundings, same denormal flush), and every row sees the same
arithmetic in the same order whatever the chunking, so `image_embeds` are
**bit-identical** to the sequential path (checked by hash on every encode).

Context switches (mm <-> attn, two per block) are ~2.4 ms each and cost ~130 ms an image whatever the host does.
Single-context reconfiguration (default on; `FLM_MINICPM_V_VISION_SINGLE_CTX=0` switches it off) replaces them: the attention dispatches of a call run in `vision_mm_a`'s
hw_context between two in-stream reconfigurations of the array
(`npu/vision_attn_reconfig.hpp`; the mechanism, its lock quiesce and its safeguards are
`../minicpm_v_reconfig.hpp`'s and `notes/tricks/2026-11-06-...`). A call with several
dispatches (a 720p image attends its source view and its slices separately) enters the
attention configuration once and leaves it once. One reconfiguration round trip costs 1.5 ms
against 4.9 ms for two context switches (measured on the shipped overlays); an image saves
80-100 ms (448x448: 423 -> 333 ms TTFT, 1080p: 1727 -> 1647 ms). Gates, each falling back to
the separate contexts: built (no preemption, CDOs parse), validated (a known-answer test at
load: attention rows through the round trip and through a two-dispatch call must equal the
separate-context result bit for bit, and a design-A matmul before and after must too), and
per call (not disabled, this process owns every hw_context on the NPU). A failure disables
it for the process (one kill switch shared with `FLM_MINICPM_SINGLE_CTX`, the language
model's attention) and the dispatch is redone through `vision_attn`'s own context. Off on
Windows and the HRX backend.

`FLM_MINICPM_V_VISION_PIPE=0` selects the sequential path (the A/B arm, and what
the verify drivers use, since they tap every intermediate);
`FLM_MINICPM_V_VISION_CHUNK=<rows>` sets the chunk target. The HRX build keeps
the sequential path (`streamable()` is false there). `make VISION_PROFILE=1`
now prints the stage table per encode, with the streamed path's device wait.

## Knobs

`FLM_MINICPM_V_VISION=0` (no tower: images rejected), `FLM_MINICPM_V_VISION_FP32=1`
(host ops in fp32, the precision control), `FLM_MINICPM_V_VISION_MM_PREROUND=0`,
`FLM_MINICPM_V_VISION_DEVICE_GELU=1`, `FLM_MINICPM_V_VISION_BACKEND=cpu` (diagnostic
host tower in the engine, for an answer-level A/B), `FLM_MINICPM_V_VISION_THREADS`,
`FLM_MINICPM_V_VISION_PROFILE=1` (engine prints vision / prefill split and a hash of `image_embeds`),
`FLM_MINICPM_V_VISION_PIPE=0`, `FLM_MINICPM_V_VISION_CHUNK`, `FLM_MINICPM_V_VISION_SINGLE_CTX=0` (off switch of the default-on single-context attention, see above),
`FLM_MINICPM_V_VISION_MM_P=0` (use `vision_mm_c.xclbin` even when `vision_mm_p.xclbin` is present),
`FLM_MINICPM_V_VISION_ATTN_KV=0` (use `vision_attn.xclbin` even when `vision_attn_kv.xclbin` is present;
see `FLM_Xclbin/MiniCPM_V/vision_attn_kv/README.md` -- kv is faster and not bit-identical).
