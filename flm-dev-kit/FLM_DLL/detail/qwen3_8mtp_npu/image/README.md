<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# Qwen3.8-27B vision tower

The vision half of the Qwen3.8-27B (MTP) engine: 27 blocks that turn one
image's patches into rows the language model consumes in place of
`<|image_pad|>` embeddings.

Phase 1 ran entirely on the CPU. The folder is laid out so that phase 2 moves
the matmuls and the attention onto bitstreams that already exist, without
touching the tower itself.

**Phase 2 is complete: both the attention and the matmuls are on the NPU**
(`npu/`, verified on hardware against the host ops). The tower reports `mm=npu attn=npu bf16`.
The two knobs that used to take it back — `FLM_Q38_VISION_NPU=0` for the whole
tower and `FLM_Q38_VISION_MM=0` for the matmuls alone — are gone. All three
vision xclbins are required and module 4 throws naming a missing one.
`FLM_Q38_VISION=0` remains and is a different thing: it disables the tower as a
*feature*, for a text-only deployment, rather than choosing an implementation
of it. The host ops (`vision::cpu_mm_op`, `cpu_attn_op`, `cpu_op_factory`) are
still here as the reference the device was scored against — including the
accuracy ladder below, which is where the `_MM=0` A/B's numbers came from.

`image_cpu.cpp` did not change to make any of that happen, which was the point
of the `vision_ops.hpp` seam.

One thing the seam did **not** anticipate: the device needs its *activations*
pre-rounded onto the mmul's bfp16 grid, not just its weights, so
`npu_vision_mm_op::run()` does per-call work on `x` that has no CPU analogue.
See `npu/README.md` — it is an 18x accuracy difference, not a refinement.

---

## Layout

| file | what it is |
| --- | --- |
| `vision_model_def.hpp` | `qwen3_8mtp_vision_desc` — geometry read from `vision_config`, plus `patch_dim()`, `merger_in()`, `token_count()` |
| `vision_weights.hpp` | mmap'd view of `vision_weight.q4nx`; `block(i)` hands out one block's eight tensors |
| `vision_ops.hpp/.cpp` | **the seam**: `mm_op`, `attn_op`, `op_factory`, and the CPU implementations |
| `vision_glue.hpp/.cpp` | layernorm, the bilinear position-embedding resample, residual adds — the O(L·D) work that stays on the host |
| `vision_rope.hpp/.cpp` | 2-D vision RoPE tables and their application to q and k |
| `image_module.hpp` | the abstract tower (`encode`, `tokens_for`, `set_tap`) |
| `image_cpu.hpp/.cpp` | the tower itself, driving the ops through the seam |
| `mrope.hpp/.cpp` | m-rope position ids for a multimodal window, and the image spans in it |
| `npu/` | phase 2 lives here; an `npu::vision_op_factory` is the only thing that needs to appear |

## The seam

`vision_ops.hpp` is the reason this folder is shaped the way it is. Two
operators account for essentially all of the tower's time, and the bitstreams
for both are already built:

```
FLM_Xclbin/Qwen3_8/vision_mm     design A (m=64,k=384,n=48), design B (m=32,k=384,n=128)
                                 y = act(x W^T + b), rtp_act in {none, silu, gelu}
FLM_Xclbin/Qwen3_8/vision_attn   16 heads, head_dim 72, full bidirectional mask,
                                 L padded to whole 512-row chunks
```

Three properties of the interface are load-bearing and should not be
"simplified" away:

* **Matmuls are weight-resident.** `bind()` is called once at load; `run()`
  streams activations only. A per-call weight pointer would force the NPU
  backend to re-upload 959 MB per image, which is the entire cost of the port.
* **Activations cross as fp32 host buffers**, matching this engine's existing
  device ops. The CPU and NPU backends are then drop-in for each other with no
  dtype churn at the call sites.
* **The activation is part of `bind()`, not `run()`.** On the device it is an
  RTP the sequence bakes in, and a mid-sequence RTP rewrite is a host/kernel
  protocol change, not a free parameter.

`op_factory` is chosen in exactly one place (`qwen3_8mtp_npu.cpp`, module 4),
so adding `image/npu/vision_op_factory.hpp` does not touch `image_cpu.cpp`.

### How phase 2 handled `gelu_erf`

`vision_mm`'s `rtp_act` has **no `gelu_erf`**, and `merger.fc1` asks for it.
`npu_vision_mm_op::bind()` binds `none` and leaves the epilogue on the host.
It does **not** substitute `gelu_tanh` — see below; N16 check 7 is what keeps
that true, and it separates the two at relL2 0.0083 (erf) against 0.0134
(tanh) on the HF `merger_fc1` tap.

## Two GELUs

The tower uses both, and they are not interchangeable:

* the 27 block MLPs take `vision_config.hidden_act`, which this checkpoint sets
  to `gelu_pytorch_tanh`;
* `Qwen3_5VisionPatchMerger` hardcodes `nn.GELU()`, the exact-erf form
  (verified in the installed transformers source, `modeling_qwen3_5.py:871`).

So the last thing before the language model sees the image is erf, not tanh.
The two agree to ~1e-3 absolute, so getting it wrong costs a few thousandths of
cosine on the final rows and nothing that reads as a bug. That is precisely why
`activation::gelu_erf` is a separate enumerator rather than a comment.

Scoring this is harder than it looks: measured through `fc2` into
`image_embeds`, the difference is averaged away by a 4608→5120 matmul and sits
below the bf16 activation floor, where which one wins is noise (tanh was ahead
by 6e-7). It only separates when scored against the `merger_fc1` tap — the
activation's own output — and only in fp32, where erf wins by 14× in relative
L2. N14's check is gated accordingly.

## Assumptions worth knowing

* **`ln_eps` is 1e-6.** `vision_config` does not state a layer-norm epsilon, so
  the desc defaults to the `nn.LayerNorm` value transformers constructs the
  tower with. A wrong epsilon here is invisible at every scale that matters.
* **MLP width is 4608, not 4304.** `vision_config.intermediate_size` says 4304;
  the shipped tensors are 4608 wide (zero-padded). The tensor width is
  authoritative and `weights` corrects `desc.D_ff` when it opens the file —
  which is why the tower must be constructed before any op is bound.
* **Patches arrive in merge-block order.** The HF processor's reshape/permute
  puts patches `4s..4s+3` consecutively, which makes the merger's 2×2 regroup a
  free reinterpretation of `[L][1152]` as `[L/4][4608]`. Feeding raster order
  is not a crash: the tower runs and the image is silently scrambled.
* **Attention runs one image at a time.** The true mask is block-diagonal over
  `cu_seqlens` — full within an image, zero across images. A single call over a
  concatenated sequence would let two images attend to each other, and with one
  image that bug is invisible.

## bf16 emulation

`vision::cpu_ops_emulate_bf16()` is **on by default**. This is not a
performance choice: the device consumes bf16 activations, so rounding here
makes the bf16 error common-mode, and phase 2 can score the NPU matmul against
this reference with a bar at cosine > 0.999 instead of somewhere vague.

`FLM_Q38_VISION_FP32=1` turns it off. That is the control that separates a
precision floor from a bug: a stage that only passes in fp32 has a range
problem, not an addressing problem. The env read is cached in a function-local
static, so a process is one mode for its whole life — the two modes are two
invocations, not two calls.

## Verification

**`FastFlowLM/src/test/qwen3_8mtp_npu/verify_preprocess.cpp`** — the *host*
preprocessing, which a tower check cannot test because it feeds the tower the reference's
own `pixel_values`. A wrong patch order there passes every tower check and
still scrambles the image. Run with `make verify`; it scores
ffmpeg + AVX-512 bicubic against PIL + the HF processor at cos 1.000000
(relL2 0.00183), with raster order, missing mean/std and a one-row shift
collapsing to 0.756, −0.410 and 0.852 respectively.

## Cost, and why phase 2 exists

Encoding 256 patches (64 merged tokens) costs ~4.4–5.3 s on the CPU backend,
about 70–83 ms per merged token. In the end-to-end runtime test that is
essentially all of an 82-token prompt's 4.73 s prefill. That number is the
whole motivation for moving `mm_op` and `attn_op` onto the bitstreams.

**256 patches is not the operating point.** `panda.png` is 639×480, and
`smart_resize` at factor 32 (patch 16 × merge 2) gives 640×480 → a 30×40 grid →
**L = 1200 patches, 300 merged tokens**. The captured `vision_ref.safetensors`
is a 16×16 grid, so the reference dump and the runtime test sit at different
sizes, and anything measured on the dump alone is measured 4.7x small.

With both halves ported, the real encode at L = 1200 — measured through this
tower, not a warm microbenchmark (N16 check 10's in-situ arm) — is:

| backend | encode |
| --- | --- |
| `cpu` | 3118–3324 ms |
| `npu-attn` | 2573–2602 ms |
| `npu` | **1633–1942 ms** |

So attention bought ~0.6 s and the matmuls a further ~0.7–0.9 s.

**What was left was this folder's own host code**, and phase 3 took it. ~1 s of
that remaining ~1.7 s was fp32 glue — the norms, the rope, the `gelu_erf`
epilogue, the merger gather, and the fp32<->bf16 conversion at all 111 seam
crossings. `make VISION_PROFILE=1` and `tools/profile_vision.cpp` broke it into
stages, and three changes took encode at L = 1200 to **599 ms**:

| | before | after |
| --- | ---: | ---: |
| encode, L = 1200, default OpenMP policy | 1662 ms | **599 ms** |
| of which device (`vision_mm` + `vision_attn`) | 542 ms | 513 ms |
| of which host | 1120 ms | **91 ms** |

The three are: fusing the fp32→bf16 narrow with the bfp16 pre-round and
vectorising both (`npu/vision_bfp16.hpp`); threading the stages that were
scalar and single-core (`vision_par.hpp`); and **sizing the OpenMP team below
`nproc`**, which alone is 2x, because a team that covers every core leaves
nothing for the thread that submits the ~140 blocking dispatches an image.

Every one of those is bit-identical — `tools/verify_vision_bfp16.cpp` gates the
conversion kernels by `memcmp`, and N14/N15/N16 print the same numbers before
and after. The seam's fp32 interface did **not** have to be renegotiated.

The tower is now device bound. See `REPLACEMENTWORK.md` §N-VISION-HOST.

The move is not free: `image_embeds` goes from relL2 0.0098 (host) to 0.0286
(attention) to 0.0401 (both) against the HF reference. See `npu/README.md`.
