<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Qwen3-ASR / Confucius4-R2T2 audio encoder

`qwen3_asr_audio_encoder` computes what the HF `Qwen3ASRAudioEncoder` computes, with the window mask applied. Given a
`qwen3_asr_audio_gemm` backend (`qwen3_asr_audio_gemm_npu`: `audio_gemm.xclbin`, built from `FLM_Xclbin/bfp16_gemm`), its matrix
products run there; without one (`nullptr`) everything runs on the CPU (AVX-512 + OpenMP), which needs no NPU/XRT and is the
reference for the NPU path.

```
mel [frames][128] fp32
 -> per 100-frame chunk (tail chunk zero-padded to 100, like the reference; mel cast to bf16 first)
      conv2d1 (1->480) + GELU, conv2d2 (480->480) + GELU, conv2d3 (480->480) + GELU    3x3, stride 2, pad 1  => 13 tokens / chunk
      token t = vector over (c, f): index c*16 + f  (7680)     conv_out Linear 7680->1024 (no bias)
      + sinusoid position table, positions 0..12 restart in EVERY chunk
 -> 24 x pre-LN layer: LN, q/k/v (1024, bias), window attention (16 heads x 64), out_proj, residual,
                       final_layer_norm, fc1 1024->4096 + GELU, fc2 4096->1024, residual
    attention is confined to windows of 104 tokens (= n_window_infer / (2 n_window) = 8 chunks x 13); the last window is the remainder
 -> ln_post, proj1 (1024->1024) + GELU, proj2 (1024->2048)  => [n_tokens][2048] bf16
n_tokens = 13 * (frames / 100) + tail,  tail = ((((r-1)/2+1)-1)/2+1 -1)/2+1 for r = frames % 100 (0 if r == 0)   (qwen3_asr_audio_tokens)
```

Numerics follow the BF16 reference: every op output is rounded to bf16 (RNE); accumulation, LayerNorm statistics, softmax
and the erf-GELU are fp32. Activations are fp32 values that are exactly bf16-representable. Attention scores and probabilities
are **not** rounded (the sdpa/flash behaviour); `R2T2_ROUND_SCORES=1` / `R2T2_ROUND_PROBS=1` switch to the eager behaviour and
were measured worse on every test clip.

Reference: `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py` + `run/r2t2_patch.py`. The stock HF code passes
`attention_mask=None` to the layers (full attention); the patch builds the window mask from `cu_seqlens` like the vLLM backend.
Things the reference does that are easy to miss: the tail chunk is padded to 100 frames and masked *after* the convs (so conv2/conv3
see `GELU(bias)` rows, not zeros, beyond the valid frames: the C++ pads too); the position table restarts per chunk; the mel is cast to bf16
before the first conv.

## Weights (`audio_weight.q4nx`, safetensors, all BF16, 397 tensors)

Names: `model.audio.conv2d{1,2,3}.{weight,bias}`, `model.audio.conv_out.weight`, `model.audio.layers.N.*`, `model.audio.ln_post.*`,
`model.audio.tower.proj{1,2}.*`. The 147 matrices are stored in the audio MM tile layout written by
`FLM_Q4NX_Converter` (`_multi_modal_mm_weight_rearrange`, K tile 512, N tile 64): zero-pad N and K up to multiples of 512, then
`(N mm_n) (K mm_k) -> N K (mm_n mm_k)`; element (n, k) lives at `((n/64)*(Kp/512) + k/512)*32768 + (n%64)*512 + k%512`.
The loader inverts this once (`unpermute_matrix`) into row-major `[out][in]` fp32; conv weights `[480][cin][3][3]` are reordered to
`[480][(kh,kw,cin)]` so an im2col row is a run of contiguous copies. The sinusoid table is computed at load (it is not in the file).

## Test

```
cd FLM_DLL/detail/qwen3_asr_npu/audio/test && make run          # builds with g++-13, runs the default clip set
./audio_encoder_test <audio_weight.q4nx> <goldens dir> clip...  # goldens/<clip>/mel.npy (fp32 [1,128,T]) + enc.npy (reference output)
DUMP_DIR=... / STAGE_DIR=...                                    # optional raw dumps (final output / per-stage activations)
```

Gates: token count equal, global cosine > 0.999, relL2 < 0.05, min per-token cosine > 0.80 (informational, see below).

### Measured (24 threads, 2026-11-02, goldens = patched BF16 CPU reference)

| clip | sec | tokens | cos | min token cos | max abs | relL2 |
|---|---|---|---|---|---|---|
| syn_short1s | 1.0 | 13 | 0.99994 | 0.9999 | 0.0011 | 0.0111 |
| syn_silence3s | 3.0 | 39 | 0.99995 | 0.9999 | 0.0015 | 0.0102 |
| repo_test | 6.7 | 88 | 0.99975 | 0.9873 | 0.0086 | 0.0225 |
| ls_4077-13751-0006 | 13.1 | 170 | 0.99985 | 0.9952 | 0.0096 | 0.0173 |
| zh_0006 | 15.2 | 198 | 0.99965 | 0.9812 | 0.0192 | 0.0265 |
| zh_0015 | 18.4 | 240 | 0.99987 | 0.9975 | 0.0060 | 0.0160 |
| syn_quiet_onset | 7.0 | 91 | 0.99956 | 0.9778 | 0.0205 | 0.0296 |
| syn_long_concat | 68.3 | 889 | 0.99993 | 0.9985 | 0.0035 | 0.0120 |

All 43 golden clips: token counts equal; global cosine 0.9981 .. 0.99995; relL2 0.0098 .. 0.0625 (worst zh_0048).
Speed: 21-35 ms per second of audio when the box is otherwise idle (a 68 s clip in about 1.5-2 s); 1 s clips ~40 ms/s (fixed per-call cost).

**How big is BF16 noise here?** Run the same patched encoder in fp32 and compare with the BF16 reference (`run/enc_noise_floor.py`):
the reference differs from fp32 by relL2 0.010 .. 0.07 (min token cos down to 0.87 on quiet audio). The C++ output differs from the reference by
0.010 .. 0.03 on the same clips, i.e. at the BF16 noise level, and is *closer to fp32 than the reference is* on 3 of 5 clips tested.

**Per-stage check** (`run/enc_stages.py`, forward hooks): up to layer 16 the C++ activations match the reference to relL2 0.05% (conv_out+pos), 0.15% (L1)
growing smoothly to 0.8% (L16), no systematic offset. Beyond that, on some clips one token diverges (e.g. zh_0024 token 23, zh_0048 token 113) at the
"massive activation" channels (columns 351, 365, 730; magnitudes up to ~7): BF16 rounding is coarse there and the divergence is chaotic, not a
bug (on zh_0048 the reference itself differs from fp32 with min cosine 0.95). That one token is what the min-token-cosine column shows.

**Task level** (`run/enc_e2e.py`): C++ encoder output fed to the reference decoder: 42 of 43 golden clips give a byte-identical transcript; the one
difference is only the language label (`English` vs `None`) on a Chinese clip (identical text).

## The NPU path

- Products on the NPU (`qwen3_asr_audio_gemm_npu`): conv2/conv3 as im2col GEMMs (K = 4320, N = 480; the packer gathers the im2col rows
  itself), conv_out (K = 7680), q/k/v fused (N = 3072), out_proj, fc1, fc2, proj1, proj2. All chunks of a call share one conv product per
  conv (8 chunks per group). The weights are packed once at load (bfp16, 9 B per 8 values) into device buffers; bias is added on
  the cores, GELU on the host once the parts below are summed.
- Precision: every product keeps A and W at bf16 precision. A = A_hi + A_lo / 256 with A_hi its bfp16 value: the rest goes as more rows
  of the same call (the weights are read once for both); W likewise as more output columns. The host adds the parts and rounds to bf16.
  Plain bfp16 products drift the conv stack on quiet input (syn_quiet_onset fails); NPU vs fp64 relL2 is ~0.0023 with both splits
  (bf16 floor 0.0017, plain bfp16 0.0092).
- Host: conv1 (K = 9), LayerNorm (statistics in double: fp32 sums fail syn_quiet_onset), window attention (16 heads x 64, <= 104 tokens,
  ~1 % of the FLOPs), GELU, position add, residuals.
- Streaming: a window of 104 tokens never changes once complete (same mel), so completed windows can be cached; the mel global-max clamp is the only
  thing that changes earlier frames (`conv_cache`; `test/conv_cache_test`).

`make run_npu XCLBIN=<audio_gemm.xclbin>` runs the gate above with the NPU path; all 8 default clips pass (cos >= 0.99963, relL2 <= 0.028).
