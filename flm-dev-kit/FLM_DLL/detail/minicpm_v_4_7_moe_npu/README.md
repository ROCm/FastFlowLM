<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# minicpm_v_4_7_moe_npu -- MiniCPM-V-4.7-35B-A3B

`openbmb/MiniCPM-V-4.7-35B-A3B`: a Qwen3.5-MoE language model (hidden 2048, 40 layers, 3 gated-delta-net : 1 full
attention, 256 experts top-8 + a shared expert) behind the same SigLIP vision tower as MiniCPM-V-4.7-1B. The engine is
exactly those two halves joined, and nothing else:

| half | taken from | what it is here |
| --- | --- | --- |
| language: decode, prefill, MoE, prefix-cache states | `qwen3_6_moe_npu` (Qwen3.6-35B-A3B, same dims) | renamed (`minicpm_v_4_7_moe_*`, `mcvmoe_*`) so it can sit next to that engine in one process |
| vision tower, slicing payload, canvas m-rope | `minicpm_v_4_7_npu` (the 1B) | `image/` is a copy with the namespaces renamed `mcv` -> `mcvm`; the projector output is 2048 (read from `vision_config`) |

No kernel is new. The `minicpm-v-4.7-moe:35b-a3b` entry in `FLM_Xclbin/kernel_assignment.md` lists the Qwen3.6 text kernels and the
Qwen3.8 vision kernels (`vision_attn`, `vision_mm_a`, `vision_mm_b`).

## What differs from `qwen3_6_moe_npu`

* No Qwen vision (`qwen3_6_moe_image.*`, `rot_pos_emb.*`, `vision_common`): `_load_vision()` builds
  `mcvm::vision_tower` on the three vision xclbins, `_encode_images()` runs it, and the rows overwrite the
  `<|image_pad|>` (248056) rows.
* Positions: `mcvm::canvas_mrope()` (the checkpoint's own rule), plain text positions when there is no image.
  `set_context_length(L, next_position)` takes the position it returns.
* vocab 248144 (Qwen3.6: 248320): both pad to 248832, so the `lm_head` kernel is the same. The q4nx `lm_head` has 248160
  rows (the converter zero-pads to a multiple of 32).
* Video is rejected (`<|video_pad|>`), as in the 1B.

## Weights

`FLM_Q4NX_Converter`: language from the llama.cpp GGUF (arch `qwen35moe`), vision from the mmproj (arch `clip`), plus the BF16
checkpoint for two tensors:

```
python convert.py -i MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf -o OUT -f minicpm-v-35b-a3b \
       -e model-00001-of-00016.safetensors --head-bf16 model-00015-of-00016.safetensors     # model.q4nx
python convert.py -i mmproj-model-f16.gguf -o OUT -f minicpm-v-35b-a3b -t vision             # vision_weight.q4nx
python package_minicpm_v_35b.py --hf <upstream folder> --out OUT                            # config + tokenizer files
```

Three things the llama.cpp Q4_K_M needs that the Qwen3.6 package did not:

* **`ssm_alpha` / `ssm_beta` are Q4_K in this GGUF; the engine reads them as plain bf16 `[D, heads]`** -> dequantized.
  Packed as Q4_K they decode to garbage (repeated `_`), with no error.
* Q6_K tensors (14 `attn_qkv`, 6 `attn_v`, 20 `ffn_down_exps`, 20 `ffn_down_shexp`) -> `RequantToQ4K`: every matrix is
  `flm_q4k` in this engine and the stock converter's fallback for Q6_K is Q4_1, which the engine does not read. The refit is
  ggml's scale search per group of 32 (about 19% lower MSE than min/max on these tensors).
* The token embedding (Q4_K in the GGUF) and the output head (Q6_K) are taken from the BF16 checkpoint: q4nx stores the first as bf16
  and the second as Q8_0 anyway, so this costs no size and drops the GGUF's loss. The 248144-row head is zero-padded to a multiple of 32.

Chat template: the upstream one (thinking only when `enable_thinking` is true). The GGUF repo's `thinking/` files are the
same weights with that template restored; the root files carry a template that has thinking switched off.

See `FLM_DLL/ADDING_A_NEW_MODEL.md` for the engine contract and `image/README.md` for the tower.
