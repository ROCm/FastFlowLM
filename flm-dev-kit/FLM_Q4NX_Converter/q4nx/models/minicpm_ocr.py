# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from ..constants import ModelArch
from ..gguf_tensor import GGUFTensor
from .minicpm_v import MiniCPM_V_1B


class MiniCPM_OCR_1B(MiniCPM_V_1B, model_arch=ModelArch.MINICPM_OCR_1B):
    """MiniCPM-OCR-1.0 language tower, and with `-t vision` its SigLIP tower.

    Upstream, `MiniCPMOcrForConditionalGeneration` is `MiniCPMV4_6ForConditionalGeneration`
    with `pass`, and MiniCPM-V-4.7 is 4.6 plus video and 3D-RoPE machinery this model never
    uses.  So the maths is MiniCPM-V-4.7's and `MiniCPM_V_1B` does the work unchanged:
    the vision q/k/v fuse, the ssm_{alpha,beta} Q8_0 + bf16-twin handling, the Qwen3.8-27B
    vision padding, and the refusal to extract a tokenizer from the GGUF.

    Two things differ, and only one of them needs code:

    * **Tied embeddings.**  `tie_word_embeddings: true`, so the checkpoint holds a single
      copy of that matrix and the GGUF has `token_embd.weight` but no `output.weight` --
      which is what `configs/minicpm_ocr_1b.json` maps to `lm_head.weight`.  Without the
      synthesis below the name map simply finds nothing and `model.q4nx` ships with no
      lm_head at all: a load-time failure at best, silently wrong logits at worst.
    * **`downsample_mode: "4x"`** skips the vit_merger, so the checkpoint carries none of
      its tensors.  That needs no code, only the absence of the `visual_vit_merger_*`
      entries from the config -- see `configs/minicpm_ocr_1b.json`.

    `vocab_size` is 248100 against MiniCPM-V-4.7's 248094.  Neither is a multiple of
    `row_block_size`, so the inherited `_pad_rows` zero-pads the head to 248128 rows; both
    pad to the same 248832 inside the engine, which is why the lm_head xclbin is reused.
    """

    def _read_gguf_tensors(self):
        # Runs before _load_config, so a tensor added here is seen by _create_name_maps and
        # mapped like any other -- the same hook MiniCPM_V_1B uses for its own synthesis.
        super()._read_gguf_tensors()
        if self.is_mmproj:
            return
        self._synthesize_tied_lm_head()

    def _synthesize_tied_lm_head(self):
        """Give the name map an `output.weight` by copying the tied `token_embd.weight`.

        Tying means the two *are* the same matrix, so this is a copy rather than a
        derivation.  `token_embd.weight` already ships Q8_0, which is exactly what the
        config asks `lm_head.weight` to be, so no re-quantization is involved and the
        inherited `_pad_rows` assertion on Q8_0 holds.
        """
        if "output.weight" in self.gguf_tensors:
            print("[INFO] GGUF already has output.weight; embeddings are not tied after all")
            return

        embd = self.gguf_tensors["token_embd.weight"]
        # Copied, not aliased: _pad_rows returns the tensor unchanged when no padding is
        # needed, and an alias would make the two q4nx tensors share one array.
        self.gguf_tensors["output.weight"] = GGUFTensor(
            name="output.weight",
            shape=embd.shape,
            data=embd.data.copy(),
            tensor_type=embd.tensor_type,
        )
        print(f"[INFO] Tied embeddings: synthesized output.weight from token_embd.weight "
              f"{tuple(embd.shape)} {embd.tensor_type.name}")
