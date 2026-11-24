# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from ..constants import ModelArch
from ..gguf_tensor import GGUFTensor
from .qwen35 import Qwen35
from .qwen38 import Qwen38_27B
from gguf import dequantize, quantize, GGMLQuantizationType
import hashlib
import numpy as np
import os
import re
import torch


class MiniCPM_V_1B(Qwen35, model_arch=ModelArch.MINICPM_V_1B):
    """MiniCPM-V-4.7-1B language tower, and with `-t vision` its SigLIP tower.

    The llama.cpp GGUF declares general.architecture = qwen35 and its text tower is
    config-identical to Qwen3.5-0.8B, so Qwen35.convert() does the work unchanged.
    What differs is the GGUF itself, which is a stock llama.cpp Q4_K_M rather than
    the FLM-produced Qwen3.5 GGUFs that path was written against.  Rather than
    teach Qwen35 about that, this class rewrites three things in self.gguf_tensors
    so the tensor set looks like one of those before Qwen35 sees it:

    * ssm_{alpha,beta}.weight arrive Q4_K, but the decode layer reads them as Q8_0
      (AB_PROJ_QUANT_TYPE in the Qwen3_5 decoding kernel).  get_used_quantization_type()
      passes Q4_K straight through, so the config's Q8_0 never takes effect; re-quantize
      them here instead.  Q4_K values sit on a 16-level grid per 32-group, so Q8_0's
      127 levels hold them with negligible loss.
    * ssm_{alpha,beta}.bf16.weight do not exist in this GGUF.  The engine reads them
      for the host-side prefill, so synthesize them from the *original* Q4_K tensor
      (not the Q8_0 copy) -- the same [16, 1024] BF16 Qwen3.5-0.8B-NPU2 ships.
    * output.weight has 248094 rows, which is not a multiple of row_block_size; the
      Q8_0 packer needs whole row blocks.  Zero-pad it.  The engine pads lm_head to
      vocab_size_padded anyway, and ids >= vocab_size are never valid tokens.

    The vision tower (mmproj-model-f16.gguf, general.architecture = clip) is
    dimensionally Qwen3.8-27B's -- 27 blocks, D 1152, 16 x 72 heads, mlp 4304 --
    and runs on the same FLM_Xclbin/Qwen3_8/vision_mm bitstreams, so it is
    converted the way Qwen38_27B._convert_vision does it: bf16 throughout, no
    (mm_n mm_k) re-order, zero-pad N and K up to the array tile.  The padding
    helpers are Qwen38_27B's own.  Three things differ:

    * q / k / v ship as three [1152, 1152] tensors; Qwen3.8 ships one fused
      attn_qkv [3456, 1152].  They are concatenated here, rows [q; k; v] and the
      biases likewise, before the name map is built -- the same split
      image_cpu.cpp makes -- for the 27 blocks and the layer-6 vit_merger.
    * the patch embedding is one conv2d kernel, not two temporal halves, so
      there is nothing to merge.  It keeps the checkpoint's (1152, 3, 14, 14).
    * vit_merger (after block 6) and the projector's 4608-wide pre_norm have no
      Qwen3.8 counterpart; they get names of their own (configs/minicpm_v_1b.json).
    """
    AB_PROJ_NAMES = ("ssm_alpha", "ssm_beta")

    #: The attn_q.weight of each attention whose q/k/v are fused; group 1 is its prefix.
    VISION_Q_WEIGHT_RE = re.compile(r"^(v\.blk\.\d+|v\.vit_merger)\.attn_q\.weight$")

    #: As Qwen3.8's.  The vit_merger tensors are named so that they match these
    #: suffixes too (vit_merger.attn.qkv, vit_merger.mlp.linear_fc1, ...).
    VISION_MM_LAYERS = Qwen38_27B.VISION_MM_LAYERS
    _vision_mm_layer = Qwen38_27B._vision_mm_layer
    _vision_n_align = Qwen38_27B._vision_n_align
    _pad_vision_mm = Qwen38_27B._pad_vision_mm

    def __init__(self, gguf_reader):
        print("[INFO] Using MiniCPM_V_1B converter")
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        # Reached through general.name like the language GGUF; only the vision
        # path may run on it.
        self.is_mmproj = "clip.has_vision_encoder" in self.gguf_reader.fields
        if self.is_mmproj:
            print("[INFO] mmproj GGUF: vision tower only")
        super().initialize()

    def _read_gguf_tensors(self):
        # Runs before _load_config, so the synthesized twins are seen by
        # _create_name_maps and mapped like any other GGUF tensor.
        super()._read_gguf_tensors()

        if self.is_mmproj:
            self._fuse_vision_qkv()
            return

        for key in list(self.gguf_tensors.keys()):
            for proj in self.AB_PROJ_NAMES:
                if key.endswith(f".{proj}.weight"):
                    self._requantize_ab_proj(key, proj)

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        if weights_type not in ("language", "vision"):
            raise ValueError(f"Unsupported weights_type: {weights_type} for MiniCPM-V-4.7-1B")
        if (weights_type == "vision") != self.is_mmproj:
            raise ValueError(
                f"-t {weights_type} was given a "
                f"{'vision mmproj' if self.is_mmproj else 'language'} GGUF; the two "
                f"towers ship in separate files")
        if weights_type == "vision":
            self.q4nx_tensors = {}
            return self._convert_vision(q4nx_path)

        if "output.weight" in self.gguf_tensors:
            self.gguf_tensors["output.weight"] = self._pad_rows(self.gguf_tensors["output.weight"])
        super().convert(q4nx_path, weights_type)

    # ------------------------------------------------------------------ #
    # vision tower
    # ------------------------------------------------------------------ #
    def _fuse_vision_qkv(self):
        """attn_{q,k,v}.{weight,bias} -> attn_qkv.{weight,bias}, rows [q; k; v].

        Done on the raw GGUF arrays (F16 weights, F32 biases), so it is exact;
        the bf16 cast happens later, per element, exactly as it would have.
        """
        for key in sorted(self.gguf_tensors):
            m = self.VISION_Q_WEIGHT_RE.match(key)
            if not m:
                continue
            prefix = m.group(1)
            for kind in ("weight", "bias"):
                parts = [self.gguf_tensors.pop(f"{prefix}.attn_{p}.{kind}") for p in "qkv"]
                types = {t.tensor_type for t in parts}
                shapes = {t.shape for t in parts}
                assert len(types) == 1 and len(shapes) == 1, (
                    f"{prefix}.attn_{{q,k,v}}.{kind}: mismatched {types} {shapes}")
                # GGUF shape is (cols, rows); the numpy data is (rows, cols).
                shape = parts[0].shape[:-1] + (3 * parts[0].shape[-1],)
                name = f"{prefix}.attn_qkv.{kind}"
                self.gguf_tensors[name] = GGUFTensor(
                    name=name, shape=shape,
                    data=np.concatenate([t.data for t in parts], axis=0),
                    tensor_type=parts[0].tensor_type)
            print(f"[INFO] Fused {prefix}.attn_{{q,k,v}} -> {prefix}.attn_qkv")

    vision_source_policy = "error"
    vision_hf_verifier = "verify_minicpm_v_vision_q4nx.py"

    def _convert_vision_source_check(self):
        # The 1B tower is verified against the checkpoint: every F32 tensor is bf16-exact, so a
        # source that is not is refused as well, and the output records where it came from.
        self._check_vision_source_is_bf16(strict_f32=True, record_provenance=True)

    def _convert_vision(self, q4nx_path: str):
        if self.vision_MM_K is None or self.vision_MM_N is None:
            raise ValueError("configs/minicpm_v_1b.json has no vision_config; "
                             "cannot size the vision MM padding")
        self._convert_vision_source_check()

        for gguf_tensor in self.gguf_tensors.values():
            name = gguf_tensor.name
            if name not in self.forward_name_map:
                print(f"[WARN] No name mapping for {name}, skipping")
                continue
            new_name = self.forward_name_map[name]

            unpacked = gguf_tensor.unpack(GGMLQuantizationType.BF16)
            assert len(unpacked) == 1, f"{name} did not unpack to a dense tensor"
            w = unpacked[0].to(torch.bfloat16)
            print(f"Processing tensor: {name} {tuple(w.shape)} -> {new_name}")

            layer = self._vision_mm_layer(new_name)
            if layer is not None:
                w = self._pad_vision_mm(w, layer, is_bias=new_name.endswith(".bias"))

            self.q4nx_tensors[new_name] = w.contiguous()

        # vision_weight.q4nx, next to the language model.q4nx.
        self._export_q4nx_tensors(q4nx_path, filename="vision_weight.q4nx")

    def _extract_tokenizer_json(self, output_folder: str) -> dict:
        # The GGUF-rebuilt tokenizer.json is wrong for this model: no NFC normalizer,
        # a generic ByteLevel pre-tokenizer instead of Qwen's split regex, and 4 of
        # the 50 added tokens (no <|image_pad|>).  The package ships the HF
        # tokenizer.json instead, and the gate re-runs convert.py into that folder,
        # so writing one here would silently clobber it.
        print("[INFO] Not extracting tokenizer.json from the GGUF; package the HF "
              "tokenizer.json from MiniCPM-V-4.7-1B-hf instead")
        return {}

    def _requantize_ab_proj(self, key: str, proj: str):
        src = self.gguf_tensors[key]
        w = dequantize(src.data, src.tensor_type)  # float32 [rows, cols], exact

        twin = key.replace(f".{proj}.weight", f".{proj}.bf16.weight")
        if twin not in self.gguf_tensors:
            w_bf16 = torch.from_numpy(w).contiguous().to(torch.bfloat16)
            self.gguf_tensors[twin] = GGUFTensor(
                name=twin, shape=src.shape,
                data=w_bf16.view(torch.uint8).numpy(),  # GGUFReader's BF16 layout
                tensor_type=GGMLQuantizationType.BF16)

        if src.tensor_type != GGMLQuantizationType.Q8_0:
            self.gguf_tensors[key] = GGUFTensor(
                name=key, shape=src.shape,
                data=quantize(w, GGMLQuantizationType.Q8_0),
                tensor_type=GGMLQuantizationType.Q8_0)

    def _pad_rows(self, t: GGUFTensor) -> GGUFTensor:
        rows = t.data.shape[0]
        pad = -rows % self.row_block_size
        if pad == 0:
            return t
        assert t.tensor_type == GGMLQuantizationType.Q8_0, \
            f"{t.name}: row padding only implemented for Q8_0, got {t.tensor_type.name}"
        print(f"[INFO] Zero-padding {t.name} from {rows} to {rows + pad} rows")
        # An all-zero Q8_0 block is scale 0, quants 0: a row that dequantizes to 0.
        data = np.concatenate([t.data, np.zeros((pad, t.data.shape[1]), dtype=t.data.dtype)])
        return GGUFTensor(name=t.name, shape=(t.shape[0], rows + pad), data=data,
                          tensor_type=t.tensor_type)
