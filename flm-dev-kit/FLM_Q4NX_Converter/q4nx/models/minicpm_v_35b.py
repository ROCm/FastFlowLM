# Copyright (c) 2023-2026 The ggml authors
# SPDX-License-Identifier: MIT
# Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

from ..constants import ModelArch
from ..gguf_tensor import GGUFTensor
from .qwen35_moe import QWEN35_A3B
from .minicpm_v import MiniCPM_V_1B
from gguf import dequantize, quantize, GGMLQuantizationType
import numpy as np
import torch


class RequantToQ4K(GGUFTensor):
    """A Q5_K / Q6_K tensor re-fitted to the Q4_K form the engine reads (every matrix is flm_q4k).

    llama.cpp's Q4_K_M keeps some tensors (half of attn_v / ffn_down, a few attn_qkv) at Q6_K; the engine has
    no Q6_K kernel, and the converter's own fallback for such tensors would be Q4_1, which the engine does
    not read.  So: dequantize, then fit 4-bit values per group of 32 the way ggml's Q4_K does (a scale search with
    the min forced <= 0, w = t q - u with u >= 0; see _fit_groups) and hand (t, u, q) to _pack_q4k, which
    does the (BF16 S', uint8) super-block re-fit.  Lossy (6 -> 4 bit) exactly where llama.cpp wanted more bits.
    """

    GROUP = 32

    def __init__(self, src: GGUFTensor):
        super().__init__(src.name, src.shape, src.data, src.tensor_type)

    def get_used_quantization_type(self, default_tensor_type):
        return GGMLQuantizationType.Q4_K

    @staticmethod
    def _fit_groups(g: np.ndarray, nstep: int = 20, rdelta: float = 0.1, rmin: float = -1.0):
        """4-bit (scale, min) per group of 32, ggml's make_qkx2_quants: try `nstep` scales around 15 / (max - min), solve the
        weighted least squares for (scale, min) at each, keep the best (weights |x| + rms, as ggml uses without an imatrix).
        About 19% lower reconstruction MSE than plain min/max on this model's Q6_K tensors.  g: (groups, 32) float32.
        Returns t (scale), u = -min >= 0 (w = t q - u) and q in 0..15."""
        wgt = np.abs(g) + np.sqrt((g * g).mean(-1, keepdims=True))
        lo = np.minimum(g.min(-1), 0.0)
        rng = g.max(-1) - lo
        best_err = np.full(lo.shape, np.inf, dtype=np.float32)
        best_s = np.zeros_like(lo)
        best_m = lo.copy()
        sw = wgt.sum(-1)
        sx = (wgt * g).sum(-1)
        for i in range(nstep + 1):
            iscale = (rmin + rdelta * i + 15) / np.where(rng > 0, rng, 1)
            L = np.clip(np.rint(iscale[..., None] * (g - lo[..., None])), 0, 15)
            sl = (wgt * L).sum(-1)
            sl2 = (wgt * L * L).sum(-1)
            sxl = (wgt * L * g).sum(-1)
            D = sw * sl2 - sl * sl
            ok = D > 0
            Dsafe = np.where(ok, D, 1)
            s = np.where(ok, (sw * sxl - sx * sl) / Dsafe, 0)
            m = np.where(ok, (sl2 * sx - sl * sxl) / Dsafe, lo)
            pos = m > 0
            m = np.where(pos, 0, m)
            s = np.where(pos, sxl / np.where(sl2 > 0, sl2, 1), s)
            err = (wgt * (s[..., None] * L + m[..., None] - g) ** 2).sum(-1)
            take = (err < best_err) & (s > 0)
            best_err = np.where(take, err, best_err)
            best_s = np.where(take, s, best_s)
            best_m = np.where(take, m, best_m)
        inv = np.where(best_s > 0, 1.0 / np.where(best_s > 0, best_s, 1.0), 0.0)
        q = np.clip(np.rint((g - best_m[..., None]) * inv[..., None]), 0, 15)
        return best_s.astype(np.float32), (-best_m).astype(np.float32), q.astype(np.float32)

    def unpack(self, default_tensor_type):
        columns = self.shape[0]
        w = dequantize(self.data, self.tensor_type).astype(np.float32).reshape(-1, columns)
        rows = w.shape[0]
        groups = w.reshape(rows * (columns // self.GROUP), self.GROUP)
        t = np.empty(groups.shape[0], dtype=np.float32)
        u = np.empty(groups.shape[0], dtype=np.float32)
        q = np.empty(groups.shape, dtype=np.float32)
        chunk = 1 << 19  # groups per step: bounds the ~12 float32 temporaries of the search to ~3 GB
        for i in range(0, groups.shape[0], chunk):
            t[i:i + chunk], u[i:i + chunk], q[i:i + chunk] = self._fit_groups(groups[i:i + chunk])
        n_groups = columns // self.GROUP
        return (torch.from_numpy(t.reshape(rows, n_groups)).contiguous(),
                torch.from_numpy(u.reshape(rows, n_groups)).contiguous(),
                torch.from_numpy(q.reshape(rows, columns)).contiguous())


class MiniCPM_V_35B_A3B(QWEN35_A3B, model_arch=ModelArch.MINICPM_V_35B_A3B):
    """MiniCPM-V-4.7-35B-A3B: the Qwen3.6-35B-A3B language tower plus MiniCPM's SigLIP tower.

    Language (`-t language`, the llama.cpp GGUF, general.architecture = qwen35moe): QWEN35_A3B.convert()
    does the work.  What differs from Qwen3.6-35B-A3B is the vocabulary, 248144 rows, which is not a
    multiple of row_block_size (32): the Q8_0 lm_head packer needs whole row blocks, so output.weight is
    zero-padded, as MiniCPM_V_1B does.  The GGUF's output.weight is Q6_K here, so it is dequantized and
    re-quantized to Q8_0 first (the engine reads the lm_head as Q8_0).  ids >= vocab_size are never valid.

    Vision (`-t vision`, mmproj-model-f16.gguf, general.architecture = clip): the tower is the 1B's
    exactly (27 blocks, D 1152, 16 x 72 heads, vit_merger after block 6), only the projector's output is
    2048 instead of 1024.  The conversion is MiniCPM_V_1B's, reused method for method.
    """

    VISION_Q_WEIGHT_RE = MiniCPM_V_1B.VISION_Q_WEIGHT_RE
    VISION_MM_LAYERS = MiniCPM_V_1B.VISION_MM_LAYERS
    _vision_mm_layer = MiniCPM_V_1B._vision_mm_layer
    _vision_n_align = MiniCPM_V_1B._vision_n_align
    _pad_vision_mm = MiniCPM_V_1B._pad_vision_mm
    _fuse_vision_qkv = MiniCPM_V_1B._fuse_vision_qkv
    _convert_vision = MiniCPM_V_1B._convert_vision
    vision_source_policy = "error"
    _convert_vision_source_check = MiniCPM_V_1B._convert_vision_source_check
    _extract_tokenizer_json = MiniCPM_V_1B._extract_tokenizer_json

    def __init__(self, gguf_reader):
        print("[INFO] Using MiniCPM_V_35B_A3B converter")
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        self.is_mmproj = "clip.has_vision_encoder" in self.gguf_reader.fields
        if self.is_mmproj:
            print("[INFO] mmproj GGUF: vision tower only")
        super().initialize()

    def _read_gguf_tensors(self):
        super()._read_gguf_tensors()
        if self.is_mmproj:
            self._fuse_vision_qkv()
            return
        # output.weight is handled in convert() (Q8_0 lm_head); the token embedding is dequantized to bf16 as is
        for key, tensor in list(self.gguf_tensors.items()):
            if key in ("output.weight", "token_embd.weight"):
                continue
            if key.endswith((".ssm_alpha.weight", ".ssm_beta.weight")):
                # the engine reads the two gate projections as plain bf16 [D, heads] (the Qwen3.6 package ships them
                # so); this GGUF has them Q4_K, so dequantize here and let the converter's bf16 path transpose them
                w = torch.from_numpy(dequantize(tensor.data, tensor.tensor_type)).contiguous().to(torch.bfloat16)
                self.gguf_tensors[key] = GGUFTensor(name=key, shape=tensor.shape, data=w.view(torch.uint8).numpy(),
                                                    tensor_type=GGMLQuantizationType.BF16)
                continue
            if tensor.tensor_type in (GGMLQuantizationType.Q5_K, GGMLQuantizationType.Q6_K):
                print(f"[INFO] {key}: {tensor.tensor_type.name} -> Q4_K (re-fit)")
                self.gguf_tensors[key] = RequantToQ4K(tensor)

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        if weights_type not in ("language", "vision"):
            raise ValueError(f"Unsupported weights_type: {weights_type} for MiniCPM-V-4.7-35B-A3B")
        if (weights_type == "vision") != self.is_mmproj:
            raise ValueError(
                f"-t {weights_type} was given a "
                f"{'vision mmproj' if self.is_mmproj else 'language'} GGUF; the two "
                f"towers ship in separate files")
        if weights_type == "vision":
            self.q4nx_tensors = {}
            return self._convert_vision(q4nx_path)

        # The GGUF keeps the token embedding at Q4_K and the output head at Q6_K, but q4nx stores the embedding as bf16 and the
        # head as Q8_0 anyway: take both from the BF16 checkpoint when it is given (no size cost, no GGUF quantization loss).
        vocab, hidden = self.gguf_tensors["token_embd.weight"].shape[1], self.gguf_tensors["token_embd.weight"].shape[0]
        emb = self._load_bf16_embedding(vocab, hidden)
        if emb is not None:
            self.gguf_tensors["token_embd.weight"] = GGUFTensor(
                name="token_embd.weight", shape=(hidden, vocab), data=emb.contiguous().view(torch.uint8).numpy(),
                tensor_type=GGMLQuantizationType.BF16)
        head = self._load_bf16_head(vocab, hidden)
        if head is not None:
            self.gguf_tensors["output.weight"] = GGUFTensor(
                name="output.weight", shape=(hidden, vocab), data=head.contiguous().view(torch.uint8).numpy(),
                tensor_type=GGMLQuantizationType.BF16)
        if "output.weight" in self.gguf_tensors:
            self.gguf_tensors["output.weight"] = self._pad_lm_head(self.gguf_tensors["output.weight"])
        super().convert(q4nx_path, weights_type)

    def _load_bf16_head(self, vocab_size: int, hidden_size: int):
        """The output head from --head-bf16 (a safetensors shard with lm_head.weight), or None if unset."""
        path = getattr(self, "head_bf16_path", None)
        if not path:
            return None
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            w = f.get_tensor("lm_head.weight")
        if tuple(w.shape) != (vocab_size, hidden_size):
            raise ValueError(f"{path}:lm_head.weight has shape {tuple(w.shape)}, expected ({vocab_size}, {hidden_size})")
        print(f"[INFO] Sourcing output.weight from {path}:lm_head.weight instead of the GGUF")
        return w.to(torch.bfloat16)

    def _pad_lm_head(self, t: GGUFTensor) -> GGUFTensor:
        rows = t.shape[-1]  # GGUF shape is (cols, rows)
        pad = -rows % self.row_block_size
        if pad == 0:
            return t
        w = dequantize(t.data, t.tensor_type)  # float32 [rows, cols]
        assert w.shape[0] == rows, f"{t.name}: dequantized {w.shape}, expected {rows} rows"
        w = np.concatenate([w, np.zeros((pad, w.shape[1]), dtype=w.dtype)])
        print(f"[INFO] Zero-padding {t.name} ({t.tensor_type.name}) from {rows} to {rows + pad} rows, as Q8_0")
        return GGUFTensor(name=t.name, shape=(t.shape[0], rows + pad),
                          data=quantize(w, GGMLQuantizationType.Q8_0),
                          tensor_type=GGMLQuantizationType.Q8_0)
