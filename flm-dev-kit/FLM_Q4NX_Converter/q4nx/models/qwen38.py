# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Qwen3.8-27B -> q4nx.

The GGUF reports `general.architecture = qwen35`, and the layer structure is the
Qwen3.5 one (gated full attention every 4th block, gated delta-net elsewhere),
but three things stop `Qwen35` from being reused as-is:

  * the delta-net v side expands 3x over the qk side (48 v heads over 16 qk
    groups, 6144 = 48 * 128), not 2x.  Every `(q g p) -> (g q p)` reorder in
    qwen35.py hardcodes `q = 2`; here the ratio is read out of the GGUF.
  * because of that, `attn_qkv` / `ssm_conv1d` no longer split down the middle:
    the qk side is 2 * 16 * 128 = 4096 of 10240 rows, so the split point is
    computed rather than `chunk(2)`.
  * block 64 is the MTP (next-token-prediction) head.  The
    `layer_id % full_attention_interval == interval - 1` rule used by qwen35.py
    would call it recurrent; `qwen35.attention.recurrent_layers` says it is not.
    Its tensors are renamed to mirror the HF checkpoint's `mtp.*` keys so a
    reference dump lines up name for name.

Unlike qwen35.py the bf16 prefill copies of alpha/beta (`ssm_{alpha,beta}.bf16`)
are reordered too -- they are the same rows as the quantized copy, so leaving
them in GGUF order would put the prefill and decode paths on different head
orders.

`-t vision` converts the companion mmproj GGUF.  That file reports
`general.architecture = clip`, so none of the `qwen35.*` geometry fields exist
and `_read_geometry` is skipped for it; the arch is still recognised through
`general.basename`.  The vision tower is bf16 end to end and its weights need
*no* re-order: FLM_Xclbin/Qwen3_8/vision_mm streams w out of DDR as plain
row-major (N, K), unlike the qwen3.5 vision designs whose kernels want the
blocked `(mm_n mm_k)` form that `vision_mm_weight_rearrange` builds.  All this
converter does is zero-pad N and K up to the tile the array can cover.
"""

import re

from einops import rearrange
from gguf import GGUFReader, GGMLQuantizationType
import torch
import torch.nn.functional as F

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch


class Qwen38_27B(__Q4NX_Converter, model_arch=ModelArch.QWEN38_27B):
    #: GGUF metadata prefix; the 27B still ships under the qwen35 architecture name.
    ARCH = "qwen35"
    GGML_GROUP_SIZE = 32

    def __init__(self, gguf_reader: GGUFReader):
        print("[INFO] Using Qwen3.8-27B converter")
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()
        # The mmproj file carries none of the qwen35.* fields -- it is a `clip`
        # GGUF that only got here through general.basename.  Reading the text
        # geometry out of it would raise, so only the vision path runs.
        self.is_mmproj = "clip.has_vision_encoder" in self.gguf_reader.fields
        if self.is_mmproj:
            print("[INFO] mmproj GGUF: vision tower only")
        else:
            self._read_geometry()

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    def _field(self, suffix, default=None):
        field = self.gguf_reader.fields.get(f"{self.ARCH}.{suffix}")
        if field is None:
            if default is None:
                raise KeyError(f"GGUF is missing {self.ARCH}.{suffix}")
            return default
        return field.contents()

    def _read_geometry(self):
        # full attention
        self.attn_head_dim = self._field("attention.value_length")

        # delta-net: q/k live in `group_count` groups, v expands over them
        self.l_head_dim = self._field("ssm.state_size")
        self.l_qk_heads = self._field("ssm.group_count")
        self.l_v_heads = self._field("ssm.inner_size") // self.l_head_dim
        assert self.l_v_heads % self.l_qk_heads == 0, (
            f"{self.l_v_heads} v heads do not divide into {self.l_qk_heads} qk groups")
        self.l_expand = self.l_v_heads // self.l_qk_heads
        #: rows of attn_qkv / ssm_conv1d that belong to q and k (v is the rest)
        self.l_qk_rows = 2 * self.l_qk_heads * self.l_head_dim

        # which blocks are delta-net.  The boolean list is authoritative: the
        # trailing MTP block breaks the modulo rule.
        recurrent = self.gguf_reader.fields.get(f"{self.ARCH}.attention.recurrent_layers")
        if recurrent is not None:
            self.recurrent_layers = [bool(v) for v in recurrent.contents()]
        else:
            # Not every llama.cpp writes the list.  Fall back to the tensors (only a full-attention
            # block, MTP included, has attn_q.weight), not to "every Nth block": that rule calls the
            # trailing MTP block recurrent and it would be converted without the q/gate split.
            self.recurrent_layers = [
                not self._is_full_attention_layer(i) for i in range(self.num_layers)]

        # MTP head: the last `nextn_predict_layers` blocks.  Only one is supported
        # here -- a second would need its own name prefix.
        n_mtp = self._field("nextn_predict_layers", 0)
        assert n_mtp <= 1, f"{n_mtp} MTP blocks, only 0 or 1 handled"
        self.mtp_layer_id = self.num_layers - n_mtp if n_mtp else None

        print(f"[INFO] delta-net: {self.l_qk_heads} qk groups x {self.l_expand} = "
              f"{self.l_v_heads} v heads of {self.l_head_dim}, qk rows {self.l_qk_rows}")
        print(f"[INFO] {sum(self.recurrent_layers)} recurrent / "
              f"{len(self.recurrent_layers)} blocks, MTP block {self.mtp_layer_id}")

    # ------------------------------------------------------------------ #
    # reorders
    #
    # d / m / qw always share the row axis, so a row permutation is applied to
    # each of them unchanged.  Column permutations are not: d and m carry one
    # entry per 32-column group where qw carries one per column.
    # ------------------------------------------------------------------ #
    def _rows_v_to_group_major(self, parts, head_dim):
        """(expand groups head_dim) rows -> (groups expand head_dim)."""
        return [rearrange(t, "(q g p) c -> (g q p) c", q=self.l_expand, p=head_dim).contiguous()
                for t in parts]

    def _reorder_qkv(self, parts):
        """Leave q/k alone, put the v rows in group-major order."""
        out = []
        for t in parts:
            qk, v = t[:self.l_qk_rows], t[self.l_qk_rows:]
            v = rearrange(v, "(q g p) c -> (g q p) c",
                          q=self.l_expand, p=self.l_head_dim).contiguous()
            out.append(torch.cat([qk, v], dim=0).contiguous())
        return out

    def _reorder_out_proj_cols(self, d, m, qw):
        """Same permutation as the v rows, applied to ssm_out's input columns."""
        p_scale = self.l_head_dim // self.GGML_GROUP_SIZE
        d = rearrange(d, "r (q g p) -> r (g q p)", q=self.l_expand, p=p_scale).contiguous()
        m = rearrange(m, "r (q g p) -> r (g q p)", q=self.l_expand, p=p_scale).contiguous()
        qw = rearrange(qw, "r (q g p) -> r (g q p)",
                       q=self.l_expand, p=self.l_head_dim).contiguous()
        return [d, m, qw]

    def _reorder_per_v_head_rows(self, parts):
        """One row per v head (alpha / beta): (expand groups) -> (groups expand)."""
        return [rearrange(t, "(q g) c -> (g q) c", q=self.l_expand).contiguous()
                for t in parts]

    def _pad_rows_to_chunk(self, parts):
        """Zero-pad the row axis up to a whole number of q4nx chunks.

        alpha / beta have one row per v head, and 48 v heads is 1.5 chunks of 32.
        `_pack_q8nx` only pads columns, so the rows are padded here; the extra
        rows decode to zero, so the runtime just reads the first `l_v_heads`.
        """
        rows = parts[0].shape[0]
        pad = (-rows) % self.row_block_size
        if pad == 0:
            return parts
        print(f"[INFO] Padding {rows} rows -> {rows + pad} to fill a "
              f"{self.row_block_size}-row chunk")
        return [F.pad(t, (0, 0, 0, pad), "constant", 0).contiguous() for t in parts]

    def _reorder_per_v_head_vector(self, t):
        """One element per v head (ssm_a, ssm_dt)."""
        return rearrange(t, "(q g) -> (g q)", q=self.l_expand).contiguous()

    def _mtp_name(self, q4nx_name):
        """model.layers.<mtp>.X -> model.mtp.layers.0.X; everything else unchanged.

        The four `nextn.*` tensors are already mapped straight to `model.mtp.*`
        by the config, so they never reach this.
        """
        if self.mtp_layer_id is None:
            return q4nx_name
        prefix = f"model.layers.{self.mtp_layer_id}."
        if q4nx_name.startswith(prefix):
            return "model.mtp.layers.0." + q4nx_name[len(prefix):]
        return q4nx_name

    # ------------------------------------------------------------------ #
    # vision tower
    # ------------------------------------------------------------------ #
    #: the six MM weights the vision_mm array runs; everything else in the
    #: tower (norms, patch embed, position table) is untiled and passes through.
    VISION_MM_LAYERS = ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2",
                        "merger.linear_fc1", "merger.linear_fc2")

    def _vision_mm_layer(self, q4nx_name):
        """The entry of VISION_MM_LAYERS this tensor belongs to, or None."""
        for layer in self.VISION_MM_LAYERS:
            if q4nx_name.endswith(layer + ".weight") or q4nx_name.endswith(layer + ".bias"):
                return layer
        return None

    def _vision_n_align(self, layer):
        """Array N tile for one MM layer -- 384 on design A, 1024 on design B."""
        override = self.q4nx_config["vision_config"].get("vision_MM_N_override", {})
        return override.get(layer, self.vision_MM_N)

    def _pad_vision_mm(self, w, layer, is_bias):
        """Zero-pad a vision MM tensor up to the tile the array can cover.

        The kernel reads w as (N, K) row-major with the N bias elements in
        front, so the pad is plain trailing zeros on both axes: a padded output
        row gets a zero bias and zero weights, and the next layer's padded K
        rows multiply it by zero again.
        """
        n_align = self._vision_n_align(layer)
        if is_bias:
            n = w.shape[0]
            pad_n = (-n) % n_align
            if pad_n:
                print(f"[INFO]   bias {n} -> {n + pad_n}")
                w = F.pad(w, (0, pad_n), "constant", 0)
            return w.contiguous()

        n, k = w.shape
        pad_n = (-n) % n_align
        pad_k = (-k) % self.vision_MM_K
        if pad_n or pad_k:
            print(f"[INFO]   ({n}, {k}) -> ({n + pad_n}, {k + pad_k})")
            w = F.pad(w, (0, pad_k, 0, pad_n), "constant", 0)
        return w.contiguous()

    def _convert_vision(self, q4nx_path: str):
        self._check_vision_source_is_bf16()
        if self.vision_MM_K is None or self.vision_MM_N is None:
            raise ValueError("configs/qwen3.8_27b.json has no vision_config; "
                             "cannot size the vision MM padding")

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

        # The two half patch embeddings are the two temporal slices of one
        # conv3d kernel: (out, in, kh, kw) each -> (out, in, 2, kh, kw).
        a = self.q4nx_tensors.pop("model.visual.patch_embed.proj.weight")
        b = self.q4nx_tensors.pop("model.visual.patch_embed.proj.weight.1")
        self.q4nx_tensors["model.visual.patch_embed.proj.weight"] = \
            torch.stack([a, b], dim=2).contiguous()
        print(f"[INFO] Merged patch_embed halves {tuple(a.shape)} x2 -> "
              f"{tuple(self.q4nx_tensors['model.visual.patch_embed.proj.weight'].shape)}")

        # vision_weight.q4nx, not model.q4nx: the two towers ship side by side
        # in one folder, and the language weights are already there.
        self._export_q4nx_tensors(q4nx_path, filename="vision_weight.q4nx")

    # ------------------------------------------------------------------ #
    def convert(self, q4nx_path: str, weights_type: str = "language"):
        if weights_type not in ("language", "vision"):
            raise ValueError(f"Unsupported weights_type: {weights_type} for Qwen3.8-27B "
                             f"(this model has no audio tower)")
        if (weights_type == "vision") != self.is_mmproj:
            raise ValueError(
                f"-t {weights_type} was given a "
                f"{'vision mmproj' if self.is_mmproj else 'language'} GGUF; the two "
                f"towers ship in separate files")

        self.q4nx_tensors = {}
        if weights_type == "vision":
            return self._convert_vision(q4nx_path)

        if self.embed_bf16_path:
            print("[WARN] --embed-bf16 ignored: this config stores token_embd.weight as "
                  "Q8_0, not BF16")
        if not self._has_lm_head():
            raise ValueError("no output.weight / lm_head.weight in the GGUF; "
                             "this converter does not tie the embedding")

        for gguf_tensor in self.gguf_tensors.values():
            name = gguf_tensor.name
            if name not in self.forward_name_map:
                print(f"[WARN] No name mapping for {name}, skipping")
                continue

            requested_dtype = self.tensor_q4nx_type_map[name]
            target_dtype = gguf_tensor.get_used_quantization_type(requested_dtype)
            new_name = self._mtp_name(self.forward_name_map[name])
            print(f"Processing tensor: {name} with type {gguf_tensor.tensor_type.name} "
                  f"-> {new_name} with dtype {target_dtype.name}")

            match = re.match(r"^blk\.(\d+)\.", name)
            layer_id = int(match.group(1)) if match else None
            is_linear = layer_id is not None and self.recurrent_layers[layer_id]

            unpacked = list(gguf_tensor.unpack(requested_dtype))

            if not is_linear:
                # Gated full attention keeps q and the output gate in one tensor,
                # interleaved per head; the kernel wants them contiguous.
                if "self_attn.q_proj" in new_name:
                    print("[INFO] Seperate q, gate for q_proj")
                    unpacked = [rearrange(t, "(g p h) c -> (p g h) c",
                                          p=2, h=self.attn_head_dim).contiguous()
                                for t in unpacked]
            else:
                if "self_attn.gate_proj" in new_name:
                    print("[INFO] Reorder Gate")
                    unpacked = self._rows_v_to_group_major(unpacked, self.l_head_dim)

                elif "qkv_proj" in new_name:
                    print("[INFO] Reorder v rows of qkv_proj")
                    unpacked = self._reorder_qkv(unpacked)

                elif "ssm_out_proj" in new_name:
                    print(f"[INFO] Reorder for {new_name}")
                    unpacked = self._reorder_out_proj_cols(*unpacked)

                elif "ssm_alpha_proj" in new_name or "ssm_beta_proj" in new_name:
                    # both the quantized copy and the bf16 prefill copy
                    print(f"[INFO] Reorder for {new_name}")
                    unpacked = self._reorder_per_v_head_rows(unpacked)
                    if target_dtype != GGMLQuantizationType.BF16:
                        # the bf16 prefill copy is a plain dense tensor, only the
                        # chunked copy has to land on a chunk boundary
                        unpacked = self._pad_rows_to_chunk(unpacked)

                elif "ssm_conv1d" in new_name:
                    print("[INFO] transpose conv1d")
                    d = self._reorder_qkv(unpacked)[0]
                    unpacked = [d.T.contiguous()]

                elif name.endswith("ssm_a"):
                    self.q4nx_tensors[new_name] = self._reorder_per_v_head_vector(
                        unpacked[0].to(torch.float32)).contiguous()
                    continue

                elif "ssm_dt" in name:
                    self.q4nx_tensors[new_name] = self._reorder_per_v_head_vector(
                        unpacked[0].to(torch.float32)).contiguous()
                    continue

            self.q4nx_tensors[new_name] = self._pack(*unpacked, tensor_type=target_dtype)

        self._extract_tokenizer_json(q4nx_path)
        self._export_q4nx_tensors(q4nx_path)
