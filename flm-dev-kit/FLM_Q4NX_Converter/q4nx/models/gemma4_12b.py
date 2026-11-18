# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from einops import rearrange
from gguf import GGMLQuantizationType, GGUFReader, dequantize, quantize
import torch

from ..model_converter import __Q4NX_Converter
from ..gguf_tensor import GGUFTensor
from ..constants import ModelArch


class Gemma4_12B(__Q4NX_Converter, model_arch=ModelArch.GEMMA4_12B):
    """Gemma-4-12B (QAT).

    Standalone converter: same tensor graph as the smaller Gemma 4 models (no per-layer
    embeddings, no output.weight, 8 of the 48 layers are global attention and carry no
    v_proj), but the GGUF mixes quantization types -- the body is Q4_0 while
    token_embd.weight is Q8_0 (q40-q80-lmhead.gguf) or Q6_K (the published QAT file). So
    every language tensor is packed in the type it actually carries instead of being
    forced through the Q4 packer, which would bit-truncate the head into 4 bits.

    Blocks are 16x256 (configs/gemma4_12b.json), matching QXNX_ROW_BLOCK_SIZE in the
    kernels: flm_q40 is 2304 B (256 B bf16 scales + 2048 B signed nibbles, no zero-point
    slice) and flm_q80 is 4352 B (256 B scales + 4096 B int8). Both agree with
    Q4NX_BLOCK_BYTES / Q8NX_BLOCK_BYTES in FLM_DLL/detail/gemma4_12b_npu/.

    Because the model ties its head, lm_head.weight is packed from token_embd.weight, and
    always as q8_0 regardless of what the GGUF carries -- gemma4_12b_desc::LM_HEAD_DTYPE
    is flm_q80 and reorder_cpy asserts that block size, so a Q6_K or Q4_0 embedding is
    dequantized and re-quantized to Q8_0 rather than inheriting its own type. That is
    free for q40-q80-lmhead.gguf (already Q8_0, passed through bit-exact) and costs one
    requantization for the published QAT file. The runtime reads these tensors by byte
    size only, so the keep_block_in_2D config flag changes the recorded shape
    ([p, q, bytes] vs [(p q), bytes]) but not a single byte of payload.

    Caveat on the embedding: carrying the Q8_0 type through only preserves whatever
    precision the GGUF actually holds. The published QAT GGUFs store token_embd.weight
    in a Q8_0/Q6_K container whose values were already rounded onto a 4-bit grid, so the
    BF16 table this converter emits is Q4-grade (~0.992 cosine against the BF16
    checkpoint) even though nothing here truncates it. _report_embedding_fidelity flags
    that at convert time; --embed-bf16 replaces the table with the real BF16 weights,
    which costs nothing because q4nx stores this tensor as BF16 either way.
    """

    def __init__(self, gguf_reader: GGUFReader):
        print("[INFO] Using Gemma4_12B converter")
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

        # read "embedding_length_per_layer_input" from gguf metadata (0 for the 12B)
        field = self.gguf_reader.fields.get("gemma4.embedding_length_per_layer_input", None)
        self.embedding_length_per_layer_input = field.contents() if field is not None else None

    def initialize(self):
        super().initialize()

    def reshape_matrix_to_block_matrix_for_mvm(self, weight: torch.Tensor, row_block_size: int = 32) -> torch.Tensor:
        """
            Assume the weights is a 2D matrix of W x H
            The function will reshape weights into (blocks, H, row_block_size)
            Where blocks = W // row_block_size

            In addition, each block(row_block_size x H) is in column-major order
        """
        assert weight.ndim == 2, "Input weight must be a 2D matrix"

        W, H = weight.shape

        assert W % row_block_size == 0, f"Weight rows {W} must be divisible by row_block_size {row_block_size}"
        blocks = W // row_block_size

        # Step 1: Reshape to (blocks, row_block_size, H)
        weight = weight.contiguous()

        weight = rearrange(weight,
                           '(blocks row_block_size) H -> blocks row_block_size H',
                           blocks=blocks, row_block_size=row_block_size
                           )
        # Step 2: Transpose each block to be in column-major order
        weight = rearrange(weight,
                           'blocks row_block_size H -> blocks H row_block_size',
                           )
        return weight

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}

        if weights_type == "language":

            # token_embd feeds two tensors -- the BF16 input table and (because this
            # model ties its head) lm_head. Resolve it once so both come from the same
            # source, whether that is the GGUF or an external BF16 checkpoint.
            embd = self.gguf_tensors["token_embd.weight"]
            embd_hidden, embd_vocab = embd.shape[0], embd.shape[1]
            assert embd_hidden == self.hidden_size, \
                f"token_embd.weight has hidden {embd_hidden}, metadata says {self.hidden_size}"
            # Kept in fp32: the only consumer below multiplies by sqrt(hidden_size) and
            # rounds once at the end. Materializing a bf16 copy here instead would round
            # twice -- worth ~0.25% relative L2 on the table, for nothing.
            embd_bf16 = self._load_bf16_embedding(embd_vocab, embd_hidden)
            if embd_bf16 is None:
                embd_w = torch.from_numpy(
                    dequantize(embd.data, embd.tensor_type)).contiguous().to(torch.float32)
                embd_src_type = embd.tensor_type
            else:
                embd_w = embd_bf16.to(torch.float32)
                embd_src_type = None
            self._report_embedding_fidelity(embd_w, embd_src_type)

            if not self._has_lm_head():
                # Fixed at Q8_0, not inherited from the embedding: the runtime hardcodes
                # gemma4_12b_desc::LM_HEAD_DTYPE = flm_q80 and reorder_cpy asserts the
                # block size, so a GGUF carrying a Q4_0 or Q6_K token_embd must still
                # come out as 4352 B q80 blocks. Passing it to unpack() as well keeps a
                # non-native source (Q6_K) from being routed through the config default
                # (Q4_0), which would bit-truncate the head into a q80 container.
                target_dtype = GGMLQuantizationType.Q8_0
                print(f"[INFO] Model does not have a lm_head, packing token_embd.weight as {target_dtype.name}")
                if embd_bf16 is None:
                    unpacked = embd.unpack(target_dtype)
                else:
                    # Re-quantize the BF16 table into the same type the GGUF would have
                    # carried, then reuse the ordinary unpack path so the packer sees
                    # bit-identical inputs to the GGUF case.
                    qbytes = quantize(embd_w.numpy(), target_dtype).copy()
                    unpacked = GGUFTensor(name="token_embd.weight", shape=embd.shape,
                                          data=qbytes, tensor_type=target_dtype
                                          ).unpack(target_dtype)
                self.q4nx_tensors["lm_head.weight"] = self._pack(*unpacked, tensor_type=target_dtype)

            for key, gguf_tensor in self.gguf_tensors.items():
                if "token_embd.weight" == gguf_tensor.name:  # this should be bf16
                    # Gemma scales the input embedding by sqrt(hidden_size); bake it in
                    # here so Embedding::forward stays a plain table lookup. lm_head above
                    # is deliberately left un-scaled.
                    w = (embd_w * float(self.hidden_size) ** 0.5).contiguous().to(torch.bfloat16)
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                    continue

                unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])

                torch.set_printoptions(threshold=16, edgeitems=5, linewidth=200)


                target_dtype = gguf_tensor.get_used_quantization_type(self.tensor_q4nx_type_map[gguf_tensor.name])
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack(*unpacked, tensor_type=target_dtype)

            self._extract_tokenizer_json(q4nx_path)
        elif weights_type == "vision":
            self._check_vision_source_is_bf16()

            for key, gguf_tensor in self.gguf_tensors.items():
                unpacked = gguf_tensor.unpack(GGMLQuantizationType.BF16)
                assert len(unpacked) == 1
                assert type(unpacked[0]) == torch.Tensor, "Vision model tensors"

                weights = unpacked[0]
                if weights.dtype != torch.bfloat16:
                    # convert to bfloat16
                    weights = weights.to(torch.bfloat16)
                if gguf_tensor.name not in self.forward_name_map:
                    # check is not start with "v"
                    if not gguf_tensor.name.startswith("v"):
                        continue
                    else:
                        raise ValueError(f"Tensor name {gguf_tensor.name} not found in forward_name_map for vision model")
                new_name = self.forward_name_map[gguf_tensor.name]

                if new_name == "model.vision.embedding_projection.weight":
                    # TODO: no need transpose?
                    weights = self.vision_mm_weight_rearrange(weights)
                elif new_name == "model.vision.patch_embedder.position_embedding_table":
                    assert weights.ndim == 3
                    assert weights.shape[0] == 2
                    # TODO: also transpose each weights first ?
                    weights = torch.stack([
                        self.vision_mm_weight_rearrange(weights[0].T.contiguous()),
                        self.vision_mm_weight_rearrange(weights[1].T.contiguous()),
                    ])

                elif new_name == "model.vision.patch_embd.weight":
                    assert (weights.ndim == 4), f"Expected patch embedding weight to be 4D, but got {weights.ndim}D"

                    # Revert llama.cpp's permute(0, 3, 1, 2) to get back to original 2D (n_embd, ksize_sq_c)
                    weights = weights.permute(0, 2, 3, 1).contiguous()
                    weights = weights.reshape(weights.shape[0], -1)
                    # weights = weights.T.contiguous()

                    weights = self.vision_mm_weight_rearrange(weights)
                elif new_name.endswith("ffn_down.weight") or new_name.endswith("ffn_gate.weight") or new_name.endswith("ffn_up.weight") \
                        or new_name.endswith("k_proj.weight") or new_name.endswith("q_proj.weight") \
                        or new_name.endswith("v_proj.weight") or new_name.endswith("out_proj.weight") \
                        or new_name.endswith("gate_proj.weight") or new_name.endswith("up_proj.weight") or new_name.endswith("down_proj.weight"):

                    weights = self.vision_mm_weight_rearrange(weights)

                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = weights

        elif weights_type == "audio":
            for key, gguf_tensor in self.gguf_tensors.items():
                unpacked = gguf_tensor.unpack(GGMLQuantizationType.BF16)
                assert len(unpacked) == 1
                assert type(unpacked[0]) == torch.Tensor, "Audio model tensors"

                weights = unpacked[0]
                if weights.dtype != torch.bfloat16:
                    # convert to bfloat16
                    weights = weights.to(torch.bfloat16)

                if gguf_tensor.name not in self.forward_name_map:
                    if not gguf_tensor.name.startswith("a"):
                        continue
                    else:
                        raise ValueError(f"Tensor name {gguf_tensor.name} not found in forward_name_map for audio model")
                new_name = self.forward_name_map[gguf_tensor.name]

                if new_name.endswith("conv_dw.weight"):
                    weights = weights.T.contiguous()
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = weights
                    continue

                matrix_suffix_for_weight_rearrange = [
                    "attn_k_proj.weight", "attn_out_proj.weight",
                    # "attn_k_proj.rel_weight",
                    "attn_q_proj.weight", "attn_v_proj.weight",
                    "ffn.down_proj.weight", "ffn.down_proj_1.weight",
                    "ffn.up_proj.weight", "ffn.up_proj_1.weight",
                    "conv_pw_1.weight", "conv_pw_2.weight",
                    "model.audio.embedding_projection.weight",
                    "model.audio.encode_input_projection.weight",
                    "model.audio.pre_encoder.weight"
                ]

                for suf in matrix_suffix_for_weight_rearrange:

                    if new_name.endswith(suf) or new_name == suf:
                        weights = self.audio_mm_weight_rearrange(weights).contiguous()
                        break

                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = weights

        else:
            raise ValueError(f"Unsupported weights_type: {weights_type} for Gemma4_12B model")
        self._export_q4nx_tensors(q4nx_path)
