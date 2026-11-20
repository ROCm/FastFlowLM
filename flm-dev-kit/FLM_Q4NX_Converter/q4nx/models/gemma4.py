# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import gguf

from ..model_converter import __Q4NX_Converter
from ..gguf_tensor import GGUFTensor
from ..constants import ModelArch
from gguf import GGMLQuantizationType, GGUFReader, dequantize, quantize
from safetensors.torch import save_file
import numpy as np
import torch
from gguf import dequantize
from einops import rearrange, unpack

class Gemma4(__Q4NX_Converter, model_arch=ModelArch.GEMMA4):
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

        
        # read "embedding_length_per_layer_input" from gguf metadata
        field = self.gguf_reader.fields.get("gemma4.embedding_length_per_layer_input", None)
        self.embedding_length_per_layer_input = field.contents() if field is not None else None 
    def initialize(self):
        super().initialize()
        self._requantize_for_q4_1_engine()

    def _requantize_for_q4_1_engine(self):
        """Make a Q4_K GGUF (e.g. NVIDIA's Nemotron-3.5-Bolt-E4B Q4_K_M) look like the Q4_1 GGUF this
        converter was written for; the Gemma4 engine's kernels read Q4_1, not Q4_K.

        - Q4_K tensors the config keeps quantized: exact repack to Q4_1 (see repack_q4_k_as_q4_1).
        - Q4_K tensors the config wants in BF16 (inp_gate, proj): dequantize; convert() reads them
          as BF16 and would otherwise take a Q4_K scale plane for the weights.
        - Q6_K and others: left alone, unpack() already dequantizes and requantizes them to Q4_1.
        """
        for name, src in list(self.gguf_tensors.items()):
            if src.tensor_type != GGMLQuantizationType.Q4_K or name not in self.tensor_q4nx_type_map:
                continue
            target = self.tensor_q4nx_type_map[name]
            if target in (GGMLQuantizationType.BF16, GGMLQuantizationType.F16, GGMLQuantizationType.F32):
                w = torch.from_numpy(dequantize(src.data, src.tensor_type)).contiguous().to(torch.bfloat16)
                self.gguf_tensors[name] = GGUFTensor(name=name, shape=src.shape,
                                                     data=w.view(torch.uint8).numpy(),  # GGUFReader's BF16 layout
                                                     tensor_type=GGMLQuantizationType.BF16)
            else:
                self.gguf_tensors[name] = GGUFTensor(
                    name=name, shape=src.shape,
                    data=GGUFTensor.repack_q4_k_as_q4_1(src.data, int(src.shape[0])),
                    tensor_type=GGMLQuantizationType.Q4_1)

    
    
    
    def _store_embedding_q8(self, gguf_tensor: GGUFTensor, factor: float):
        """Store an embedding table the way gemma4e_npu's embedding_q8_0 reads it: row-major int8
        `<name>` [vocab, dim] and float32 `<name>.scale` [vocab, dim / 32], one scale per group of 32.

        Gemma scales embeddings by sqrt(width) after the lookup; the engine does not, so that factor is
        folded into the scale. A Q8_0 source is kept as it is (its FP16 scale times `factor`, exact in
        float32); any other source is dequantized and quantized to Q8_0 first.
        """
        name = self.forward_name_map[gguf_tensor.name]
        data = gguf_tensor.data
        if gguf_tensor.tensor_type != GGMLQuantizationType.Q8_0:
            data = quantize(dequantize(data, gguf_tensor.tensor_type), GGMLQuantizationType.Q8_0)
        vocab = data.shape[0]
        blocks = np.ascontiguousarray(data).view(np.uint8).reshape(vocab, -1, 34)  # fp16 d + 32 int8
        d = blocks[:, :, :2].copy().view(np.float16).astype(np.float32).reshape(vocab, -1)
        q = blocks[:, :, 2:].copy().view(np.int8).reshape(vocab, -1)
        self.q4nx_tensors[name] = torch.from_numpy(q).contiguous()
        self.q4nx_tensors[name + ".scale"] = torch.from_numpy((d * np.float32(factor)).astype(np.float32)).contiguous()

    def reshape_matrix_to_block_matrix_for_mvm(self, weight: torch.Tensor, row_block_size: int=32) -> torch.Tensor:
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
            
            if not self._has_lm_head():
                print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
                # The lm_head kernel reads Q4_1. unpack() keeps a Q8_0 / Q4_K source as it is, and feeding its
                # int8 values to the 4-bit packer gives a lm_head of the right size and garbage content, so
                # requantize to the default type first (a Q4_1 embedding passes through unchanged).
                embd = self.gguf_tensors["token_embd.weight"]
                if embd.tensor_type != self.default_tensor_type:
                    w = dequantize(embd.data, embd.tensor_type)
                    embd = GGUFTensor(name=embd.name, shape=embd.shape,
                                      data=quantize(w, self.default_tensor_type),
                                      tensor_type=self.default_tensor_type)
                unpacked = embd.unpack(self.default_tensor_type)
                self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)

            for key, gguf_tensor in self.gguf_tensors.items():
                if "token_embd.weight"  == gguf_tensor.name:
                    self._store_embedding_q8(gguf_tensor, float(self.hidden_size) ** 0.5)
                    continue
                elif "per_layer_token_embd.weight"  ==  gguf_tensor.name:
                    self._store_embedding_q8(gguf_tensor, float(self.embedding_length_per_layer_input) ** 0.5)
                    continue
                elif "per_layer_model_proj.weight" in gguf_tensor.name:
                    unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])
                    w=unpacked[0]
                    
                    # since output projection == self.embedding_length_per_layer_input* self.num_layers, we want to actually split the matrxi
                    
                    assert w.shape[0] == self.embedding_length_per_layer_input * self.num_layers, f"Expected output projection weight shape[0] to be {self.embedding_length_per_layer_input * self.num_layers}, but got {w.shape[0]}"
                    
                    # let us save a prefill version of  W for debugging
                    #NOTE: TODO: We use vision MM
                    w_for_prefill = self.vision_mm_weight_rearrange(w).contiguous().to(torch.bfloat16)
                    self.q4nx_tensors[f"{self.forward_name_map[gguf_tensor.name]}_prefill"] = w_for_prefill
                    
                    w_per_layer = w.reshape(self.num_layers, self.embedding_length_per_layer_input, w.shape[1])
                    
                    for layer_idx in range(self.num_layers):
                        layer_w = w_per_layer[layer_idx]
                        layer_w = self.reshape_matrix_to_block_matrix_for_mvm(layer_w)
                        layer_w = layer_w.contiguous().to(torch.bfloat16)
                        self.q4nx_tensors[f"{self.forward_name_map[gguf_tensor.name]}_layer{layer_idx}"] = layer_w
                    
                    
                    # w = self.reshape_matrix_to_block_matrix_for_mvm(unpacked[0])
                    # w = w.contiguous().to(torch.bfloat16)
                    # self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w

                    continue
                                        
                elif "inp_gate.weight" in gguf_tensor.name or ".proj.weight" in gguf_tensor.name:
                    
                    unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])
                    
                    
                    if "inp_gate.weight" in gguf_tensor.name or ".proj.weight" in gguf_tensor.name:
                        # dedicate for prefill with vision MM
                        w_for_prefill = self.vision_mm_weight_rearrange(unpacked[0]).contiguous().to(torch.bfloat16)
                        self.q4nx_tensors[f"{self.forward_name_map[gguf_tensor.name]}_prefill"] = w_for_prefill                    
                    
                    
                    
                    w = self.reshape_matrix_to_block_matrix_for_mvm(unpacked[0])
                    w = w.contiguous().to(torch.bfloat16)
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                    
                    continue
                
                unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])

                torch.set_printoptions(threshold=16, edgeitems=5, linewidth=200)

                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_q4nx(*unpacked)

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
                    #TODO: no need transpose?
                    
                    weights = self.vision_mm_weight_rearrange(weights)                    
                elif new_name == "model.vision.patch_embedder.position_embedding_table":
                    assert weights.ndim == 3
                    assert weights.shape[0] ==2
                    #TODO: also transpose each weights first ?
                    
                    
                    weights = torch.stack([
                        self.vision_mm_weight_rearrange(weights[0].T.contiguous()),
                        self.vision_mm_weight_rearrange(weights[1].T.contiguous()),
                    ])
                    
                elif new_name == "model.vision.patch_embd.weight":
                    assert(weights.ndim == 4), f"Expected patch embedding weight to be 4D, but got {weights.ndim}D"
                    
                    # Revert llama.cpp's permute(0, 3, 1, 2) to get back to original 2D (n_embd, ksize_sq_c)
                    weights = weights.permute(0, 2, 3, 1).contiguous()
                    weights = weights.reshape(weights.shape[0], -1)
                    # weights = weights.T.contiguous()
                    
                    weights = self.vision_mm_weight_rearrange(weights)
                elif new_name.endswith("ffn_down.weight") or new_name.endswith("ffn_gate.weight") or new_name.endswith("ffn_up.weight") \
                    or new_name.endswith("k_proj.weight") or new_name.endswith("q_proj.weight")\
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
            raise ValueError(f"Unsupported weights_type: {weights_type} for Gemma4 model")
        self._export_q4nx_tensors(q4nx_path)
