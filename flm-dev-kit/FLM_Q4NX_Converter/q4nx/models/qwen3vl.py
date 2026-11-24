# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, quantize, GGMLQuantizationType
from safetensors.torch import save_file
import torch

class Qwen3VL(__Q4NX_Converter, model_arch=ModelArch.QWEN3VL):
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}
        if weights_type == "language":
            if not self._has_lm_head():
                print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
                unpacked = self.gguf_tensors["token_embd.weight"].unpack(self.default_tensor_type)
                self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)

            for key, gguf_tensor in self.gguf_tensors.items():
                if "token_embd.weight" in gguf_tensor.name: # this should be bf16
                    w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                    w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                    continue

                unpacked = gguf_tensor.unpack(self.default_tensor_type)
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
                    
                new_name = self.forward_name_map[gguf_tensor.name]                        
                
                if new_name.endswith("fc2.weight") or new_name.endswith("fc1.weight")\
                    or new_name.endswith("attn.proj.weight") or new_name.endswith("attn.qkv.weight"):
                    weights = self.vision_mm_weight_rearrange(weights)
                
                self.q4nx_tensors[new_name] = weights
                
            
            # now, we first do special case for merge two patch_embd weights 
            combined_patched_embeding= torch.stack(
                [self.q4nx_tensors["model.visual.patch_embed.proj.weight"],
                         self.q4nx_tensors["model.visual.patch_embed.proj.weight.1"]
                 ], dim=2
            )
            # delete the two original weights
            del self.q4nx_tensors["model.visual.patch_embed.proj.weight"]
            del self.q4nx_tensors["model.visual.patch_embed.proj.weight.1"]
            self.q4nx_tensors["model.visual.patch_embed.proj.weight"] = combined_patched_embeding
            
            
            
    
        else:
            raise ValueError(f"Unsupported weights_type: {weights_type} for Qwen3VL model")

        self._export_q4nx_tensors(q4nx_path)
