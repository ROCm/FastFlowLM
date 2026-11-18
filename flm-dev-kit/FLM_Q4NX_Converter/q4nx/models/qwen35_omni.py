# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from ..constants import ModelArch
from .qwen35 import Qwen35
from gguf import GGUFReader, GGMLQuantizationType
import torch
from gguf import GGUFReader, dequantize, quantize, GGMLQuantizationType

from einops import rearrange, repeat

class Qwen35Omni(Qwen35, model_arch=ModelArch.QWEN35_OMNI):
    vision_source_policy = "warn"   # vision conversion not checked on a real BF16 mmproj yet
    """Qwen3.5 Omni thinker converter.

    Same layout as Qwen3.5, with one difference: quantized weights (Q4_1) keep
    their quantization, and every other tensor is stored as bf16. Only ssm_a and
    ssm_dt.bias stay f32 (handled in the base converter's ssm_a/ssm_dt paths,
    which bypass _pack).
    """

    def __init__(self, gguf_reader: GGUFReader):
        print("[INFO] Using Qwen35Omni converter")
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def convert(self, q4nx_path:str, weights_type:str="language"):
        
        self.q4nx_tensors = {}
        
        if weights_type== "language":
            reorder_linear_required = False
            if reorder_linear_required:
                print("[INFO] Reorder linear required!")

            self._drop_mtp_tensors()
            if not self._has_lm_head():
                print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
                unpacked = self.gguf_tensors["token_embd.weight"].unpack(GGMLQuantizationType.Q8_0)
                target_dtype = self.gguf_tensors["token_embd.weight"].get_used_quantization_type(GGMLQuantizationType.Q8_0)
                self.q4nx_tensors["lm_head.weight"] = self._pack(*unpacked, tensor_type=target_dtype)

            for key, gguf_tensor in self.gguf_tensors.items():
                target_dtype = gguf_tensor.get_used_quantization_type(self.tensor_q4nx_type_map[gguf_tensor.name])
                print(f"Processing tensor: {gguf_tensor.name} with type {gguf_tensor.tensor_type.name} -> {self.forward_name_map[gguf_tensor.name]} with dtype {target_dtype.name}")
                if "token_embd.weight" in gguf_tensor.name: # this should be bf16
                    # Use lm_head directly
                    continue
                
                new_name = self.forward_name_map[gguf_tensor.name]
                layer_id = 0
                if "layers." in new_name:
                    layer_id = int(new_name.split("layers.")[1].split(".")[0])

                unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])

                if self._is_full_attention_layer(layer_id):
                    if "q_proj" in self.forward_name_map[gguf_tensor.name]: # for llama q_proj, the order is special
                        print("[INFO] Seperate q, gate for q_proj")
                        DH = self.gguf_reader.fields["qwen35.attention.value_length"].contents()
                        d, m, qw = unpacked
                        d = rearrange(d, '(g p h) c -> (p g h) c', p = 2, h = DH).contiguous()
                        m = rearrange(m, '(g p h) c -> (p g h) c', p = 2, h = DH).contiguous()
                        qw = rearrange(qw, '(g p h) c -> (p g h) c', p = 2, h = DH).contiguous()
                        unpacked = (d, m, qw)

                else: # linear attention
                    if "self_attn.gate_proj" in self.forward_name_map[gguf_tensor.name]: # for llama q_proj, the order is special
                        if reorder_linear_required:
                            print("[INFO] Reorder Gate")
                            DH = self.gguf_reader.fields["qwen35.ssm.state_size"].contents()
                            d, m, qw = unpacked
                            d = rearrange(d, '(q g p) c -> (g q p) c', p = DH, q = 2).contiguous()
                            m = rearrange(m, '(q g p) c -> (g q p) c', p = DH, q = 2).contiguous()
                            qw = rearrange(qw, '(q g p) c -> (g q p) c', p = DH, q = 2).contiguous()
                            unpacked = (d, m, qw)

                    if "qkv_proj" in self.forward_name_map[gguf_tensor.name]: # for llama q_proj, the order is special
                        if reorder_linear_required:
                            print("[INFO] Seperate q, gate for q_proj")
                            DH = self.gguf_reader.fields["qwen35.ssm.state_size"].contents()
                            d, m, qw = unpacked
                            d0, d1 = d.chunk(2, dim = 0)
                            m0, m1 = m.chunk(2, dim = 0)
                            qw0, qw1 = qw.chunk(2, dim = 0)
                            print(d0.shape, d1.shape, m0.shape, m1.shape, qw0.shape, qw1.shape)
                            pp = DH

                            d1 = rearrange(d1, '(q g p) c -> (g q p) c', p = pp, q = 2).contiguous()
                            m1 = rearrange(m1, '(q g p) c -> (g q p) c', p = pp, q = 2).contiguous()
                            qw1 = rearrange(qw1, '(q g p) c -> (g q p) c', p = pp, q = 2).contiguous()

                            d = torch.cat([d0, d1], dim = 0).contiguous()
                            m = torch.cat([m0, m1], dim = 0).contiguous()
                            qw = torch.cat([qw0, qw1], dim = 0).contiguous()
                            unpacked = (d, m, qw)

                    if "ssm_out_proj" in self.forward_name_map[gguf_tensor.name]: # for ssm_alpha_proj, the order is special
                        if reorder_linear_required:
                            print(f"[INFO] Reorder for {self.forward_name_map[gguf_tensor.name]}")
                            d, m, qw = unpacked
                            DH = self.gguf_reader.fields["qwen35.ssm.state_size"].contents()
                            DH = DH // 32 # reorder in group size
                            BLOCK_SIZE = 32
                            d = rearrange(d, 'r (q g p) -> r (g q p)', p = DH, q = 2).contiguous()
                            m = rearrange(m, 'r (q g p) -> r (g q p)', p = DH, q = 2).contiguous()
                            # qw has 32 elements per block vs 1 for d/m, so use DH * BLOCK_SIZE to keep block alignment
                            qw = rearrange(qw, 'r (q g p) -> r (g q p)', p = DH * BLOCK_SIZE, q = 2).contiguous()

                            unpacked = (d, m, qw)

                    if "ssm_alpha_proj" in self.forward_name_map[gguf_tensor.name] or "ssm_beta_proj" in self.forward_name_map[gguf_tensor.name]: # for ssm_alpha_proj, the order is special
                        d, m, qw = unpacked
                        # save bf16 copy
                        w = gguf_tensor.dequantize()
                        if reorder_linear_required:
                            w = rearrange(w, '(q g) c -> (g q) c', q = 2).contiguous()

                        new_name = self.forward_name_map[gguf_tensor.name]
                        new_name = new_name.replace("alpha_proj", "alpha_proj.bf16").replace("beta_proj", "beta_proj.bf16")
                        self.q4nx_tensors[new_name] = w
                        if reorder_linear_required:
                            print(f"[INFO] Reorder for {self.forward_name_map[gguf_tensor.name]}")
                            d = rearrange(d, '(q g) c -> (g q) c', q = 2).contiguous()
                            m = rearrange(m, '(q g) c -> (g q) c', q = 2).contiguous()
                            qw = rearrange(qw, '(q g) c -> (g q) c', q = 2).contiguous()

                        if (d.shape[0] < 32): # duplicate on the first dimension for better performance when the state size is small
                            d = repeat(d, 'd c -> (r d) c', r = 2).contiguous()
                            m = repeat(m, 'd c -> (r d) c', r = 2).contiguous()
                            qw = repeat(qw, 'd c -> (r d) c', r = 2).contiguous()

                        unpacked = (d, m, qw)


                    if "ssm_conv1d" in self.forward_name_map[gguf_tensor.name]:
                        print("[INFO] transpose conv1d")

                        DH = self.gguf_reader.fields["qwen35.ssm.state_size"].contents()
                        d = unpacked[0]
                        
                        if reorder_linear_required:
                            d0, d1 = d.chunk(2, dim = 0)

                            d1 = rearrange(d1, '(q g p) c -> (g q p) c', p = DH, q = 2).contiguous()
                        
                            d = torch.cat([d0, d1], dim = 0).contiguous()
                        d = d.T.contiguous()
                        unpacked = [d]
                
                    if "ssm_a" in gguf_tensor.name[-5:]: # [-5:] to avoid confusing with "ssm_alpha_proj" or "ssm_dt_proj"
                        val = unpacked[0].to(torch.float32).contiguous()
                        if reorder_linear_required:
                            val = rearrange(val, '(q g) -> (g q)', q = 2).contiguous() # for ssm_a, we also need to reorder the g and p dim
                        self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = val
                        continue

                    if "ssm_dt" in gguf_tensor.name: # [-5:] to avoid confusing with "ssm_alpha_proj" or "ssm_dt_proj"
                        val = unpacked[0].to(torch.float32).contiguous()
                        if reorder_linear_required:
                            val = rearrange(val, '(q g) -> (g q)', q = 2).contiguous() # for ssm_dt, we also need to reorder the g and p dim
                        self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = val
                        continue

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
                    
                new_name = self.forward_name_map[gguf_tensor.name]                        
                
                if new_name.endswith("fc2.weight") or new_name.endswith("fc1.weight")\
                    or new_name.endswith("attn.proj.weight") or new_name.endswith("attn.qkv.weight"):
                    weights = self.vision_mm_weight_rearrange(weights)
                
                self.q4nx_tensors[new_name] = weights
                
            
            
            
            
            if "model.visual.patch_embed.proj.weight.1" in self.q4nx_tensors.keys():
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
        elif weights_type == "audio":
            for key, gguf_tensor in self.gguf_tensors.items():
                unpacked = gguf_tensor.unpack(GGMLQuantizationType.BF16)
                assert len(unpacked) == 1
                assert type(unpacked[0]) == torch.Tensor, "Vision model tensors"
                weights = unpacked[0]
                if weights.dtype != torch.bfloat16:
                    # convert to bfloat16
                    weights = weights.to(torch.bfloat16)
                    
                new_name = self.forward_name_map[gguf_tensor.name]                        
                

                matrix_suffix_for_weight_rearrange = [
                    "fc1.weight","fc2.weight",
                    "k_proj.weight", "out_proj.weight",
                    "q_proj.weight", "v_proj.weight",
                    "conv_out.weight", 
                    "proj1.weight", "proj2.weight"
                ]
                
                
                for suf in matrix_suffix_for_weight_rearrange:

                    if new_name.endswith(suf) or new_name == suf:
                        weights = self.audio_mm_weight_rearrange(weights).contiguous()
                        break

                self.q4nx_tensors[new_name] = weights
                
            
            
        
        else:
            
            raise ValueError(f"Unsupported weights_type {weights_type} for Qwen3.5 omni model")
        self._export_q4nx_tensors(q4nx_path)