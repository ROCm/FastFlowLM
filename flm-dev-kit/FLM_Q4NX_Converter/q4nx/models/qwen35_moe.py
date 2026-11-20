# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from pprint import pp

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, quantize, GGMLQuantizationType
from safetensors.torch import save_file
from einops import rearrange, repeat
import torch

class QWEN35_A3B(__Q4NX_Converter, model_arch=ModelArch.QWEN35_A3B):
    def __init__(self, gguf_reader: GGUFReader):
        print("[INFO] Using Qwen35_A3BB converter")
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        print("Converting moe!")
        self.q4nx_tensors = {}
        if weights_type == "language":
            reorder_linear_required = True
            # if self.gguf_reader.fields["qwen35moe.expert_feed_forward_length"].contents() <= 6144 :
            #     #TODO: Check this # small model, no qk share on linear head
            #     reorder_linear_required = False
            if reorder_linear_required:
                print("[INFO] Reorder linear required!")

            self._drop_mtp_tensors()
            if not self._has_lm_head():
                print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
                unpacked = self.gguf_tensors["token_embd.weight"].unpack(GGMLQuantizationType.Q8_0)
                target_dtype = self.gguf_tensors["token_embd.weight"].get_used_quantization_type(GGMLQuantizationType.Q8_0)
                self.q4nx_tensors["lm_head.weight"] = self._pack(*unpacked, tensor_type=target_dtype)

            for key, gguf_tensor in self.gguf_tensors.items():
                if gguf_tensor.name not in self.forward_name_map:
                    # Some GGUFs ship an extra unquantized copy of alpha/beta
                    # (blk.{bid}.ssm_{alpha,beta}.bf16.weight).  The bf16 copy is
                    # derived from the quantized tensor below, so drop the spare.
                    print(f"[WARN] No name mapping for {gguf_tensor.name}, skipping")
                    continue

                target_dtype = gguf_tensor.get_used_quantization_type(self.tensor_q4nx_type_map[gguf_tensor.name])
                print(f"Processing tensor: {gguf_tensor.name} with type {gguf_tensor.tensor_type.name} -> {self.forward_name_map[gguf_tensor.name]} with dtype {target_dtype.name}")
                if "token_embd.weight" in gguf_tensor.name: # this should be bf16
                    w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                    w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                    continue
                
                elif "norm.weight" in  self.forward_name_map[gguf_tensor.name]:
                    # save in bf16
                    w =gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w[0].to(torch.bfloat16).contiguous()
                    continue
                elif "moe_router.weight" in self.forward_name_map[gguf_tensor.name]:
                    # for now, keep in row major order of fp32
                    w =gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w[0].to(torch.bfloat16).clone().transpose(0,1).contiguous()
                    continue
                elif "shared_expert_gate.weight" in self.forward_name_map[gguf_tensor.name]:
                    w =gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w[0].to(torch.bfloat16).contiguous()
                    continue
                new_name = self.forward_name_map[gguf_tensor.name]
                layer_id = 0
                if "model.layer." in new_name:
                    layer_id = int(new_name.split("model.layer.")[1].split(".")[0])

                unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])

                if self._is_full_attention_layer(layer_id):
                    if "q_proj" in self.forward_name_map[gguf_tensor.name]: # for llama q_proj, the order is special
                        print("[INFO] Seperate q, gate for q_proj")
                        DH = self.gguf_reader.fields["qwen35moe.attention.value_length"].contents()
                        d, m, qw = unpacked
                        d = rearrange(d, '(g p h) c -> (p g h) c', p = 2, h = DH).contiguous()
                        m = rearrange(m, '(g p h) c -> (p g h) c', p = 2, h = DH).contiguous()
                        qw = rearrange(qw, '(g p h) c -> (p g h) c', p = 2, h = DH).contiguous()
                        unpacked = (d, m, qw)

                else: # linear attention
                    if "self_attn.gate_proj" in self.forward_name_map[gguf_tensor.name]: # for llama q_proj, the order is special
                        if reorder_linear_required:
                            print("[INFO] Reorder Gate")
                            DH = self.gguf_reader.fields["qwen35moe.ssm.state_size"].contents()
                            d, m, qw = unpacked
                            d = rearrange(d, '(q g p) c -> (g q p) c', p = DH, q = 2).contiguous()
                            m = rearrange(m, '(q g p) c -> (g q p) c', p = DH, q = 2).contiguous()
                            qw = rearrange(qw, '(q g p) c -> (g q p) c', p = DH, q = 2).contiguous()
                            unpacked = (d, m, qw)

                    if "qkv_proj" in self.forward_name_map[gguf_tensor.name]: # for llama q_proj, the order is special
                        if reorder_linear_required:
                            print("[INFO] Seperate q, gate for q_proj")
                            DH = self.gguf_reader.fields["qwen35moe.ssm.state_size"].contents()
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
                            DH = self.gguf_reader.fields["qwen35moe.ssm.state_size"].contents()
                            DH = DH // 32 # reorder in group size
                            BLOCK_SIZE = 32
                            d = rearrange(d, 'r (q g p) -> r (g q p)', p = DH, q = 2).contiguous()
                            m = rearrange(m, 'r (q g p) -> r (g q p)', p = DH, q = 2).contiguous()
                            # qw has 32 elements per block vs 1 for d/m, so use DH * BLOCK_SIZE to keep block alignment
                            qw = rearrange(qw, 'r (q g p) -> r (g q p)', p = DH * BLOCK_SIZE, q = 2).contiguous()

                            unpacked = (d, m, qw)

                    if "ssm_alpha_proj" in self.forward_name_map[gguf_tensor.name] or "ssm_beta_proj" in self.forward_name_map[gguf_tensor.name]:
                        
                        if len(unpacked) == 3:
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
                        else:
                            # save in bf16 into the file
                            w = unpacked[0].contiguous().to(torch.bfloat16)
                            if reorder_linear_required:
                                w = rearrange(w, '(q g) c -> (g q) c', q = 2).contiguous()
                            # save matrix in colum major order instead
                            self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w.clone().transpose(0,1).contiguous()
                            continue
                            # is in fp16


                    if "ssm_conv1d" in self.forward_name_map[gguf_tensor.name]:
                        print("[INFO] transpose conv1d")

                        DH = self.gguf_reader.fields["qwen35moe.ssm.state_size"].contents()
                        d = unpacked[0]
                        
                        if reorder_linear_required:
                            d0, d1 = d.chunk(2, dim = 0)

                            d1 = rearrange(d1, '(q g p) c -> (g q p) c', p = DH, q = 2).contiguous()
                        
                            d = torch.cat([d0, d1], dim = 0).contiguous()
                        d = d.T.contiguous()
                        # unpacked = [d]
                        self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = d.to(torch.bfloat16)
                        continue
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
            raise ValueError(f"Unsupported weights_type: {weights_type} for Qwen35 model")

        self._export_q4nx_tensors(q4nx_path)




