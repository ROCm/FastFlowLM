# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from typing import Dict
from .constants import ModelArch, ModelArchNames, QuantType
from abc import ABCMeta

_MODEL_CONFIG_REGISTRY: Dict[str, type['ModelConfig']] = {}

class ModelConfig(metaclass=ABCMeta):
    """
    Model configuration for the model.

    Delta-net only. The full-attention shape parameters (head_dim, the GQA
    padding, RoPE, the attn/rope link_with names) belong to the sibling project
    ../../gated_attention_verify and are not carried here.
    """
    # From external config
    name: str
    intermediate_size: int
    hidden_size: int
    vocab_size: int
    glu_slice: int

    q4nx_row_block_size: int
    q4nx_col_block_size: int
    q4nx_group_size: int

    # convenience values
    D: int

    # W_BLOCK_SIZE is size of bytes / sizeof(bf16)
    W_BLOCK_SIZE: int
    AB_W_BLOCK_SIZE: int

    x_slice_size: int = 256

    rms_residual_link_with: str = "rms_residual.o"
    glu_link_with: str = "glu.o"
    gate_link_with: str = "gate.o"
    @staticmethod
    def _q4_1_size(elems: int) -> int:
        return int(elems * 4.625 // 8)
    @staticmethod
    def _q8_0_size(elems: int) -> int:
        return int(elems * 8.5 // 8)
    @staticmethod
    def _q4_k_size(elems: int) -> int:
        # q4k_block_t: u8 scales + u8 mins per group of 32, 4-bit quants, bf16 S/M per 256
        return elems // 32 * 2 + elems // 2 + elems // 256 * 4

    def __init__(
        self,
        name: str,
        intermediate_size: int,
        hidden_size: int,
        glu_slice: int,
        vocab_size: int,
        num_delta_net_cts: int,
        PROJ_QUANT_TYPE: QuantType,
        AB_PROJ_QUANT_TYPE: QuantType,
        linear_num_heads: int = None,
        linear_num_qk_heads: int = None,
        linear_qk_head_dim: int = None,
        linear_v_head_dim: int = None,
        linear_conv_kernel_size: int = None,
        q4nx_row_block_size: int = 32,
        q4nx_col_block_size: int = 256,
        q4nx_group_size: int = 32,
        rms_residual_link_with: str = "rms_residual.o",
        glu_link_with: str = "glu.o",
    ):
        self.name = name
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.glu_slice = glu_slice
        self.vocab_size = vocab_size

        self.q4nx_row_block_size = q4nx_row_block_size
        self.q4nx_col_block_size = q4nx_col_block_size
        self.q4nx_group_size = q4nx_group_size

        self.linear_num_heads = linear_num_heads
        self.linear_num_heads_padded = (linear_num_heads + 31) // 32 * 32
        self.linear_num_qk_heads = linear_num_qk_heads
        self.linear_qk_head_dim = linear_qk_head_dim
        self.linear_v_head_dim = linear_v_head_dim
        self.linear_conv_kernel_size = linear_conv_kernel_size

        self.linear_DQ = linear_num_qk_heads * linear_qk_head_dim
        self.linear_DK = linear_num_qk_heads * linear_qk_head_dim
        self.linear_DV = linear_num_heads * linear_v_head_dim

        if PROJ_QUANT_TYPE == QuantType.Q4_1:
            self.W_BLOCK_SIZE = ModelConfig._q4_1_size(self.q4nx_row_block_size * self.q4nx_col_block_size) // 2
        elif PROJ_QUANT_TYPE == QuantType.Q8_0:
            self.W_BLOCK_SIZE = ModelConfig._q8_0_size(self.q4nx_row_block_size * self.q4nx_col_block_size) // 2
        elif PROJ_QUANT_TYPE == QuantType.Q4_K:
            self.W_BLOCK_SIZE = ModelConfig._q4_k_size(self.q4nx_row_block_size * self.q4nx_col_block_size) // 2

        if AB_PROJ_QUANT_TYPE == QuantType.Q4_1:
            self.AB_W_BLOCK_SIZE = ModelConfig._q4_1_size(self.q4nx_row_block_size * self.q4nx_col_block_size) // 2
        elif AB_PROJ_QUANT_TYPE == QuantType.Q8_0:
            self.AB_W_BLOCK_SIZE = ModelConfig._q8_0_size(self.q4nx_row_block_size * self.q4nx_col_block_size) // 2
        elif AB_PROJ_QUANT_TYPE == QuantType.Q4_K:
            self.AB_W_BLOCK_SIZE = ModelConfig._q4_k_size(self.q4nx_row_block_size * self.q4nx_col_block_size) // 2
        
        self.num_delta_net_cts = num_delta_net_cts

        self.D = hidden_size

        self.rms_residual_link_with = rms_residual_link_with
        self.glu_link_with = glu_link_with

        # do check
        # all input must be multiple of x_slice_size
        if self.D % self.x_slice_size != 0:
            raise ValueError(f"D must be multiple of {self.x_slice_size}, got {self.D}")
        if self.linear_DV % self.x_slice_size != 0:
            raise ValueError(f"linear_DV must be multiple of {self.x_slice_size}, got {self.linear_DV}")
        if self.intermediate_size % self.x_slice_size != 0:
            raise ValueError(f"intermediate_size must be multiple of {self.x_slice_size}, got {self.intermediate_size}")

    def __init_subclass__(cls, model_arch: ModelArch, **kwargs):
        super().__init_subclass__(**kwargs)
        _MODEL_CONFIG_REGISTRY[ModelArchNames[model_arch]] = cls
    
    def __str__(self):
        return f"ModelConfig(name={self.name}, linear_num_heads={self.linear_num_heads}, linear_num_qk_heads={self.linear_num_qk_heads}, linear_qk_head_dim={self.linear_qk_head_dim}, linear_v_head_dim={self.linear_v_head_dim}, intermediate_size={self.intermediate_size}, hidden_size={self.hidden_size}, vocab_size={self.vocab_size}, q4nx_row_block_size={self.q4nx_row_block_size}, q4nx_col_block_size={self.q4nx_col_block_size}, q4nx_group_size={self.q4nx_group_size})"

def get_model_config(ModelName: str) -> ModelConfig:
    if ModelName not in _MODEL_CONFIG_REGISTRY:
        raise ValueError(f"Model {ModelName} not found")
    return _MODEL_CONFIG_REGISTRY[ModelName]()