# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from typing import Dict
from .constants import ModelArch, ModelArchNames, QuantType
from abc import ABCMeta

_MODEL_CONFIG_REGISTRY: Dict[str, type['ModelConfig']] = {}

class ModelConfig(metaclass=ABCMeta):
    """
    Model configuration for the model.
    """
    # From external config
    name: str
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    hidden_size: int
    vocab_size: int
    head_dim: int
    q_heads_padding: int
    glu_slice: int
    has_qk_norm: bool
    has_attn_bias: bool
    kv_head_per_cu: int


    num_experts: int
    num_active_experts: int
    num_shared_experts: int

    q4nx_row_block_size: int
    q4nx_col_block_size: int
    q4nx_group_size: int

    # self computed values
    DQ: int
    DK: int
    DV: int

    # convenience values
    D: int
    NH: int
    DH: int

    PADDED_DQ: int
    NUM_Q_HEADS_PADDED: int
    # W_BLOCK_SIZE is size of bytes / sizeof(bf16)
    W_BLOCK_SIZE: int

    x_slice_size: int = 256
   
    lq: int = 16
    lk: int = 16

    num_proj_column = 4
    
    rms_residual_link_with: str = "rms_residual.o"
    rope_link_with: str = "rope.o"
    glu_link_with: str = "glu.o"
    attn_qk_link_with: str = "attn_qk.o"
    attn_kv_link_with: str = "attn_kv.o"
    gate_link_with: str = "gate.o"
    @staticmethod
    def _q4_1_size(elems: int) -> int:
        return int(elems * 5 // 8)
    @staticmethod
    def _q4_k_size(elems: int) -> int:
        # 4 bits of quant per weight, a uint8 scale and a uint8 min per group of
        # 32, and a bf16 S / bf16 M per super-block of 256: 4.625 bits per weight.
        return int(elems * 4.625 // 8)
    @staticmethod
    def _q8_0_size(elems: int) -> int:
        return int(elems * 8.5 // 8)

    def __init__(
        self,
        name: str,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        hidden_size: int,
        glu_slice: int,
        vocab_size: int,
        head_dim: int,
        kv_head_per_cu: int,
        num_delta_net_cts: int,
        num_attn_cus: int,
        PROJ_QUANT_TYPE: QuantType,
        linear_num_heads: int = None,
        linear_num_qk_heads: int = None,
        linear_qk_head_dim: int = None,
        linear_v_head_dim: int = None,
        linear_conv_kernel_size: int = None,
        num_experts: int = 256,
        num_active_experts: int  = 8,
        num_shared_experts: int  = 1,
        rope_partial_factor: float = 1.0,
        q_heads_padding: int = 4,
        has_qk_norm: bool = False,
        has_attn_bias: bool = False,
        q4nx_row_block_size: int = 32,
        q4nx_col_block_size: int = 256,
        q4nx_group_size: int = 32,
        rms_residual_link_with: str = "rms_residual.o",
        rope_link_with: str = "rope.o",
        glu_link_with: str = "glu.o",
        attn_qk_link_with: str = "attn_qk.o",
        attn_kv_link_with: str = "attn_kv.o"
    ):
        self.name = name
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.glu_slice = glu_slice
        self.vocab_size = vocab_size
        self.head_dim = head_dim
        self.has_qk_norm = has_qk_norm
        self.has_attn_bias = has_attn_bias
        self.q_heads_padding = q_heads_padding
        self.kv_head_per_cu = kv_head_per_cu

        self.q4nx_row_block_size = q4nx_row_block_size
        self.q4nx_col_block_size = q4nx_col_block_size
        self.q4nx_group_size = q4nx_group_size

        self.DQ = num_heads * head_dim
        self.DK = num_kv_heads * head_dim
        self.DV = num_kv_heads * head_dim

        self.linear_num_heads = linear_num_heads
        self.linear_num_heads_padded = (linear_num_heads + 31) // 32 * 32
        self.linear_num_qk_heads = linear_num_qk_heads
        self.linear_qk_head_dim = linear_qk_head_dim
        self.linear_v_head_dim = linear_v_head_dim
        self.linear_conv_kernel_size = linear_conv_kernel_size

        self.linear_DQ = linear_num_qk_heads * linear_qk_head_dim
        self.linear_DK = linear_num_qk_heads * linear_qk_head_dim
        self.linear_DV = linear_num_heads * linear_v_head_dim

        self.ROPE_DIM = int(head_dim * rope_partial_factor)

        kv_groups = num_heads // num_kv_heads
        self.PADDED_DQ = ((kv_groups + self.q_heads_padding - 1) // self.q_heads_padding * self.q_heads_padding) * num_kv_heads * head_dim
        self.NUM_Q_HEADS_PADDED = self.PADDED_DQ // head_dim
        self.ATTN_ROUNDS = self.num_kv_heads // self.kv_head_per_cu
        self.Q_HEAD_PER_ROUND = self.num_heads // self.ATTN_ROUNDS
        self.Q_HEAD_PER_ROUND_PADDED = self.NUM_Q_HEADS_PADDED // self.ATTN_ROUNDS
        if PROJ_QUANT_TYPE == QuantType.Q4_K:
            self.W_BLOCK_SIZE = ModelConfig._q4_k_size(self.q4nx_row_block_size * self.q4nx_col_block_size) // 2
        elif PROJ_QUANT_TYPE == QuantType.Q4_1:
            self.W_BLOCK_SIZE = ModelConfig._q4_1_size(self.q4nx_row_block_size * self.q4nx_col_block_size) // 2
        elif PROJ_QUANT_TYPE == QuantType.Q8_0:
            self.W_BLOCK_SIZE = ModelConfig._q8_0_size(self.q4nx_row_block_size * self.q4nx_col_block_size) // 2

        self.num_delta_net_cts = num_delta_net_cts
        self.num_attn_cus = num_attn_cus

        self.D = hidden_size
        self.NH = num_heads
        self.DH = head_dim

        self.rms_residual_link_with = rms_residual_link_with
        self.rope_link_with = rope_link_with
        self.glu_link_with = glu_link_with
        self.attn_qk_link_with = attn_qk_link_with
        self.attn_kv_link_with = attn_kv_link_with

        num_experts = num_experts
        num_active_experts = num_active_experts
        num_shared_experts = num_shared_experts

        # do check
        # all input must be multiple of x_slice_size
        if self.D % self.x_slice_size != 0:
            raise ValueError(f"D must be multiple of {self.x_slice_size}, got {self.D}")
        if self.DQ % self.x_slice_size != 0:
            raise ValueError(f"DQ must be multiple of {self.x_slice_size}, got {self.DQ}")
        if self.intermediate_size % self.x_slice_size != 0:
            raise ValueError(f"intermediate_size must be multiple of {self.x_slice_size}, got {self.intermediate_size}")

    def __init_subclass__(cls, model_arch: ModelArch, **kwargs):
        super().__init_subclass__(**kwargs)
        _MODEL_CONFIG_REGISTRY[ModelArchNames[model_arch]] = cls
    
    def __str__(self):
        return f"ModelConfig(name={self.name}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, intermediate_size={self.intermediate_size}, hidden_size={self.hidden_size}, vocab_size={self.vocab_size}, head_dim={self.head_dim}, q4nx_row_block_size={self.q4nx_row_block_size}, q4nx_col_block_size={self.q4nx_col_block_size}, q4nx_group_size={self.q4nx_group_size})"

def get_model_config(ModelName: str) -> ModelConfig:
    if ModelName not in _MODEL_CONFIG_REGISTRY:
        raise ValueError(f"Model {ModelName} not found")
    return _MODEL_CONFIG_REGISTRY[ModelName]()