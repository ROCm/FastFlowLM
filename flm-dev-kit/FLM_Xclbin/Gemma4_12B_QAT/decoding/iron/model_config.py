# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from typing import Dict
from .constants import ModelArch, ModelArchNames
from abc import ABCMeta

_MODEL_CONFIG_REGISTRY: Dict[str, type['ModelConfig']] = {}

def _round_up(n: int, m: int) -> int:
    return (n + m - 1) // m * m

class ModelConfig(metaclass=ABCMeta):
    """
    Model configuration for the model.
    """
    # From external config
    name: str
    num_heads: int
    num_kv_heads: int
    swa_num_kv_heads: int
    intermediate_size: int
    hidden_size: int
    vocab_size: int
    head_dim: int
    # Kept for callers that still pass it; the Q-group padding granularity is now
    # derived from GQA_LANES and the KV heads a round holds.
    q_heads_padding: int
    glu_slice: int
    has_qk_norm: bool
    has_attn_bias: bool
    kv_head_per_round: int
    global_kv_head_per_round: int

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

    SWA_DH: int
    SWA_DQ: int
    SWA_DK: int
    SWA_DV: int

    enable_double_wide_mlp: bool = False

    per_layer_input_dim: int
    PADDED_DQ: int
    NUM_Q_HEADS_PADDED: int
    SWA_NUM_Q_HEADS_PADDED: int
    # W_BLOCK_SIZE is size of bytes / sizeof(bf16)
    W_BLOCK_SIZE: int

    # A compute unit issues one r x s x GQA_LANES mmul per pass, so a round always
    # retires exactly GQA_LANES Q heads no matter how they are grouped. This is the
    # t of aie::mmul<r, s, t> on the kernel side; see kernels/model_spec.h.
    GQA_LANES: int = 8

    x_slice_size: int = 256
    lq: int = 16
    lk: int = 16

    bf16_x_slice_size: int = 256
    bf16_w_row_block_size: int = 16
    bf16_w_col_block_size: int = 256

    rms_residual_link_with: str = "rms_residual.o"
    rope_link_with: str = "rope.o"
    swa_rope_link_with: str = "swa_rope.o"
    glu_link_with: str = "glu.o"
    attn_qk_link_with: str = "attn_qk.o"
    attn_kv_link_with: str = "attn_kv.o"
    swa_attn_qk_link_with: str = "swa_attn_qk.o"
    swa_attn_kv_link_with: str = "swa_attn_kv.o"
    gate_link_with: str = "gate.o"
    
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
        kv_head_per_round: int,
        num_attn_cus: int,
        # models without a separate SWA / global KV split leave these unset
        swa_num_kv_heads: int = 0,
        global_kv_head_per_round: int = 0,
        q_heads_padding: int = 4,
        has_qk_norm: bool = False,
        has_attn_bias: bool = False,
        swa_head_dim: int = 0,
        enable_double_wide_mlp: bool = False,
        per_layer_input_dim: int = 0,
        q4nx_row_block_size: int = 16,
        q4nx_col_block_size: int = 256,
        q4nx_group_size: int = 32,
        bf16_x_slice_size: int = 256,
        bf16_w_row_block_size: int = 16,
        bf16_w_col_block_size: int = 256,
        rms_residual_link_with: str = "rms_residual.o",
        rope_link_with: str = "rope.o",
        glu_link_with: str = "glu.o",
        attn_qk_link_with: str = "attn_qk.o",
        attn_kv_link_with: str = "attn_kv.o",
        swa_rope_link_with: str = "swa_rope.o",
        swa_attn_qk_link_with: str = "swa_attn_qk.o",
        swa_attn_kv_link_with: str = "swa_attn_kv.o"
    ):
        self.name = name
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        # fall back to the global config when the model has a single attention flavor
        self.swa_num_kv_heads = swa_num_kv_heads
        self.global_kv_head_per_round = global_kv_head_per_round
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.glu_slice = glu_slice
        self.vocab_size = vocab_size
        self.head_dim = head_dim
        self.has_qk_norm = has_qk_norm
        self.has_attn_bias = has_attn_bias
        self.q_heads_padding = q_heads_padding
        self.kv_head_per_round = kv_head_per_round
        self.enable_double_wide_mlp = enable_double_wide_mlp

        self.q4nx_row_block_size = q4nx_row_block_size
        self.q4nx_col_block_size = q4nx_col_block_size
        self.q4nx_group_size = q4nx_group_size

        self.bf16_x_slice_size = bf16_x_slice_size
        self.bf16_w_row_block_size = bf16_w_row_block_size
        self.bf16_w_col_block_size = bf16_w_col_block_size

        self.DQ = num_heads * head_dim
        self.DK = num_kv_heads * head_dim
        self.DV = num_kv_heads * head_dim
        self.SWA_DQ = num_heads * swa_head_dim
        self.SWA_DK = swa_num_kv_heads * swa_head_dim
        self.SWA_DV = swa_num_kv_heads * swa_head_dim

        kv_groups = num_heads // num_kv_heads
        swa_kv_groups = num_heads // swa_num_kv_heads

        # The KV heads a CU holds share its GQA_LANES output lanes, so each of them
        # gets GQA_LANES // kv_heads_per_round Q heads per pass. That quotient is the
        # granularity a Q group is padded up to (kernel-side GQA_SEGMENT_SIZE), and a
        # group wider than one segment has to be walked in several passes.
        global_segment = self.GQA_LANES // global_kv_head_per_round
        swa_segment = self.GQA_LANES // kv_head_per_round
        global_groups_padded = _round_up(kv_groups, global_segment)
        swa_groups_padded = _round_up(swa_kv_groups, swa_segment)

        self.NUM_Q_HEADS_PADDED = global_groups_padded * num_kv_heads
        self.SWA_NUM_Q_HEADS_PADDED = swa_groups_padded * swa_num_kv_heads
        self.PADDED_DQ = self.NUM_Q_HEADS_PADDED * head_dim

        # One round is one (KV-head slice, Q pass) pair: global layers consume
        # global_kv_head_per_round KV heads per round, SWA layers consume
        # kv_head_per_round of the (larger) SWA KV head set, and either flavor
        # repeats the slice once per Q pass when its groups exceed a segment.
        self.GLOBAL_Q_PASSES = global_groups_padded // global_segment
        self.SWA_Q_PASSES = swa_groups_padded // swa_segment
        self.ATTN_ROUNDS = max(1, num_kv_heads // global_kv_head_per_round) * self.GLOBAL_Q_PASSES
        self.SWA_ATTN_ROUNDS = max(1, swa_num_kv_heads // kv_head_per_round) * self.SWA_Q_PASSES

        self.Q_HEAD_PER_ROUND = num_heads // self.ATTN_ROUNDS
        self.Q_HEAD_PER_ROUND_PADDED = self.NUM_Q_HEADS_PADDED // self.ATTN_ROUNDS
        self.SWA_Q_HEAD_PER_ROUND = num_heads // self.SWA_ATTN_ROUNDS
        self.SWA_Q_HEAD_PER_ROUND_PADDED = self.SWA_NUM_Q_HEADS_PADDED // self.SWA_ATTN_ROUNDS
        self.W_BLOCK_SIZE = self.q4nx_row_block_size * self.q4nx_col_block_size * 9 // 16 // 2
        self.num_attn_cus = num_attn_cus

        self.D = hidden_size
        self.NH = num_heads
        self.DH = head_dim

        self.SWA_DH = swa_head_dim
        self.SWA_DQ = num_heads * swa_head_dim
        self.SWA_DK = swa_num_kv_heads * swa_head_dim
        self.SWA_DV = swa_num_kv_heads * swa_head_dim

        self.per_layer_input_dim = per_layer_input_dim

        self.rms_residual_link_with = rms_residual_link_with
        self.rope_link_with = rope_link_with
        self.glu_link_with = glu_link_with
        self.attn_qk_link_with = attn_qk_link_with
        self.attn_kv_link_with = attn_kv_link_with
        self.swa_rope_link_with = swa_rope_link_with
        self.swa_attn_qk_link_with = swa_attn_qk_link_with
        self.swa_attn_kv_link_with = swa_attn_kv_link_with

        # do check
        # all input must be multiple of x_slice_size
        if self.D % self.x_slice_size != 0:
            raise ValueError(f"D must be multiple of {self.x_slice_size}, got {self.D}")
        if self.DQ % self.x_slice_size != 0:
            raise ValueError(f"DQ must be multiple of {self.x_slice_size}, got {self.DQ}")
        if self.intermediate_size % self.x_slice_size != 0:
            raise ValueError(f"intermediate_size must be multiple of {self.x_slice_size}, got {self.intermediate_size}")
        # A round must fill the mmul exactly, or the kernel's static_assert on
        # KV_HEADS_PER_CU * Q_HEADS_PER_GROUP_PADDED == GQA_T would not hold either.
        if self.Q_HEAD_PER_ROUND_PADDED != self.GQA_LANES:
            raise ValueError(f"global round covers {self.Q_HEAD_PER_ROUND_PADDED} Q heads, expected {self.GQA_LANES}")
        if self.SWA_Q_HEAD_PER_ROUND_PADDED != self.GQA_LANES:
            raise ValueError(f"SWA round covers {self.SWA_Q_HEAD_PER_ROUND_PADDED} Q heads, expected {self.GQA_LANES}")
        if self.GQA_LANES % global_kv_head_per_round != 0:
            raise ValueError(f"global_kv_head_per_round must divide {self.GQA_LANES}, got {global_kv_head_per_round}")
        if self.GQA_LANES % kv_head_per_round != 0:
            raise ValueError(f"kv_head_per_round must divide {self.GQA_LANES}, got {kv_head_per_round}")

    def __init_subclass__(cls, model_arch: ModelArch, **kwargs):
        super().__init_subclass__(**kwargs)
        _MODEL_CONFIG_REGISTRY[ModelArchNames[model_arch]] = cls
    
    def __str__(self):
        return f"ModelConfig(name={self.name}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, intermediate_size={self.intermediate_size}, hidden_size={self.hidden_size}, vocab_size={self.vocab_size}, head_dim={self.head_dim}, q4nx_row_block_size={self.q4nx_row_block_size}, q4nx_col_block_size={self.q4nx_col_block_size}, q4nx_group_size={self.q4nx_group_size})"

def get_model_config(ModelName: str) -> ModelConfig:
    if ModelName not in _MODEL_CONFIG_REGISTRY:
        raise ValueError(f"Model {ModelName} not found")
    return _MODEL_CONFIG_REGISTRY[ModelName]()