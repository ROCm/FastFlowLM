# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from typing import Dict, Optional, Set, Tuple
from .constants import ModelArch, ModelArchNames, QuantType, QuantBytesPerWeight
from abc import ABCMeta
from functools import lru_cache
from pathlib import Path
import re

_MODEL_CONFIG_REGISTRY: Dict[str, type['ModelConfig']] = {}

# The kernel headers the Makefile compiles against, from the same MODEL_TYPE
# symbol it passes to this script.
_KERNEL_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_QUANT_TYPE_MACROS: Dict[str, QuantType] = {
    "__Q4_1__": QuantType.Q4_1,
    "__Q8_0__": QuantType.Q8_0,
    "__Q4_K__": QuantType.Q4_K,
}


@lru_cache(maxsize=1)
def _kernel_quant_types() -> Tuple[Dict[str, QuantType], Set[str]]:
    """Read QUANT_TYPE out of models/<model>.h, keyed by MODEL_TYPE symbol.

    Resolved the way the preprocessor does it: each header's body sits inside
    `#if MODEL_TYPE == <symbol>`, so the `#define QUANT_TYPE` within that guard
    is what the symbol compiles with.  Returns the explicit choices plus the set
    of every symbol the headers declare -- a declared symbol with no QUANT_TYPE
    of its own inherits all_models.h's Q4_1 fallback.
    """
    explicit: Dict[str, QuantType] = {}
    declared: Set[str] = set()
    if not _KERNEL_MODELS_DIR.is_dir():
        return explicit, declared
    for header in sorted(_KERNEL_MODELS_DIR.glob("*.h")):
        if header.name == "all_models.h":
            continue
        guard: Optional[str] = None
        for line in header.read_text().splitlines():
            match = re.match(r"\s*#if\s+MODEL_TYPE\s*==\s*(\w+)", line)
            if match:
                guard = match.group(1)
                declared.add(guard)
                continue
            if re.match(r"\s*#endif", line):
                guard = None
                continue
            match = re.match(r"\s*#define\s+QUANT_TYPE\s+(__\w+__)", line)
            if match and guard and match.group(1) in _QUANT_TYPE_MACROS:
                explicit[guard] = _QUANT_TYPE_MACROS[match.group(1)]
    return explicit, declared

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

    q4nx_row_block_size: int
    q4nx_col_block_size: int
    q4nx_group_size: int
    # Projection weight format. Must match QUANT_TYPE in kernels' models/<model>.h
    # -- it sets the packed block size, i.e. every weight DMA length.
    quant_type: QuantType

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

    rms_residual_link_with: str = "rms_residual.o"
    rope_link_with: str = "rope.o"
    glu_link_with: str = "glu.o"
    attn_qk_link_with: str = "attn_qk.o"
    attn_kv_link_with: str = "attn_kv.o"
    
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
        q_heads_padding: int = 4,
        has_qk_norm: bool = False,
        has_attn_bias: bool = False,
        q4nx_row_block_size: int = 32,
        q4nx_col_block_size: int = 256,
        q4nx_group_size: int = 32,
        quant_type: QuantType = QuantType.Q4_1,
        rms_residual_link_with: str = "rms_residual.o",
        rope_link_with: str = "rope.o",
        glu_link_with: str = "glu.o",
        attn_qk_link_with: str = "attn_qk.o",
        attn_kv_link_with: str = "attn_kv.o",
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

        self.q4nx_row_block_size = q4nx_row_block_size
        self.q4nx_col_block_size = q4nx_col_block_size
        self.q4nx_group_size = q4nx_group_size
        self.quant_type = quant_type

        self.DQ = num_heads * head_dim
        self.DK = num_kv_heads * head_dim
        self.DV = num_kv_heads * head_dim

        kv_groups = num_heads // num_kv_heads
        # Same rounding the RoPE tile does (iron/RoPE.py) and the same
        # GQA_SEGMENT_SIZE the kernels derive from ATTN_IMPL
        # (kernels/model_spec.h). Hardcoding 4 here silently disagreed with a
        # config that asked for 8.
        padding = self.q_heads_padding
        self.PADDED_DQ = ((kv_groups + padding - 1) // padding * padding) * num_kv_heads * head_dim
        self.NUM_Q_HEADS_PADDED = self.PADDED_DQ // head_dim
        # One packed block in units of sizeof(bf16), because the weight streams
        # are typed bf16 all the way from the shim. Derived from the format so it
        # follows QUANT_TYPE; the kernel side pins the same numbers with
        # static_asserts on the block structs (kernels/model_spec.h).
        bpw_num, bpw_den = QuantBytesPerWeight[quant_type]
        block_bytes = q4nx_row_block_size * q4nx_col_block_size * bpw_num // bpw_den
        if q4nx_row_block_size * q4nx_col_block_size * bpw_num % (bpw_den * 2) != 0:
            raise ValueError(
                f"{quant_type.name} block of {q4nx_row_block_size}x{q4nx_col_block_size} "
                f"is not a whole number of bf16 words"
            )
        self.W_BLOCK_SIZE = block_bytes // 2

        self.D = hidden_size
        self.NH = num_heads
        self.DH = head_dim

        self.rms_residual_link_with = rms_residual_link_with
        self.rope_link_with = rope_link_with
        self.glu_link_with = glu_link_with
        self.attn_qk_link_with = attn_qk_link_with
        self.attn_kv_link_with = attn_kv_link_with

        # do check
        # all input must be multiple of x_slice_size
        if self.D % self.x_slice_size != 0:
            raise ValueError(f"D must be multiple of {self.x_slice_size}, got {self.D}")
        if self.DQ % self.x_slice_size != 0:
            raise ValueError(f"DQ must be multiple of {self.x_slice_size}, got {self.DQ}")
        if self.intermediate_size % self.x_slice_size != 0:
            raise ValueError(f"intermediate_size must be multiple of {self.x_slice_size}, got {self.intermediate_size}")

        self._check_quant_type_matches_kernel()

    def __init_subclass__(cls, model_arch: ModelArch, **kwargs):
        super().__init_subclass__(**kwargs)
        # Also the MODEL_TYPE the Makefile hands the kernels, so the config can
        # find the header it has to agree with.
        cls.model_type = ModelArchNames[model_arch]
        _MODEL_CONFIG_REGISTRY[ModelArchNames[model_arch]] = cls

    def _check_quant_type_matches_kernel(self) -> None:
        """Fail the build if this config and models/<model>.h disagree.

        The format is declared twice -- here for the DMA lengths, and in the
        kernel header for the block struct -- so it can drift, and a drift is a
        wrong weight DMA length rather than an error.  Compare them instead.
        """
        model_type = getattr(type(self), "model_type", None)
        explicit, declared = _kernel_quant_types()
        if model_type is None or model_type not in declared:
            return  # no kernel header declares it; nothing to compare against
        kernel_quant_type = explicit.get(model_type, QuantType.Q4_1)  # all_models.h fallback
        if kernel_quant_type is not self.quant_type:
            raise ValueError(
                f"{self.name}: quant_type is {self.quant_type.name} but "
                f"models/<model>.h compiles MODEL_TYPE={model_type} as "
                f"__{kernel_quant_type.name}__. They set the same packed block "
                f"size and must agree; fix whichever is stale."
            )
    
    def __str__(self):
        return f"ModelConfig(name={self.name}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, intermediate_size={self.intermediate_size}, hidden_size={self.hidden_size}, vocab_size={self.vocab_size}, head_dim={self.head_dim}, q4nx_row_block_size={self.q4nx_row_block_size}, q4nx_col_block_size={self.q4nx_col_block_size}, q4nx_group_size={self.q4nx_group_size}, quant_type={self.quant_type.name}, W_BLOCK_SIZE={self.W_BLOCK_SIZE})"

def get_model_config(ModelName: str) -> ModelConfig:
    if ModelName not in _MODEL_CONFIG_REGISTRY:
        raise ValueError(f"Model {ModelName} not found")
    return _MODEL_CONFIG_REGISTRY[ModelName]()