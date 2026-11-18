# Copyright (c) 2023 Georgi Gerganov
# SPDX-License-Identifier: MIT
# Modifications Copyright(C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
from gguf.constants import GGMLQuantizationType
from gguf import dequantize, quantize
from gguf.constants import GGML_QUANT_SIZES

from typing import Tuple

from dataclasses import dataclass
from mpmath.libmp import int_types
import numpy as np
import torch

def _to_bf16_up(x: np.ndarray) -> np.ndarray:
    """Round the magnitude of a non-negative float32 UP to a BF16 value."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = np.where(u & np.uint32(0xFFFF), np.uint32(0x10000), np.uint32(0))
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32)


def _bf16_next_up(x: np.ndarray, n: int) -> np.ndarray:
    """The n-th BF16 value above a BF16-representable, non-negative x (n=0 is x)."""
    if n == 0:
        return np.asarray(x, dtype=np.float32)
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return (u + np.uint32(n) * np.uint32(0x10000)).view(np.float32)


def _refit_one_side(t: np.ndarray, search: int = 3,
                    cap: float = 255.0) -> Tuple[np.ndarray, np.ndarray]:
    """Fit one metadata side of a super-block: exact per-group t_j -> (BF16 P', uint8 p'_j).

    Direct port of the pseudocode in quant.md, "Re-fitting (uint6, FP16) into
    (uint8, BF16)".  The kernel only ever uses the product t_j = P * p_j, so the
    caller hands over exactly that -- Q4_K's (FP16 P, uint6 p_j) collapsed into a
    single exact float32 -- and this preserves it directly.

    The two extra bits of p'_j are spent *absorbing* P's rounding error rather
    than as a plain x4 (which would be a no-op, since P/4 has P's significand):
    P' is fixed first, rounded away from zero so p'_j can never overflow the cap,
    then each p'_j is re-derived against the rounded P'.  Since t_j / P' runs up
    to 255, the quotient can absorb up to 255 * 2^-8 ~ 1 integer step, which the
    uint8 grid -- 4x finer than the uint6 grid it came from -- can represent.

    `search` also tries that many BF16 values above the base P' and keeps
    whichever minimizes sum_j (P' p'_j - t_j)^2.  A slightly larger P' often
    aligns better with several t_j at once than the smallest admissible one, and
    it is what removes the re-fit's sensitivity to how spread the t_j are; 3 is
    the knee of the measured sweep.

    Parameters
    ----------
    t : np.ndarray
        Effective per-group scale (or min), shape (..., groups_per_super_block),
        exact in float32.
    search : int
        Number of extra BF16 candidates above the base P' to score.
    cap : float
        Largest representable p'_j (255 for uint8).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        P' (BF16-representable float32, shape t.shape[:-1]) and p'_j (float32
        integers in [0, cap], shape of t).  Effective value: P' p'_j ~ t_j.
    """
    t = np.ascontiguousarray(t, dtype=np.float32)
    tmax = np.abs(t).max(axis=-1)
    # p'_j is unsigned, so the sign rides on P'; take it from the largest entry
    # (Q4_K's d/dmin are non-negative, so every t_j in a group shares one sign).
    lead = np.take_along_axis(t, np.argmax(np.abs(t), axis=-1)[..., None], axis=-1)[..., 0]
    sigma = np.where(lead < 0, np.float32(-1.0), np.float32(1.0))
    t = sigma[..., None] * t
    live = tmax > 0                                              # dead super-block -> all zero
    base = _to_bf16_up(tmax / np.float32(cap))

    best_P = np.zeros_like(base)
    best_p = np.zeros_like(t)
    best_e = np.full(base.shape, np.inf, dtype=np.float64)
    for c in range(search + 1):
        Pc = np.where(live, _bf16_next_up(base, c), np.float32(0.0)).astype(np.float32)
        inv = np.where(live, 1.0 / np.where(live, Pc, np.float32(1.0)), np.float32(0.0)).astype(np.float32)
        pc = np.clip(np.rint(t * inv[..., None]), 0.0, cap).astype(np.float32)
        err = np.sum((Pc[..., None] * pc - t).astype(np.float64) ** 2, axis=-1)
        take = err < best_e
        best_e = np.where(take, err, best_e)
        best_P = np.where(take, Pc, best_P)
        best_p = np.where(take[..., None], pc, best_p)

    return (sigma * best_P).astype(np.float32), best_p


class GGUFTensor:
    name: str
    shape: Tuple[int, ...]
    data: np.ndarray
    tensor_type: GGMLQuantizationType

    # Q4_K shares one uint6 scale/min per 32 weights, 8 of them per 256-weight super-block.
    Q4_K_GROUP_SIZE = 32

    def __init__(self, name: str, shape: Tuple[int, ...], data: np.ndarray, tensor_type: GGMLQuantizationType):
        self.name = name
        self.shape = shape
        self.data = data
        self.tensor_type = tensor_type

    @staticmethod
    def unpack_q4_0(tensor: np.ndarray, columns: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_0]
        data = tensor.view(np.uint8)
        shape = data.shape
        n_blocks = data.size // type_size
        blocks = data.reshape((n_blocks, type_size))
        
        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.int8) - np.int8(8)

        d = torch.from_numpy(d)
        m = torch.zeros_like(d)
        qs = torch.from_numpy(qs)
        d = d.view(-1, columns // block_size)
        m = m.view(-1, columns // block_size)
        qs = qs.view(-1, columns)

        return d, m, qs

    @staticmethod
    def unpack_q4_1(tensor: np.ndarray, columns: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_1]
        data = tensor.view(np.uint8)
        shape = data.shape
        n_blocks = data.size // type_size
        blocks = data.reshape((n_blocks, type_size))
        
        d, rest = np.hsplit(blocks, [2])
        m, qs = np.hsplit(rest, [2])

        d = d.view(np.float16).astype(np.float32)
        m = m.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.float32)

        d = torch.from_numpy(d).contiguous()
        m = torch.from_numpy(m).contiguous()
        qs = torch.from_numpy(qs).contiguous()
        assert columns % block_size == 0, "Columns must be divisible by block size"

        d = d.view(-1, int(columns // block_size))
        m = m.view(-1, int(columns // block_size))
        qs = qs.view(-1, int(columns))

        return d, m, qs
    
    @staticmethod
    def unpack_q4_k(tensor: np.ndarray, columns: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack GGML Q4_K into per-group effective scale / min plus the raw uint4 quants.

        A Q4_K block is 144 bytes covering 256 weights = 8 groups of 32:

            2 B   d      FP16 super-block scale S
            2 B   dmin   FP16 super-block min   M
            12 B  scales 8 uint6 s_j + 8 uint6 m_j, bit-packed (get_scale_min_k4)
            128 B qs     256 uint4 q, nibble-packed

        and dequantizes as w^j_i = (S s_j) q^j_i - (M m_j): the min is an unsigned
        magnitude that is *subtracted*, unlike Q4_1's signed +m.

        What comes back here is the *factored-out* form t_j = S s_j and u_j = M m_j,
        one pair per group of 32, held exactly in float32 (11-bit FP16 significand
        times a 6-bit integer needs 17 bits, so the product is exact).  Nothing is
        re-quantized and no super-block structure survives, which means the result
        has the same shape and the same 32-column granularity as unpack_q4_1's
        (d, m, qw) and can go through the model-specific row/column reorders
        untouched.  The (BF16 S', uint8 s'_j) re-fit of quant.md happens later, in
        _pack_q4k, over whichever 8 groups actually end up sharing a super-block
        after those reorders.

        Parameters
        ----------
        tensor : np.ndarray
            Raw Q4_K tensor bytes.
        columns : int
            Row length K; must be a multiple of 256.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            t (rows, K/32), u (rows, K/32), q (rows, K), all float32.  u is the
            unsigned magnitude that is *subtracted*: w = t_j q - u_j.
        """
        block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_K]
        assert columns % block_size == 0, "Columns must be divisible by the Q4_K super-block size"
        data = tensor.view(np.uint8)
        n_blocks = data.size // type_size
        blocks = data.reshape((n_blocks, type_size))

        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, qs = np.hsplit(rest, [12])

        S = d.view(np.float16).astype(np.float32).reshape(n_blocks)
        M = dmin.view(np.float16).astype(np.float32).reshape(n_blocks)

        # get_scale_min_k4: groups 0..3 take a plain 6-bit field, groups 4..7 take
        # a low nibble from scales[j+4] plus the two high bits of scales[j-4].
        lo = scales[:, 0:4]      # j = 0..3
        hi = scales[:, 4:8]      # j = 4..7 for the min side, high bits for j = 4..7
        top = scales[:, 8:12]
        s6 = np.concatenate([lo & np.uint8(0x3F),
                             (top & np.uint8(0x0F)) | ((lo >> np.uint8(6)) << np.uint8(4))], axis=1)
        m6 = np.concatenate([hi & np.uint8(0x3F),
                             (top >> np.uint8(4)) | ((hi >> np.uint8(6)) << np.uint8(4))], axis=1)
        s6 = s6.astype(np.float32)
        m6 = m6.astype(np.float32)

        # qs holds four runs of 32 bytes; within a run the low nibbles are the
        # first group of 32 and the high nibbles the next one.
        qb = qs.reshape((n_blocks, 4, 32))
        q = np.stack([qb & np.uint8(0x0F), qb >> np.uint8(4)], axis=2)  # (nb, 4, 2, 32)
        q = q.reshape((n_blocks, block_size)).astype(np.float32)

        # Exact in float32: 11-bit significand x 6-bit integer fits in 24 bits.
        t = (S[:, None] * s6).astype(np.float32)
        u = (M[:, None] * m6).astype(np.float32)

        n_groups = int(columns // GGUFTensor.Q4_K_GROUP_SIZE)
        t = torch.from_numpy(t).contiguous().view(-1, n_groups)
        u = torch.from_numpy(u).contiguous().view(-1, n_groups)
        q = torch.from_numpy(q).contiguous().view(-1, int(columns))

        return t, u, q

    @staticmethod
    def repack_q4_k_as_q4_1(tensor: np.ndarray, columns: int) -> np.ndarray:
        """Exact GGML Q4_K -> GGML Q4_1 repack, for engines whose kernels only read Q4_1.

        A Q4_K group of 32 dequantizes as w = t_j q - u_j (t_j = S s_j, u_j = M m_j); Q4_1 stores
        w = d q + m with one FP16 (d, m) per group of 32.  So d = t_j, m = -u_j and the 4-bit q
        are kept as they are: the only loss is rounding t_j and u_j to FP16 (about 7e-4 relative L2).
        Nothing is re-fitted, which keeps a tuned (imatrix) Q4_K as it was.

        Returns Q4_1 bytes shaped (rows, columns / 32 * 20), the layout GGUFReader gives for Q4_1.
        """
        t, u, q = GGUFTensor.unpack_q4_k(tensor, columns)
        rows = t.shape[0]
        d = t.numpy().astype(np.float16)
        m = (-u.numpy()).astype(np.float16)
        groups = q.numpy().astype(np.uint8).reshape(rows, -1, 32)
        # Q4_1 nibble order: byte i holds q[i] (low) and q[i + 16] (high)
        nibbles = groups[:, :, :16] | (groups[:, :, 16:] << 4)
        blocks = np.concatenate([d.view(np.uint8).reshape(rows, -1, 2),
                                 m.view(np.uint8).reshape(rows, -1, 2), nibbles], axis=2)
        return np.ascontiguousarray(blocks.reshape(rows, -1))

    @staticmethod
    def unpack_q8_0(tensors:np.ndarray, columns:int):
        """Split GGML Q8_0 data into scales and quantized values

        Parameters
        ----------
        tensors : np.ndarray
            Q8_0 tensor data
            Format per block (34 bytes):
                - 2 bytes: scale (float16)
                - 32 bytes: 32 x 8-bit quantized values

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            scales, data

        Raises
        ------
        ValueError
            _description_
        """
        byte_per_blocks = 34
        assert tensors.dtype == np.uint8 or tensors.dtype == np.int8, "Input must be np.uint8 or np.int8"
        original_shape = tensors.shape
        assert original_shape[-1] % byte_per_blocks == 0, "The last dimension must be a multiple of 34"
        
        
        blocks = tensors.reshape(*original_shape[:-1], -1, byte_per_blocks)
        
        # Use view() to reinterpret the bytes as float16 bits, not convert the values
        scales = blocks[..., 0:2].view(np.float16) 
        
        # Use view() to reinterpret bytes as signed int8
        data = blocks[..., 2:].view(np.int8)
        
        Q8_block_size = 32
        # Reshape to (rows, cols_per_type) using columns, matching unpack_q4_1 output shape
        scales = np.ascontiguousarray(scales).reshape(-1, columns // Q8_block_size)
        data = np.ascontiguousarray(data).reshape(-1, columns)
        
        return torch.from_numpy(scales.copy()), torch.from_numpy(scales.copy()), torch.from_numpy(data.copy())
        
        
    
    
    @staticmethod
    def e8m0_to_fp32_half(x: np.ndarray) -> np.ndarray:
        bits = np.where(x < 2, np.uint32(0x00200000) << np.uint32(x), np.uint32(x - 1) << np.uint32(23))
        return bits.view(np.float32)

    @staticmethod
    def reverse_transform_nibble_layout( tensor: torch.Tensor) -> torch.Tensor:
        """Reverses the custom nibble layout transformation."""
        assert tensor.dtype == torch.uint8
        assert tensor.shape[-1] == 16

        # 1. Reverse the final nibble swap
        t_lo = tensor & 0x0F
        t_hi = tensor & 0xF0
        interleaved = (t_lo << 4) | (t_hi >> 4)

        # 2. De-interleave the nibbles from abababab... back to aaaa...bbbb...
        # The high nibbles of 'interleaved' contain the nibbles for the first half (blk_a)
        nibbles_a_parts = interleaved & 0xF0
        # The low nibbles of 'interleaved' contain the nibbles for the second half (blk_b)
        nibbles_b_parts = interleaved & 0x0F

        # Reconstruct blk_a by packing the high nibbles back together
        # Pair up nibbles: (1st high nibble) | (2nd high nibble >> 4)
        blk_a = nibbles_a_parts[..., 0::2] | (nibbles_a_parts[..., 1::2] >> 4)

        # Reconstruct blk_b by packing the low nibbles back together
        # Pair up nibbles: (1st low nibble << 4) | (2nd low nibble)
        blk_b = (nibbles_b_parts[..., 0::2] << 4) | nibbles_b_parts[..., 1::2]

        deinterleaved = torch.cat((blk_a, blk_b), dim=-1)

        # 3. Reverse the initial nibble swap
        t_lo = deinterleaved & 0x0F
        t_hi = deinterleaved & 0xF0
        original_tensor = (t_lo << 4) | (t_hi >> 4)

        return original_tensor       
     
    @staticmethod
    def split_ggml_mxfpx_to_scale_blocks(structured_data: np.ndarray):
        """Split GGML MXFP4 data into scales and data blocks

        Format per block (17 bytes):
            - 1 byte: scale (uint8)
            - 16 bytes: 32 x 4-bit float values (2 exponent bits + 1 mantissa bit each)
        """
        
        assert (structured_data.dtype == np.uint8 or structured_data.dtype == np.int8), "Input must be np.uint8 or np.int8"

        original_shape = structured_data.shape
        assert original_shape[-1] % 17 == 0, "The last dimension must be a multiple of 17"
        
        # Reshape the last dimension into blocks of 17 bytes
        blocks = structured_data.reshape(*original_shape[:-1], -1, 17)
        
        # Extract scales (first byte of each block)
        scales = blocks[..., 0].astype(np.uint8)
        
        # Extract data (remaining 16 bytes, keep as uint8 for 4-bit unpacking)
        data = GGUFTensor.reverse_transform_nibble_layout( torch.from_numpy( blocks[..., 1:].astype(np.uint8))).numpy()     
        return scales, data
    
    @staticmethod
    def unpack_mxfp4(tensor: np.ndarray, columns: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, data = GGUFTensor.split_ggml_mxfpx_to_scale_blocks(tensor)
        return torch.from_numpy(scale), torch.from_numpy(data)

    
    def dequantize(self) -> torch.Tensor:
        w = dequantize(self.data, self.tensor_type)
        w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
        return w

    def get_used_quantization_type(self, default_tensor_type: GGMLQuantizationType) -> GGMLQuantizationType:
        
        if self.tensor_type in [GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16, GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1, GGMLQuantizationType.Q8_0, GGMLQuantizationType.MXFP4, GGMLQuantizationType.Q4_K]:
            return self.tensor_type
        else:
            # For unsupported types, we will dequantize and then quantize to default_tensor_type.
            # Q4_K is a read-only format here -- nothing can encode into it -- so a
            # config asking for it as the fallback target gets Q4_1 instead.
            if default_tensor_type == GGMLQuantizationType.Q4_K:
                return GGMLQuantizationType.Q4_1
            return default_tensor_type

    def unpack(self, default_tensor_type: GGMLQuantizationType) -> np.ndarray:
        if self.tensor_type == GGMLQuantizationType.F32:
            return [torch.Tensor(np.array(self.data.view(np.float32)))]
        elif self.tensor_type == GGMLQuantizationType.F16:
            return [torch.Tensor(np.array(self.data.view(np.float16).astype(np.float32)))]
        elif self.tensor_type == GGMLQuantizationType.BF16:
            return [torch.from_numpy(self.data.copy()).view(torch.bfloat16)]

        elif self.tensor_type == GGMLQuantizationType.Q4_0:
            return self.unpack_q4_0(self.data, self.shape[0])
        elif self.tensor_type == GGMLQuantizationType.Q4_1:
            return self.unpack_q4_1(self.data, self.shape[0])
        elif self.tensor_type == GGMLQuantizationType.Q8_0:
            return self.unpack_q8_0(self.data, self.shape[0])
        elif self.tensor_type == GGMLQuantizationType.MXFP4:
            return self.unpack_mxfp4(self.data, self.shape[0])
        elif self.tensor_type == GGMLQuantizationType.Q4_K:
            return self.unpack_q4_k(self.data, self.shape[0])
        else:
            """
                If the tensor type is not supported, try to dequantize it and then quantize it back to Q4_1
                This is a workaround for the fact that the GGUF format does not support all tensor types
                and we need to convert it to a supported type before converting to Q4NX
            """
            # Q4_K does not come here: it has a direct path above. Stacking a
            # dequantize onto a re-quantize costs 0.240 bits of ENOB *and* 0.375
            # bpw against reading it natively, and damages the tail far more than
            # the mean (see quant.md).  q5/q6 still take this path.
            if default_tensor_type == GGMLQuantizationType.Q4_K:
                default_tensor_type = GGMLQuantizationType.Q4_1  # nothing can encode Q4_K
            if default_tensor_type in (GGMLQuantizationType.BF16, GGMLQuantizationType.F16,
                                       GGMLQuantizationType.F32):
                # An unquantized target only needs the dequantize; ggml's quantize()
                # has no encoder for these, so the branch below would raise.
                return [self.dequantize()]
            try:
                w = dequantize(self.data, self.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                
                w = w.to(torch.float32).numpy()
                data_quantized = quantize(w, default_tensor_type).copy()
                if default_tensor_type == GGMLQuantizationType.Q4_1:
                    d, m, qw = self.unpack_q4_1(data_quantized, self.shape[0])
                elif default_tensor_type == GGMLQuantizationType.Q4_0:
                    d, m, qw = self.unpack_q4_0(data_quantized, self.shape[0])
                elif default_tensor_type == GGMLQuantizationType.Q8_0:
                    d, m, qw = self.unpack_q8_0(data_quantized, self.shape[0])
                else:
                    raise ValueError(f"Unsupported tensor type: {default_tensor_type.name}")
                return d, m, qw
            except Exception as e:
                print(f"Error unpacking {self.tensor_type.name}: {e}")
                return None, None, None