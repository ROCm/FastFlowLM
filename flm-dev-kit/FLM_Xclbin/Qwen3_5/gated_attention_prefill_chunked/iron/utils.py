# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from dataclasses import dataclass
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
try:
    from aie.helpers.dialects.ext.scf import _for as range_
except ImportError:
    from aie.helpers.dialects.scf import _for as range_
from aie.dialects import memref
from typing import Dict, Any, List, Optional
from aie.extras import types as T

@dataclass
class tile_with_location:
    tile: tile
    row: int
    col: int

@dataclass
class dma_port:
    tile: tile_with_location
    channel: int

import inspect
# The mlir-aie API moved between the CI toolchain and newer ones; the design builds with both.
# newer: link_with lives on external_func and dma_bd takes sizes/strides/transfer_len;
# CI:    link_with lives on the core and dma_bd takes dimensions/len.
NEW_LINK_API = "link_with" in inspect.signature(aie.external_func.__init__).parameters
NEW_DMA_BD_API = "transfer_len" in inspect.signature(aie.dma_bd).parameters


def core_link_kwargs(obj_name: str):
    """kwargs for @aie.core that link the kernel object (empty on the newer API, where the
    external_func declaration carries it)."""
    return {} if NEW_LINK_API else {"link_with": obj_name}


class mlir_context:
    def __init__(self):
        self.data_tys = {}
        self.funcs = {}
        self.buffers = {}
        self.parameters = {}
        self.object_fifos = {}
        self.packet_ids = {}

    def data_ty(self, name: str, shape: List[int], dtype: Any):
        d_ty = np.ndarray[shape, dtype]
        self.data_tys[name] = d_ty
        if not hasattr(self, name):
            setattr(self, name, d_ty)
        else:
            raise ValueError(f"Data type {name} already exists")
        return d_ty

    def buffer(self, name: str, tile: tile_with_location, dtype: Any, use_write_rtp: bool = False):
        new_buffer = aie.buffer(tile.tile, dtype, name = name, use_write_rtp = use_write_rtp)
        self.buffers[name] = new_buffer
        if not hasattr(self, name):
            setattr(self, name, new_buffer)
        else:
            raise ValueError(f"Buffer {name} already exists")

        return new_buffer

    def func(self, name: str, inputs: List[Any]):
        if NEW_LINK_API:
            this_func = aie.external_func(name, inputs, link_with=f"{name}.o")
        else:
            this_func = aie.external_func(name, inputs)
        self.funcs[name] = this_func
        if not hasattr(self, name):
            setattr(self, name, this_func)
        else:
            raise ValueError(f"Function {name} already exists")
        return this_func

    def parameter(self, name: str, value: Any):
        self.parameters[name] = value
        if not hasattr(self, name):
            setattr(self, name, value)
        else:
            raise ValueError(f"Parameter {name} already exists")
        return value

    def object_fifo(self, name: str, src_tile: tile_with_location, dst_tile: tile_with_location | List[tile_with_location], depth: int, data_ty: Any, *args, **kwargs):
        if isinstance(dst_tile, tile_with_location):
            fifo = object_fifo(name, src_tile.tile, dst_tile.tile, depth, data_ty, *args, **kwargs)
        else:
            fifo = object_fifo(name, src_tile.tile, [t.tile for t in dst_tile], depth, data_ty, *args, **kwargs)
        self.object_fifos[name] = fifo
        if not hasattr(self, name):
            setattr(self, name, fifo)
        else:
            raise ValueError(f"Object fifo {name} already exists")
        return fifo



def dma_mm2s_ch(ch: int):
    return {
        "source_port": WireBundle.DMA,
        "source_channel": ch
    }

def dma_s2mm_ch(ch: int):
    return {
        "dest_port": WireBundle.DMA,
        "dest_channel": ch
    }

def flow_mm2s_ch(ch: int):
    return {
        "source_bundle": WireBundle.DMA,
        "source_channel": ch
    }

def flow_s2mm_ch(ch: int):
    return {
        "dest_bundle": WireBundle.DMA,
        "dest_channel": ch
    }

def generate_tile_map(total_rows: int, total_cols: int) -> Tuple[List[tile_with_location], List[tile_with_location], List[List[tile_with_location]]]:
    # Tile declarations
    ShimTiles = []
    MemTiles = []
    cores = []
    for col in range(total_cols):
        ShimTiles.append(tile_with_location(tile(col, 0), 0, col))
        MemTiles.append(tile_with_location(tile(col, 1), 0, col))

    for row in range(total_rows - 2):
        ComputeTilesRow = []
        for col in range(total_cols):
            ComputeTilesRow.append(tile_with_location(tile(col, row + 2), row + 2, col))

        cores.append(ComputeTilesRow)

    return ShimTiles, MemTiles, cores

def connect_dma(src: tile_with_location, dst: tile_with_location, src_ch: int, dst_ch: int, pkt_id: int = -1, keep_pkt_header: bool = False):
    if pkt_id < 0:
        aie.flow(source = src.tile, **flow_mm2s_ch(src_ch), dest = dst.tile, **flow_s2mm_ch(dst_ch))
    else:
        aie.packetflow(pkt_id = pkt_id, source = src.tile, **dma_mm2s_ch(src_ch), dest = dst.tile, **dma_s2mm_ch(dst_ch), keep_pkt_header = keep_pkt_header)

def connect_port(src: dma_port, dst: dma_port, pkt_id: int = -1, keep_pkt_header: bool = False):
    if pkt_id < 0:
        aie.flow(source = src.tile, **flow_mm2s_ch(src.channel), dest = dst.tile, **flow_s2mm_ch(dst.channel))
    else:
        aie.packetflow(pkt_id = pkt_id, source = src.tile, **dma_mm2s_ch(src.channel), dest = dst.tile, **dma_s2mm_ch(dst.channel), keep_pkt_header = keep_pkt_header)


def lock_acquire(lock: Any, value: int):
    aie.use_lock(lock, aie.LockAction.AcquireGreaterEqual, value = value)

def lock_release(lock: Any, value: int):
    aie.use_lock(lock, aie.LockAction.Release, value = value)


def dma_bd_compat(buffer, offset=None, len=None, dimensions=None):
    """aie.dma_bd(buffer, offset=, len=, dimensions=[(size, stride), ...]) on either API."""
    if not NEW_DMA_BD_API:
        kw = {}
        if offset is not None: kw["offset"] = offset
        if len is not None: kw["len"] = len
        if dimensions: kw["dimensions"] = dimensions
        return aie.dma_bd(buffer, **kw)
    kw = {}
    if dimensions:
        kw["sizes"] = [d[0] for d in dimensions]
        kw["strides"] = [d[1] for d in dimensions]
    return aie.dma_bd(buffer, offset=offset, transfer_len=len, **kw)
