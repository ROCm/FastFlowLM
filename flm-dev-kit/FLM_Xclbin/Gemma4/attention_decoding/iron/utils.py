# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from dataclasses import dataclass
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils
from aie.utils.trace import MemTilePortEvent, MemTileEvent, PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from aie.dialects import memref
from typing import Dict, Any, List, Optional
from aie.extras import types as T

import os

# Which AIE core compiler the enclosing `make` is driving (see makefiles/kernel.mk).
# Peano needs a tile-allocation workaround that chess does not; gating it on this
# keeps the chess-generated MLIR byte-identical to the pre-Peano design.
ENABLE_CHESSCC = os.environ.get("ENABLE_CHESSCC", "1") != "0"


@dataclass
class tile_with_location:
    tile: tile
    row: int
    col: int

@dataclass
class dma_port:
    tile: tile_with_location
    channel: int

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
            # The kv compute row (row index 1 -> tile row 3) packs its data memory
            # to the top bank (v/y buffers), leaving only interior gaps. Peano's
            # ldscript generator only grants a .data region above the highest
            # buffer, so a top-filling bank-aware layout yields a 0-length data
            # region and the link fails. basic-sequential packs buffers from the
            # bottom, leaving a contiguous free tail for the core's .data/.bss.
            if ENABLE_CHESSCC:
                t = tile(col, row + 2)
            else:
                scheme = "basic-sequential" if row == 1 else None
                t = tile(col, row + 2, allocation_scheme=scheme)
            ComputeTilesRow.append(tile_with_location(t, row + 2, col))

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
