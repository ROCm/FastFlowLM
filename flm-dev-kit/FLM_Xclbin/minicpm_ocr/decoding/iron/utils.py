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
from abc import ABC, abstractmethod
import os

# Which AIE core compiler the enclosing `make` is driving (see makefiles/kernel.mk).
# Peano-only layout changes are gated on this so the chess MLIR stays unchanged.
ENABLE_CHESSCC = os.environ.get("ENABLE_CHESSCC", "1") != "0"

def comment_print(msg: str):
    print("// " + msg)

@dataclass
class tile_with_location:
    tile: tile
    row: int
    col: int

    @property
    def S2MM(self):
        return [dma_port(self, i) for i in range(6)]

    @property
    def MM2S(self):
        return [dma_port(self, i) for i in range(6)]


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
        """
        Docstring for data_ty, unique name for each data type
        
        :param self: Description
        :param name: Description
        :type name: str
        :param shape: Description
        :type shape: List[int]
        :param dtype: Description
        :type dtype: Any
        """
        comment_print(f"Defining shared data type {name} with shape {shape} and dtype {dtype}")
        d_ty = np.ndarray[shape, dtype]
        self.data_tys[name] = d_ty
        if not hasattr(self, name):
            setattr(self, name, d_ty)
        else:
            raise ValueError(f"Data type {name} already exists")
        return d_ty

    def buffer(self, name: str, tile: tile_with_location, dtype: Any, use_write_rtp: bool = False, use_address: int = None):
        comment_print(f"Defining shared buffer {name} at tile ({tile.row}, {tile.col}) with dtype {dtype}, use_write_rtp={use_write_rtp}")
        # this should be unique
        if use_write_rtp == True:
            if name[0:4] != "RTP_":
                name = "RTP_" + name
            
        if not hasattr(self, name):
            if (use_address is not None):
                new_buffer = aie.buffer(tile.tile, dtype, name = name, use_write_rtp = use_write_rtp, address = use_address)
            else:
                new_buffer = aie.buffer(tile.tile, dtype, name = name, use_write_rtp = use_write_rtp)
            self.buffers[name] = new_buffer
            setattr(self, name, new_buffer)
            return new_buffer
        else:
            comment_print(f"Buffer {name} already exists, returning existing buffer")
            return self.buffers[name]


    def func(self, name: str, inputs: List[Any]):
        # this should be unique
        if not hasattr(self, name):
            this_func = aie.external_func(name, inputs)
            self.funcs[name] = this_func
            setattr(self, name, this_func)
            return this_func
        else:
            comment_print(f"Function {name} already exists, returning existing function")
            return self.funcs[name]

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

class ShimTile:
    located_tile: tile_with_location
    st: tile
    st_row: int
    st_col: int

    def __init__(self, tile: tile_with_location):
        self.located_tile = tile
        self.st = tile.tile
        self.st_row = tile.row
        self.st_col = tile.col
    
    @property
    def S2MM(self):
        return [dma_port(self.located_tile, i) for i in range(2)]
  
    @property
    def MM2S(self):
        return [dma_port(self.located_tile, i) for i in range(2)]
    
def generate_tile_map(total_rows: int, total_cols: int) -> Tuple[List[tile_with_location], List[tile_with_location], List[List[tile_with_location]]]:
    # Tile declarations
    ShimTiles = []
    MemTiles = []
    cores = []
    for col in range(total_cols):
        ShimTiles.append(ShimTile(tile_with_location(tile(col, 0), 0, col)))
        MemTiles.append(tile_with_location(tile(col, 1), 1, col))

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
        aie.flow(source = src.tile.tile, **flow_mm2s_ch(src.channel), dest = dst.tile.tile, **flow_s2mm_ch(dst.channel))
    else:
        comment_print(f"Connecting port with pkt_id {pkt_id} from tile ({src.tile.row}, {src.tile.col}) channel {src.channel} to tile ({dst.tile.row}, {dst.tile.col}) channel {dst.channel}")
        aie.packetflow(pkt_id = pkt_id, source = src.tile.tile, **dma_mm2s_ch(src.channel), dest = dst.tile.tile, **dma_s2mm_ch(dst.channel), keep_pkt_header = keep_pkt_header)


def lock_acquire(lock: Any, value: int):
    aie.use_lock(lock, aie.LockAction.AcquireGreaterEqual, value = value)

def lock_release(lock: Any, value: int):
    aie.use_lock(lock, aie.LockAction.Release, value = value)

class AIE_CT_IP(ABC):
    stack_size: int
    link_with: str
    located_tile: tile_with_location
    ct: tile
    ct_row: int
    ct_col: int

    core_func: external_func
    def __init__(self, tile: tile_with_location, stack_size: int = 1024, link_with: str = ""):
        self.located_tile = tile
        self.ct = tile.tile
        self.ct_row = tile.row
        self.ct_col = tile.col
        self.stack_size = stack_size
        self.link_with = link_with
        self.core_func = None

    @abstractmethod
    def __call__(self):
        pass

    def set_stack_size(self, stack_size: int):
        self.stack_size = stack_size

    def set_link(self, link_with: str):
        self.link_with = link_with

    @property
    def S2MM(self):
        return [dma_port(self.located_tile, i) for i in range(2)]

    @property
    def MM2S(self):
        return [dma_port(self.located_tile, i) for i in range(2)]
        
class AIE_MT_IP(ABC):
    located_tile: tile_with_location
    mt: tile
    mt_row: int
    mt_col: int

    def __init__(self, tile: tile_with_location):
        self.located_tile = tile
        self.mt = tile.tile
        self.mt_row = tile.row
        self.mt_col = tile.col

    @abstractmethod
    def __call__(self):
        pass

    @property
    def S2MM(self):
        return [dma_port(self.located_tile, i) for i in range(6)]

    @property
    def MM2S(self):
        return [dma_port(self.located_tile, i) for i in range(6)]
