# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from dataclasses import dataclass
from typing import Any

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from utils import mlir_context, tile_with_location, connect_dma, lock_acquire, lock_release


@dataclass
class attn_cfg:
    """Everything that differs between the full-attention and sliding-window
    columns of the fused array.

    The standalone Gemma4 designs each got to own the whole `mlir_context`, so
    they read `mctx.d`, `mctx.lq`, ... directly.  Here the two variants share one
    module, so each carries its own types and geometry instead.
    """
    name: str          # "full" / "swa" -- prefixes buffer and RTP names
    kernel_obj: str    # object file to link the core against
    func: Any          # external_func handle for that object's entry point
    d: int
    lq: int
    lk: int
    lk_mt: int
    # types
    q_ty: Any
    in_ty: Any
    s_ty: Any
    m_ty: Any
    y_ty: Any
    o_ty: Any
    q_mem_ty: Any
    in_mem_ty: Any
    o_mem_ty: Any
    pkt_base: int = 0  # first of the five packet ids this column owns


# A fused attention column is a complete CU: four cores, one per compute row.
N_ROWS = 4


def attn_cttile(tile: tile_with_location, cfg: attn_cfg, mctx: mlir_context):
    """One attention compute tile.

    Identical in shape to the standalone Gemma4 core tile -- q in on s2mm 0, k/v
    ping-pong on s2mm 1, o ping-pong out on mm2s 0 -- but the core's `col`
    argument is now always 0: a solo column has no partner to split the staged q
    with, so the row alone selects this core's lq tokens.
    """
    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col
    p = cfg.name

    L_begin_buffer = mctx.buffers[f"L_begin_attn_{p}_core_{ct_row}_{ct_col}"]
    L_end_buffer = mctx.buffers[f"L_end_attn_{p}_core_{ct_row}_{ct_col}"]
    window_size_buffer = mctx.buffers[f"window_size_attn_{p}_core_{ct_row}_{ct_col}"]

    q_buffer = aie.buffer(tile=ct, datatype=cfg.q_ty, name=f"q_{ct_row}_{ct_col}")
    in_buffer_0 = aie.buffer(tile=ct, datatype=cfg.in_ty, name=f"in_0_{ct_row}_{ct_col}")
    in_buffer_1 = aie.buffer(tile=ct, datatype=cfg.in_ty, name=f"in_1_{ct_row}_{ct_col}")
    o_buffer_0 = aie.buffer(tile=ct, datatype=cfg.o_ty, name=f"o_0_{ct_row}_{ct_col}")
    o_buffer_1 = aie.buffer(tile=ct, datatype=cfg.o_ty, name=f"o_1_{ct_row}_{ct_col}")
    s_buffer = aie.buffer(tile=ct, datatype=cfg.s_ty, name=f"s_{ct_row}_{ct_col}")
    m_buffer = aie.buffer(tile=ct, datatype=cfg.m_ty, name=f"m_{ct_row}_{ct_col}")
    y_buffer = aie.buffer(tile=ct, datatype=cfg.y_ty, name=f"y_{ct_row}_{ct_col}")

    q_prod_lock = aie.lock(ct, lock_id=0, init=1)
    q_cons_lock = aie.lock(ct, lock_id=1, init=0)
    in_prod_lock = aie.lock(ct, lock_id=2, init=2)
    in_cons_lock = aie.lock(ct, lock_id=3, init=0)
    o_prod_lock = aie.lock(ct, lock_id=4, init=2)
    o_cons_lock = aie.lock(ct, lock_id=5, init=0)

    kernel = cfg.func

    @aie.core(ct, link_with=cfg.kernel_obj, stack_size=1024 * 3)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            kernel(o_buffer_0, o_buffer_1, q_buffer, in_buffer_0, in_buffer_1,
                   s_buffer, m_buffer, y_buffer,
                   L_begin_buffer, L_end_buffer, window_size_buffer,
                   ct_row - 2, 0)

    @aie.mem(ct)
    def mem_body(block):
        aie.dma_start(channel_dir=aie.DMAChannelDir.S2MM, channel_index=0, dest=block[1], chain=block[2])

        with block[1]:
            lock_acquire(q_prod_lock, 1)
            aie.dma_bd(q_buffer)
            lock_release(q_cons_lock, 1)
            aie.next_bd(block[1])

        with block[2]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.S2MM, channel_index=1, dest=block[3], chain=block[5])

        with block[3]:
            lock_acquire(in_prod_lock, 1)
            aie.dma_bd(in_buffer_0)
            lock_release(in_cons_lock, 1)
            aie.next_bd(block[4])

        with block[4]:
            lock_acquire(in_prod_lock, 1)
            aie.dma_bd(in_buffer_1)
            lock_release(in_cons_lock, 1)
            aie.next_bd(block[3])

        with block[5]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.MM2S, channel_index=0, dest=block[6], chain=block[8])

        with block[6]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_0)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[7])

        with block[7]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_1)
            lock_release(o_prod_lock, 1)
            aie.next_bd(block[6])

        with block[8]:
            aie.EndOp()


def attn_memtile(tile: tile_with_location, cfg: attn_cfg, mctx: mlir_context):
    """The memtile of a solo attention column.

    q no longer passes through here.  It is small -- a few percent of the
    column's inbound bytes -- so it goes straight from the shim to the four
    cores as four packets, which buys back the s2mm channel k and v need to
    arrive on channels of their own:

        s2mm  0 = k                              mm2s  4    = k/v broadcast
        s2mm  1 = v                              mm2s  5    = o -> shim
        s2mm  2..5 = o from rows 0..3

    The old map put k and v on one s2mm channel, so the host had to hop from
    the k half of the cache to the v half with a BD *stride* of
    KV_CACHE_SIZE/2.  That field is 20 bits wide in 32-bit words, and
    KV_CACHE_SIZE/2 = DK * MAX_L grows with the whole cache: E2B at MAX_L 4096
    sat exactly on the limit and E4B, with twice the kv heads, was 2x over it
    and wrote truncated descriptors.  Split across two channels the hop becomes
    a base *address* instead -- unchecked, and the largest remaining stride is
    lk_mt * d * Nd, which depends on the block size rather than the context
    length.

    Ordering used to come for free from the single channel's four-BD chain.
    Now two independent channels fill the two halves of a buffer, so a lock
    chain enforces it: k takes a buffer credit and hands off to v, v hands the
    finished pair to the broadcast, and the broadcast hands the credit back.
    A hardware BD has one acquire and one release, which is exactly what each
    step needs.
    """
    mt = tile.tile
    mt_row = tile.row
    mt_col = tile.col

    d = cfg.d
    lq = cfg.lq
    lk = cfg.lk
    lk_mt = cfg.lk_mt

    in_buffer_0 = aie.buffer(tile=mt, datatype=cfg.in_mem_ty, name=f"in_0_{mt_row}_{mt_col}")
    in_buffer_1 = aie.buffer(tile=mt, datatype=cfg.in_mem_ty, name=f"in_1_{mt_row}_{mt_col}")
    o_buffer_0 = aie.buffer(tile=mt, datatype=cfg.o_mem_ty, name=f"o_0_{mt_row}_{mt_col}")
    o_buffer_1 = aie.buffer(tile=mt, datatype=cfg.o_mem_ty, name=f"o_1_{mt_row}_{mt_col}")

    # k/v: two buffers, and a buffer moves k -> v -> broadcast -> free.
    in_prod_lock = aie.lock(mt, lock_id=5, init=2)   # a free buffer, taken by k
    kv_mid_lock = aie.lock(mt, lock_id=6, init=0)    # k half done, v may fill its half
    in_cons_lock = aie.lock(mt, lock_id=12, init=0)  # pair complete, broadcast may ship

    o_prod_lock_0 = aie.lock(mt, lock_id=7, init=2)
    o_prod_lock_1 = aie.lock(mt, lock_id=8, init=0)
    o_prod_lock_2 = aie.lock(mt, lock_id=9, init=0)
    o_prod_lock_3 = aie.lock(mt, lock_id=10, init=0)
    o_cons_lock = aie.lock(mt, lock_id=11, init=0)

    # The core's _attn_qk wants k as 8x8 sub-tiles, so the staging BD transposes
    # on the way in.  Generalised from the two standalone spellings -- DH512's
    # (64, 64) and DH256's (32, 128) are both (d // 8, lk * 8).
    kv_dims = [(lk_mt // lk, lk * d), (lk, 8), (d // 8, lk * 8), (8, 1)]
    o_dims = [(lq // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)]

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        # ---------------- k in ----------------
        aie.dma_start(channel_dir=aie.DMAChannelDir.S2MM, channel_index=0, dest=block[1], chain=block[3])

        with block[1]:
            lock_acquire(in_prod_lock, 1)
            aie.dma_bd(in_buffer_0, offset=0, len=lk_mt * d, dimensions=kv_dims)
            lock_release(kv_mid_lock, 1)
            aie.next_bd(block[2])

        with block[2]:
            lock_acquire(in_prod_lock, 1)
            aie.dma_bd(in_buffer_1, offset=0, len=lk_mt * d, dimensions=kv_dims)
            lock_release(kv_mid_lock, 1)
            aie.next_bd(block[1])

        # ---------------- v in ----------------
        # Chained behind k on the same buffer order, so the credit v takes is
        # always for the buffer whose k half just landed.
        with block[3]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.S2MM, channel_index=1, dest=block[4], chain=block[6])

        with block[4]:
            lock_acquire(kv_mid_lock, 1)
            aie.dma_bd(in_buffer_0, offset=lk_mt * d, len=lk_mt * d, dimensions=kv_dims)
            lock_release(in_cons_lock, 1)
            aie.next_bd(block[5])

        with block[5]:
            lock_acquire(kv_mid_lock, 1)
            aie.dma_bd(in_buffer_1, offset=lk_mt * d, len=lk_mt * d, dimensions=kv_dims)
            lock_release(in_cons_lock, 1)
            aie.next_bd(block[4])

        # ---------------- k/v broadcast to all four rows ----------------
        with block[6]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.MM2S, channel_index=4, dest=block[7], chain=block[9])

        with block[7]:
            lock_acquire(in_cons_lock, 1)
            aie.dma_bd(in_buffer_0, offset=0, len=2 * lk_mt * d,
                       dimensions=[(2 * lk_mt // lk, lk * d), (lk, d), (d, 1)])
            lock_release(in_prod_lock, 1)
            aie.next_bd(block[8])

        with block[8]:
            lock_acquire(in_cons_lock, 1)
            aie.dma_bd(in_buffer_1, offset=0, len=2 * lk_mt * d,
                       dimensions=[(2 * lk_mt // lk, lk * d), (lk, d), (d, 1)])
            lock_release(in_prod_lock, 1)
            aie.next_bd(block[7])

        # ---------------- o in, one channel per compute row ----------------
        # Chained so the four rows land in row order and the pair of buffers is
        # only released once all four have written.
        with block[9]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.S2MM, channel_index=2, dest=block[10], chain=block[12])

        with block[10]:
            lock_acquire(o_prod_lock_0, 1)
            aie.dma_bd(o_buffer_0, offset=0, len=lq * d, dimensions=o_dims)
            lock_release(o_prod_lock_1, 1)
            aie.next_bd(block[11])

        with block[11]:
            lock_acquire(o_prod_lock_0, 1)
            aie.dma_bd(o_buffer_1, offset=0, len=lq * d, dimensions=o_dims)
            lock_release(o_prod_lock_1, 1)
            aie.next_bd(block[10])

        with block[12]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.S2MM, channel_index=3, dest=block[13], chain=block[15])

        with block[13]:
            lock_acquire(o_prod_lock_1, 1)
            aie.dma_bd(o_buffer_0, offset=lq * d, len=lq * d, dimensions=o_dims)
            lock_release(o_prod_lock_2, 1)
            aie.next_bd(block[14])

        with block[14]:
            lock_acquire(o_prod_lock_1, 1)
            aie.dma_bd(o_buffer_1, offset=lq * d, len=lq * d, dimensions=o_dims)
            lock_release(o_prod_lock_2, 1)
            aie.next_bd(block[13])

        with block[15]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.S2MM, channel_index=4, dest=block[16], chain=block[18])

        with block[16]:
            lock_acquire(o_prod_lock_2, 1)
            aie.dma_bd(o_buffer_0, offset=2 * lq * d, len=lq * d, dimensions=o_dims)
            lock_release(o_prod_lock_3, 1)
            aie.next_bd(block[17])

        with block[17]:
            lock_acquire(o_prod_lock_2, 1)
            aie.dma_bd(o_buffer_1, offset=2 * lq * d, len=lq * d, dimensions=o_dims)
            lock_release(o_prod_lock_3, 1)
            aie.next_bd(block[16])

        with block[18]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.S2MM, channel_index=5, dest=block[19], chain=block[21])

        with block[19]:
            lock_acquire(o_prod_lock_3, 1)
            aie.dma_bd(o_buffer_0, offset=3 * lq * d, len=lq * d, dimensions=o_dims)
            lock_release(o_cons_lock, 1)
            aie.next_bd(block[20])

        with block[20]:
            lock_acquire(o_prod_lock_3, 1)
            aie.dma_bd(o_buffer_1, offset=3 * lq * d, len=lq * d, dimensions=o_dims)
            lock_release(o_cons_lock, 1)
            aie.next_bd(block[19])

        # ---------------- o out ----------------
        with block[21]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.MM2S, channel_index=5, dest=block[22], chain=block[24])

        with block[22]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_0)
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[23])

        with block[23]:
            lock_acquire(o_cons_lock, 1)
            aie.dma_bd(o_buffer_1)
            lock_release(o_prod_lock_0, 1)
            aie.next_bd(block[22])

        with block[24]:
            aie.EndOp()


def q_relay_resources(tile: tile_with_location, cfg: attn_cfg):
    """Buffers and locks for staging one attention column's q on a borrowed memtile.

    The attention column's own memtile has no s2mm channel left once k and v
    arrive separately, so q was briefly packet switched from the column's shim
    straight to the four cores, sharing shim mm2s 0 with k.  That deadlocks.
    One packet-switched port serialises its packets, so a core that has not yet
    consumed its previous q blocks the port head-of-line; k queues behind it,
    the column memtile never completes a k/v pair, the broadcast the cores are
    waiting on never fires, and the core never drains the q that is blocking
    everything.  The driver's 2 s TDR then fired, four times, which is the
    ~8.18 s the attention engine was costing per layer.

    q is staged on an mm column that is not one of X_COLS instead.  Such a
    column's dequant_mm memtile used to build an x path that nothing is ever
    routed to, so with `host_x=False` its s2mm 0, its mm2s 0 and its locks 0..4
    come free; mm2s 3/4/5 were never used at all.  That is one inbound channel
    plus the four outbound channels the four cores need, and the whole q path
    is circuit switched again with the 8x8 sub-tiling back on chip.

    These BDs share a memtile_dma body with that column's y and qw traffic --
    a tile has exactly one -- so the emitter below is handed the caller's block
    list rather than opening its own.
    """
    mt = tile.tile
    d = cfg.d
    lq_ct = cfg.lq

    q_buffers = [
        aie.buffer(tile=mt, datatype=cfg.q_mem_ty, name=f"q_0_{tile.row}_{tile.col}"),
        aie.buffer(tile=mt, datatype=cfg.q_mem_ty, name=f"q_1_{tile.row}_{tile.col}"),
    ]
    # y holds locks 5..9 and qw 10..11; 0..4 are the ones host_x=False frees.
    q_prod_lock = aie.lock(mt, lock_id=0, init=2)
    q_cons_locks = [aie.lock(mt, lock_id=1 + i, init=0) for i in range(N_ROWS)]

    return dict(
        buffers=q_buffers,
        prod=q_prod_lock,
        cons=q_cons_locks,
        # The core's _attn_qk wants its lq x d slice as 8x8 sub-tiles.  Doing
        # that here is free; expressing it on the shim's DDR read instead would
        # make the innermost contiguous run 8 bf16 = 16 bytes.
        dims=[(lq_ct // 8, 8 * d), (d // 8, 8), (8, d), (8, 1)],
        slice_len=lq_ct * d,
        # mm2s 1 and 2 carry y-out and the w broadcast, so q takes 0 then 3/4/5.
        out_channels=[0, 3, 4, 5],
    )


def q_relay_emit(block, res, base: int, tail):
    """Emit the q relay's BDs into an existing memtile_dma body.

    `base` is the first block key to use (well clear of the caller's own), and
    `tail` is the dma_start the last q chain hands off to, so q simply becomes
    the head of that memtile's existing chain.
    """
    bufs = res["buffers"]
    prod = res["prod"]
    cons = res["cons"]

    # ---------------- q in from the borrowed shim ----------------
    aie.dma_start(channel_dir=aie.DMAChannelDir.S2MM, channel_index=0,
                  dest=block[base], chain=block[base + 2])
    for i, buf in enumerate(bufs):
        with block[base + i]:
            lock_acquire(prod, 1)
            aie.dma_bd(buf)
            lock_release(cons[0], 1)
            aie.next_bd(block[base + (1 - i)])

    # ---------------- one slice out per compute row ----------------
    # Chained row by row, so the pair of buffers is only freed once all four
    # rows have taken their slice.
    for i, ch in enumerate(res["out_channels"]):
        head = base + 2 + i * 3
        nxt = head + 3 if i < N_ROWS - 1 else None
        with block[head]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.MM2S, channel_index=ch,
                          dest=block[head + 1],
                          chain=block[nxt] if nxt is not None else block[tail])
        rel = cons[i + 1] if i < N_ROWS - 1 else prod
        for b, buf in enumerate(bufs):
            with block[head + 1 + b]:
                lock_acquire(cons[i], 1)
                aie.dma_bd(buf, offset=i * res["slice_len"], len=res["slice_len"],
                           dimensions=res["dims"])
                lock_release(rel, 1)
                aie.next_bd(block[head + 1 + (1 - b)])


def attn_column(shim: tile_with_location, memtile: tile_with_location,
                cores, cfg: attn_cfg, mctx: mlir_context,
                q_shim: tile_with_location, q_memtile: tile_with_location):
    """Build one whole attention column and wire its shim <-> memtile <-> cores.

    The column's own shim has exactly two mm2s channels and the column wants
    three inbound streams, so q is issued from a *different* column's shim --
    one of the two mm columns that host no x broadcast -- and staged on that
    column's otherwise idle memtile, which fans it out to these four cores.
    Every flow is circuit switched; nothing here is packet switched any more.
    See q_relay_memtile for why the packet-switched version deadlocked.
    """
    for core in cores:
        attn_cttile(core, cfg, mctx)
    attn_memtile(memtile, cfg, mctx)

    connect_dma(q_shim, q_memtile, 0, 0)                  # q in, on the borrowed column
    for i, core in enumerate(cores):
        connect_dma(q_memtile, core, [0, 3, 4, 5][i], 0)  # q slice -> row i
    connect_dma(shim, memtile, 0, 0)                      # k in
    connect_dma(shim, memtile, 1, 1)                      # v in
    for i, core in enumerate(cores):
        connect_dma(memtile, core, 4, 1)      # k/v broadcast
        connect_dma(core, memtile, 0, 2 + i)  # o from row i
    connect_dma(memtile, shim, 5, 0)          # o out
