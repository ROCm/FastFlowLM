// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
//
// Control packets from a core to the shim tiles: read a shim BD's address words once, then point shim BDs at
// device-chosen offsets of the same buffer and push them on the shim's MM2S task queues. The core sends the
// packets through its own MM2S channel CTRL_PORT (BD CTRL_OUT_BD, buffer `out`, locks OUT_PROD / OUT_CONS) and
// takes the read responses on S2MM CTRL_PORT (one word into `in`, locks IN_PROD / IN_CONS). Needs the core's
// processor bus (host: maskwrite 0x32038 bit 0) to resize its own BD.

#pragma once
#include <stdint.h>

#include "aie_kernel_utils.h"

namespace hub_ctrl {

constexpr uint32_t CTRL_PORT = 1, CTRL_OUT_BD = 10;
constexpr int OUT_PROD = 6, OUT_CONS = 7, IN_PROD = 8, IN_CONS = 9;

constexpr uint32_t bd_addr(uint32_t bd) { return 0x1D000u + bd * 0x20u; }
constexpr uint32_t SHIM_MM2S_QUEUE[2] = {0x1D214u, 0x1D21Cu};

inline uint32_t parity_bit(uint32_t v) { return (uint32_t)(__builtin_popcount(v) & 1) ^ 1u; }   // makes the count odd

// control word: response packet id, op 0 write / 1 read, beats = data words - 1, register address
inline uint32_t ctrl_word(uint32_t resp_id, uint32_t op, uint32_t beats, uint32_t addr) {
    const uint32_t w = resp_id << 24 | op << 22 | beats << 20 | addr;
    return w | parity_bit(w) << 31;
}

// packet header from this core (type 0)
inline uint32_t pkt_header(uint32_t id) {
    const uint32_t h = (uint32_t)get_coreid() << 16 | id;
    return h | parity_bit(h) << 31;
}

inline void bus_write(uint32_t v, uint32_t addr) { write_tm(v, addr); }
inline uint32_t bus_read(uint32_t addr) { return read_tm(addr); }

inline void sched_barrier() {
    __builtin_aie2p_sched_barrier();
    volatile int32_t k = 0;
    k++;
    __builtin_aie2p_sched_barrier();
}

// resize this core's control BD to `words` and start it once
inline void send_len(uint32_t words) {
    const uint32_t ctrl = CTRL_PORT == 0 ? 0x1DE10u : 0x1DE18u, queue = CTRL_PORT == 0 ? 0x1DE14u : 0x1DE1Cu;
    const uint32_t w0 = bus_read(bd_addr(CTRL_OUT_BD));
    sched_barrier();
    bus_write((w0 & ~0x3FFFu) | words, bd_addr(CTRL_OUT_BD));
    sched_barrier();
    bus_write(1u << 1, ctrl);
    sched_barrier();
    bus_write(1u, ctrl);   // as hy-mem
    sched_barrier();
    bus_write(CTRL_OUT_BD, queue);
    sched_barrier();
}

// one register of a shim tile (packet id `id`, responses on `resp_id`)
inline uint32_t shim_read(uint32_t* out, volatile uint32_t* in, uint32_t id, uint32_t resp_id, uint32_t addr, void (*tp)(uint32_t) = nullptr) {
    send_len(2);
    if (tp) tp(0x11);
    _lock_acquire(OUT_PROD);
    out[0] = pkt_header(id);
    out[1] = ctrl_word(resp_id, 1, 0, addr);
    _lock_release(OUT_CONS);
    if (tp) tp(0x12);
    _lock_acquire(IN_CONS);
    const uint32_t v = in[0];
    _lock_release(IN_PROD);
    return v;
}

// shim BD `bd`: words 0-2 = length (32-bit words), address base + offset (48 bits, carry into word 2)
inline void shim_bd(uint32_t* out, uint32_t id, uint32_t bd, uint32_t base_lo, uint32_t base_hi, uint32_t offset, uint32_t words) {
    const uint32_t lo = base_lo + offset;
    const uint32_t hi = (base_hi & 0xFFFF0000u) | (((base_hi & 0xFFFFu) + (lo < base_lo)) & 0xFFFFu);
    send_len(5);
    _lock_acquire(OUT_PROD);
    out[0] = pkt_header(id);
    out[1] = ctrl_word(0, 0, 2, bd_addr(bd));
    out[2] = words;
    out[3] = lo;
    out[4] = hi;
    _lock_release(OUT_CONS);
}

// push shim BD `bd` on MM2S channel `ch`'s task queue
inline void shim_push(uint32_t* out, uint32_t id, uint32_t ch, uint32_t bd) {
    send_len(3);
    _lock_acquire(OUT_PROD);
    out[0] = pkt_header(id);
    out[1] = ctrl_word(0, 0, 0, SHIM_MM2S_QUEUE[ch]);
    out[2] = bd;
    _lock_release(OUT_CONS);
}

}  // namespace hub_ctrl
