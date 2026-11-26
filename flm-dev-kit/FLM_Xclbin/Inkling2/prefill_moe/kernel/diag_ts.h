// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
// DIAG build helpers: per-phase core-timer timestamps recorded in the kernel and handed to the host in the first
// y chunk of the call (output values are garbage in these builds).
//  DIAG_TS : record up to TS_MAX timer samples (Timer_Low through the core processor bus; the host sequence writes
//            Core_Processor_Bus = 1 on every core first), stored as a 32-bit base + 16-bit saturating deltas in y chunk 0
//  DIAG_MAP: write c*256+e into element e of y chunk c (calibrates where chunk elements land in the y matrix)
#ifdef DIAG_TS
constexpr int TS_MAX = 224;
static inline __attribute__((always_inline)) unsigned ts_now() {
    unsigned t = read_tm(0x340F8);
    __builtin_aie2p_sched_barrier();
    return t;
}
static unsigned tsbuf[TS_MAX];
static bool ts_started = false;
#define TS_DECL int nts = 0;
// guarded sample: before the first call of the first dispatch the host has not enabled the processor bus yet
#define TSG() do { if (nts < TS_MAX) { tsbuf[nts] = ts_started ? ts_now() : 0u; nts++; } } while (0)
#define TS_DONE() do { ts_started = true; } while (0)
#define TS() do { if (nts < TS_MAX) { tsbuf[nts] = ts_now(); nts++; } } while (0)
#define TS_EMIT(y_, chunk_) do { { \
    unsigned short *yy = (unsigned short *)(y_); \
    yy[0] = (unsigned short)(tsbuf[0] & 0xFFFF); yy[1] = (unsigned short)(tsbuf[0] >> 16); yy[2] = (unsigned short)nts; \
    for (int q = 1; q < nts; q++) { unsigned d = tsbuf[q] - tsbuf[q - 1]; yy[2 + q] = d > 65535 ? (unsigned short)65535 : (unsigned short)d; } } } while (0)
#else
#define TS_DECL
#define TSG()
#define TS_DONE()
#define TS()
#define TS_EMIT(y_, chunk_)
#endif
#ifdef DIAG_MAP
#define MAP_EMIT(y_, chunk_) do { unsigned short *yy = (unsigned short *)(y_); for (int q = 0; q < y_slice; q++) yy[q] = (unsigned short)((chunk_) * 256 + q); } while (0)
#else
#define MAP_EMIT(y_, chunk_)
#endif
