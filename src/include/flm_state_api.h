/// \file flm_state_api.h
/// \brief the whole of what a prefix cache (or any multi-context server) needs from an engine library
/// \note Plain C: no classes, no templates, no allocation crossing the library boundary. An engine library that can
///       save its context exports ONE function returning a table of these calls (see qwen3_5vl_npu.hpp for the
///       pattern); a library that cannot, exports nothing and is simply "not supported". The engine only reads its
///       live context into a state it was handed and writes one back. How many states exist, which one is used and
///       when one is dropped is the caller's business: the engine never sees a slot number or a slot count.
///       The state is opaque to the caller; the engine that made it allocates it, fills it, and frees it.
///       `version` grows when a call is added at the end of the table; a caller must not call past `struct_size`.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLM_STATE_API_VERSION 1

typedef struct flm_state flm_state; ///< opaque: one saved context of one engine

typedef struct flm_state_api {
    uint32_t version;     ///< FLM_STATE_API_VERSION the library was built with
    uint32_t struct_size; ///< sizeof(flm_state_api) the library was built with

    /// an empty state of this engine's kind, to be filled by state_read; allocates nothing big. NULL if it cannot.
    flm_state* (*state_new)(void* engine);
    void (*state_free)(flm_state* state);

    /// copy the engine's live context into `state` (K/V rows from `kv_from` on; 0 = self-contained).
    /// returns the context length read, or -1 if `state` is not this engine's
    int (*state_read)(void* engine, flm_state* state, int kv_from);
    /// put `state` back: recurrent state, context length, K/V rows from its kv_from on, rows past the length zeroed.
    /// returns the context length written, or -1 if `state` is empty or not this engine's
    int (*state_write)(void* engine, const flm_state* state);
    /// put back only the K/V rows `state` holds (the shared head of a prompt under a state that keeps its tail).
    /// returns the context length of `state`, or -1
    int (*state_write_kv)(void* engine, const flm_state* state);

    int (*state_context_length)(const flm_state* state); ///< length it was read at; 0 until filled
    int (*state_kv_from)(const flm_state* state);        ///< first position whose K/V rows it holds
    /// host bytes a state read at `length` with K/V rows from `from` on would hold; works on an unfilled state
    size_t (*state_estimate_bytes)(const flm_state* state, int length, int from);
} flm_state_api;

#ifdef __cplusplus
}
#endif
