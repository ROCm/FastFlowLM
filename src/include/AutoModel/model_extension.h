/// \file model_extension.h
/// \brief Stable C ABI for FastFlowLM model extensions
#pragma once

#include <stddef.h>
#include <stdint.h>

/// Define this when compiling the provider shared library so its query symbol
/// is exported on Windows. Consumers of the ABI leave it undefined.
#if defined(_WIN32) && defined(FLM_MODEL_EXTENSION_IMPLEMENTATION)
#define FLM_MODEL_EXTENSION_EXPORT __declspec(dllexport)
#elif defined(_WIN32)
#define FLM_MODEL_EXTENSION_EXPORT
#else
#define FLM_MODEL_EXTENSION_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
    FLM_MODEL_EXTENSION_ABI_V1 = 1,
};

/// \defgroup flm_model_extension_v1 Stable model-extension ABI v1
/// \brief Model-family and backend-provider extension contract.
/// \note The boundary contains C records and opaque provider objects only.
///       Providers may use C++ internally but must catch every exception.
/// \note One native bundle targets either XRT or HRX. NPU generation and model
///       compatibility belong in catalog/artifact metadata, not this ABI.
/// \note Fields may only be appended. Consumers check the fixed
///       `*_V1_MIN_SIZE` for the fields they use and ignore unknown tails.
/// @{

typedef enum flm_model_extension_status_v1 {
    FLM_MODEL_EXTENSION_STATUS_OK = 0,
    FLM_MODEL_EXTENSION_STATUS_ERROR = 1,
    FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT = 2,
    FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED = 3,
    FLM_MODEL_EXTENSION_STATUS_CANCELLED = 4,
} flm_model_extension_status_v1;

typedef enum flm_model_extension_npu_runtime_v1 {
    FLM_MODEL_EXTENSION_NPU_RUNTIME_XRT = 1,
    FLM_MODEL_EXTENSION_NPU_RUNTIME_HRX = 2,
} flm_model_extension_npu_runtime_v1;

/// \brief Non-owning byte sequence. Text fields contain UTF-8 without a
///        required trailing null.
typedef struct flm_span_v1 {
    const void* data;
    size_t size;
} flm_span_v1;

/// \brief Provider-owned BF16 output, valid until the next operation on the
///        same backend instance or until that instance is destroyed.
/// \note Forward and prefill return the logits consumed by FastFlowLM's
///       sampler. Cache callbacks return the engine's native flattened cache
///       view for the requested layer and index.
typedef struct flm_bf16_view_v1 {
    const uint16_t* data;
    size_t size;
} flm_bf16_view_v1;

/// \brief Provider-owned token sequence with the same lifetime as a BF16 view.
typedef struct flm_i32_view_v1 {
    const int32_t* data;
    size_t size;
} flm_i32_view_v1;

/// \brief Provider-private metadata borrowed for one prefill call.
/// \note Null means text-only prefill. FastFlowLM never dereferences `object`.
///       A complete model extension registers its frontend and backend together,
///       so both sides may agree on any private representation without adding
///       model-specific records to this ABI.
typedef struct flm_prefill_payload_v1 {
    size_t struct_size;
    /// Provider identity assigned by the host. The selected backend provider
    /// must match before the private object is forwarded.
    flm_span_v1 provider;
    const void* object;
} flm_prefill_payload_v1;

#define FLM_PREFILL_PAYLOAD_V1_MIN_SIZE \
    (offsetof(flm_prefill_payload_v1, object) + \
     sizeof(((flm_prefill_payload_v1*)0)->object))

/// \brief Host error receiver passed to fallible provider callbacks.
/// \note `set_error` copies `size` message bytes before returning.
typedef struct flm_error_sink_v1 {
    void* context;
    void (*set_error)(void* context, const char* message, size_t size);
} flm_error_sink_v1;

/// \brief Borrowed cancellation probe valid only during the current callback.
typedef struct flm_cancellation_v1 {
    void* context;
    int (*is_cancelled)(void* context);
} flm_cancellation_v1;

/// \brief Stable construction data for one backend engine.
/// \note Every span is borrowed only for the duration of `create`.
///       `runtime_device_handle` is the runtime's C handle, never a pointer to
///       an FLM or runtime C++ object. Its underlying device remains valid
///       until the returned backend instance is destroyed, but the provider
///       must not close the borrowed host handle. XRT providers may retain a
///       non-owning wrapper around the xclDeviceHandle. HRX providers use the
///       process-wide C runtime and receive a non-null availability token.
typedef struct flm_backend_create_context_v1 {
    size_t struct_size;
    flm_span_v1 model_path;
    flm_span_v1 model_info_json;
    flm_span_v1 tokenizer_config_json;
    flm_span_v1 extension_root;
    flm_span_v1 asset_root;
    void* runtime_device_handle;
    uint32_t npu_runtime;
    uint32_t context_length;
    uint32_t enable_preemption;
} flm_backend_create_context_v1;

#define FLM_BACKEND_CREATE_CONTEXT_V1_MIN_SIZE \
    (offsetof(flm_backend_create_context_v1, enable_preemption) + \
     sizeof(((flm_backend_create_context_v1*)0)->enable_preemption))

enum {
    FLM_BACKEND_TRAIT_REQUIRES_NPU_DEVICE = 1u << 0,
    FLM_BACKEND_TRAIT_SUPPORTS_PREEMPTION = 1u << 1,
    FLM_BACKEND_TRAIT_FORWARDS_PAST_EOS = 1u << 2,
    FLM_BACKEND_TRAIT_ALIGNS_PREFILL_CHUNKS = 1u << 3,
};

typedef struct flm_backend_traits_v1 {
    size_t struct_size;
    uint32_t flags;
    uint32_t max_context_length;
    uint32_t max_decode_length;
    int32_t forced_bos_id;
    uint32_t has_forced_bos_id;
    /// Borrowed during `register_backend_provider`; the host copies the ids.
    const int32_t* forced_eos_ids;
    size_t forced_eos_ids_size;
} flm_backend_traits_v1;

#define FLM_BACKEND_TRAITS_V1_MIN_SIZE \
    (offsetof(flm_backend_traits_v1, forced_eos_ids_size) + \
     sizeof(((flm_backend_traits_v1*)0)->forced_eos_ids_size))

/// \brief Operations implemented by one opaque causal-language-model engine.
/// \note Providers must catch all exceptions before returning through this ABI.
///       A nonzero return is reported with `error` when possible.
/// \note Every callback in this table is required. State snapshots, cache
///       access, speculative decoding, and health reporting are optional
///       capabilities queried separately, so this core does not grow whenever
///       FastFlowLM gains an execution feature.
/// \note Calls for one instance are serialized by FastFlowLM. Different model
///       instances may be called concurrently and must not share mutable state
///       without provider-side synchronization.
typedef struct flm_causal_lm_v1 {
    size_t struct_size;
    /// Decode one token and return the next-token logits.
    int (*forward)(void* object, int32_t token,
                   flm_bf16_view_v1* output,
                   const flm_error_sink_v1* error);
    /// Insert a token sequence and return the next-token logits. `payload` is
    /// null for text-only requests.
    int (*prefill)(void* object, const int32_t* tokens, size_t token_count,
                   const flm_prefill_payload_v1* payload,
                   flm_bf16_view_v1* output,
                   const flm_error_sink_v1* error);
    int (*set_context_length)(void* object, int32_t context_length,
                              const flm_error_sink_v1* error);
    int (*update_max_length)(void* object, uint32_t max_length,
                             const flm_error_sink_v1* error);
    int (*clear_context)(void* object, const flm_error_sink_v1* error);
    int (*get_current_context_length)(void* object, int32_t* context_length,
                                      const flm_error_sink_v1* error);
} flm_causal_lm_v1;

#define FLM_CAUSAL_LM_V1_MIN_SIZE \
    (offsetof(flm_causal_lm_v1, get_current_context_length) + \
     sizeof(((flm_causal_lm_v1*)0)->get_current_context_length))

enum {
    FLM_BACKEND_CAPABILITY_STATE = 1,
    FLM_BACKEND_CAPABILITY_SPECULATION = 2,
    FLM_BACKEND_CAPABILITY_HEALTH = 3,
    FLM_BACKEND_STATE_ABI_V1 = 1,
    FLM_BACKEND_SPECULATION_ABI_V1 = 1,
    FLM_BACKEND_HEALTH_ABI_V1 = 1,
};

/// \brief Optional checkpoint/restore support.
/// \note A provider paired with its own frontend may omit this table. An
///       alternate provider for a built-in frontend must implement it.
typedef struct flm_backend_state_v1 {
    size_t struct_size;
    /// Save the engine's single internal restore point and return its context
    /// length. FastFlowLM does not pass an external checkpoint identifier.
    int (*checkpoint)(void* object, int32_t* checkpoint,
                      const flm_error_sink_v1* error);
    /// Restore the most recently saved point and return the restored length.
    int (*restore)(void* object, int32_t* context_length,
                   const flm_error_sink_v1* error);
} flm_backend_state_v1;

#define FLM_BACKEND_STATE_V1_MIN_SIZE \
    (offsetof(flm_backend_state_v1, restore) + \
     sizeof(((flm_backend_state_v1*)0)->restore))

/// \brief Optional speculative-decoding support.
typedef struct flm_backend_speculation_v1 {
    size_t struct_size;
    int (*supports_speculation)(void* object);
    int (*speculate)(void* object, int32_t last_token, int32_t max_draft,
                     flm_i32_view_v1* output,
                     const flm_error_sink_v1* error);
    /// Optional timing detail; null reports zero.
    uint64_t (*last_speculation_prime_us)(void* object);
} flm_backend_speculation_v1;

#define FLM_BACKEND_SPECULATION_V1_MIN_SIZE \
    (offsetof(flm_backend_speculation_v1, speculate) + \
     sizeof(((flm_backend_speculation_v1*)0)->speculate))

#define FLM_BACKEND_SPECULATION_V1_PRIME_TIME_SIZE \
    (offsetof(flm_backend_speculation_v1, last_speculation_prime_us) + \
     sizeof(((flm_backend_speculation_v1*)0)->last_speculation_prime_us))

/// \brief Optional unrecoverable-failure reporting.
typedef struct flm_backend_health_v1 {
    size_t struct_size;
    int (*poisoned)(void* object);
} flm_backend_health_v1;

#define FLM_BACKEND_HEALTH_V1_MIN_SIZE \
    (offsetof(flm_backend_health_v1, poisoned) + \
     sizeof(((flm_backend_health_v1*)0)->poisoned))

/// \brief Return a provider-owned capability table or null when unsupported.
/// \note The returned table remains valid until the backend is destroyed.
///       `capability_abi` is the requested table ABI, currently v1.
typedef const void* (*flm_backend_query_capability_v1)(
    void* object, uint32_t capability, uint32_t capability_abi);

/// \brief One provider-owned opaque backend engine.
/// \note On entry, `struct_size` is the host's capacity. A provider must not
///       write appended fields beyond it and returns the initialized size.
typedef struct flm_backend_instance_v1 {
    size_t struct_size;
    void* object;
    void (*destroy)(void* object);
    const flm_causal_lm_v1* causal_lm;
    flm_span_v1 detail;
    flm_backend_query_capability_v1 query_capability;
} flm_backend_instance_v1;

#define FLM_BACKEND_INSTANCE_V1_MIN_SIZE \
    (offsetof(flm_backend_instance_v1, detail) + \
     sizeof(((flm_backend_instance_v1*)0)->detail))

#define FLM_BACKEND_INSTANCE_V1_QUERY_CAPABILITY_SIZE \
    (offsetof(flm_backend_instance_v1, query_capability) + \
     sizeof(((flm_backend_instance_v1*)0)->query_capability))

typedef int (*flm_backend_create_v1)(
    void* factory_context,
    const flm_backend_create_context_v1* context,
    flm_backend_instance_v1* instance,
    const flm_error_sink_v1* error);

enum {
    FLM_FRONTEND_TRAIT_SINGLE_TURN = 1u << 0,
    FLM_FRONTEND_TRAIT_SUPPORTS_LOGPROBS = 1u << 1,
    FLM_FRONTEND_TRAIT_SUPPORTS_CONTINUE_FINAL_MESSAGE = 1u << 2,
    FLM_FRONTEND_TRAIT_THINKING_ENABLED = 1u << 3,
    FLM_FRONTEND_TRAIT_TOOLS_ENABLED = 1u << 4,
};

/// \brief Static information needed by FLM's model-agnostic generation loop.
/// \note On entry, `struct_size` is the host's capacity. The provider returns
///       the size it initialized and must not write beyond the supplied size.
/// \note Every span and array remains owned by the provider for the lifetime of
///       the frontend instance.
/// \note `chat_format` uses a name accepted by FLM's output parser.
///       `tokenizer_json_path` is required for JSON-schema constraints.
typedef struct flm_frontend_traits_v1 {
    size_t struct_size;
    uint32_t flags;
    uint32_t vocabulary_size;
    int32_t bos_token_id;
    uint32_t has_bos_token_id;
    const int32_t* eos_token_ids;
    size_t eos_token_ids_size;
    flm_span_v1 chat_format;
    flm_span_v1 tokenizer_json_path;
} flm_frontend_traits_v1;

#define FLM_FRONTEND_TRAITS_V1_MIN_SIZE \
    (offsetof(flm_frontend_traits_v1, tokenizer_json_path) + \
     sizeof(((flm_frontend_traits_v1*)0)->tokenizer_json_path))

/// \brief Construction data for a provider-owned model frontend.
/// \note Spans and `runtime_device_handle` follow the same borrowing and
///       runtime rules as `flm_backend_create_context_v1`.
typedef struct flm_frontend_create_context_v1 {
    size_t struct_size;
    flm_span_v1 extension_root;
    flm_span_v1 asset_root;
    void* runtime_device_handle;
    uint32_t npu_runtime;
} flm_frontend_create_context_v1;

#define FLM_FRONTEND_CREATE_CONTEXT_V1_MIN_SIZE \
    (offsetof(flm_frontend_create_context_v1, npu_runtime) + \
     sizeof(((flm_frontend_create_context_v1*)0)->npu_runtime))

/// \brief Inputs supplied when the host loads one catalog-selected model.
typedef struct flm_frontend_load_context_v1 {
    size_t struct_size;
    flm_span_v1 model_path;
    flm_span_v1 model_info_json;
    flm_span_v1 backend;
    flm_span_v1 backend_provider;
    uint32_t context_length;
    uint32_t enable_preemption;
} flm_frontend_load_context_v1;

#define FLM_FRONTEND_LOAD_CONTEXT_V1_MIN_SIZE \
    (offsetof(flm_frontend_load_context_v1, enable_preemption) + \
     sizeof(((flm_frontend_load_context_v1*)0)->enable_preemption))

/// \brief Provider-owned result of preparing one model request.
/// \note On entry, `struct_size` is the host's capacity. The provider returns
///       the size it initialized and must not write beyond the supplied size.
/// \note `tokens` and `payload.object` remain valid until `release` is called.
///       `release` is required and is called exactly once, after all prefill
///       calls using the result have returned.
typedef struct flm_prepared_request_v1 {
    size_t struct_size;
    flm_i32_view_v1 tokens;
    flm_prefill_payload_v1 payload;
    uint32_t has_payload;
    uint32_t first_payload_token_count;
    void* release_context;
    void (*release)(void* release_context);
    /// Current assistant turn when `continue_final_message` was requested.
    flm_span_v1 prompt_tail;
} flm_prepared_request_v1;

#define FLM_PREPARED_REQUEST_V1_MIN_SIZE \
    (offsetof(flm_prepared_request_v1, release) + \
     sizeof(((flm_prepared_request_v1*)0)->release))

#define FLM_PREPARED_REQUEST_V1_PROMPT_TAIL_SIZE \
    (offsetof(flm_prepared_request_v1, prompt_tail) + \
     sizeof(((flm_prepared_request_v1*)0)->prompt_tail))

/// \brief Provider-owned chat-template output, valid until the next frontend
///        callback. `prompt_tail` is the current assistant turn when continuing
///        a final message and may be empty otherwise.
/// \note On entry, `struct_size` is the host's capacity. The provider returns
///       the size it initialized and must not write beyond the supplied size.
typedef struct flm_chat_template_result_v1 {
    size_t struct_size;
    flm_span_v1 text;
    flm_span_v1 prompt_tail;
} flm_chat_template_result_v1;

#define FLM_CHAT_TEMPLATE_RESULT_V1_MIN_SIZE \
    (offsetof(flm_chat_template_result_v1, prompt_tail) + \
     sizeof(((flm_chat_template_result_v1*)0)->prompt_tail))

/// \brief Model-specific preprocessing callbacks implemented by a provider.
/// \note `request_json` is a UTF-8 object. `prompt` is a string; `messages` and
///       `tools` are ordered JSON values; `images` and `audios` are string
///       arrays; their matching `*_payload_types` are integer arrays using
///       0=file, 1=base64, and 2=URL; `requested_max_new_tokens` is an optional
///       positive integer; `continue_final_message` is a boolean. Fields are
///       never removed or retyped.
///       Provider-private preprocessed data is returned only as an opaque
///       payload.
/// \note Calls for one instance are serialized. Providers must catch all
///       exceptions before returning.
/// \note `load`, `prepare`, `encode`, `decode_token`, and
///       `apply_chat_template` are required. `configure` and `query_flags` are
///       optional and must be left null when unsupported.
/// \note `configure` may run before or after `load`; request and tokenizer
///       callbacks run only after a successful `load`.
/// \note Views returned by `encode`, `decode_token`, and
///       `apply_chat_template` remain valid until the next callback on that
///       instance. The host copies them before returning.
/// \note On factory entry, `struct_size` is the host's capacity. The provider
///       returns the size it initialized and must not write beyond it.
typedef struct flm_model_frontend_v1 {
    size_t struct_size;
    void* object;
    void (*destroy)(void* object);
    int (*load)(void* object, const flm_frontend_load_context_v1* context,
                flm_frontend_traits_v1* traits,
                const flm_error_sink_v1* error);
    int (*prepare)(void* object, flm_span_v1 request_json,
                   const flm_cancellation_v1* cancellation,
                   flm_prepared_request_v1* request,
                   const flm_error_sink_v1* error);
    int (*encode)(void* object, flm_span_v1 text, flm_i32_view_v1* tokens,
                  const flm_error_sink_v1* error);
    int (*decode_token)(void* object, int32_t token, flm_span_v1* text,
                        const flm_error_sink_v1* error);
    int (*apply_chat_template)(void* object, flm_span_v1 messages_json,
                               flm_span_v1 tools_json,
                               uint32_t continue_final_message,
                               flm_chat_template_result_v1* result,
                               const flm_error_sink_v1* error);
    /// Apply a named frontend option. `value_json` is one complete JSON value.
    /// Unknown names return `FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED`.
    int (*configure)(void* object, flm_span_v1 name,
                     flm_span_v1 value_json,
                     const flm_error_sink_v1* error);
    /// Return current frontend flags after dynamic configuration. Optional;
    /// null keeps the flags returned by `load`.
    int (*query_flags)(void* object, uint32_t* flags,
                       const flm_error_sink_v1* error);
} flm_model_frontend_v1;

#define FLM_MODEL_FRONTEND_V1_MIN_SIZE \
    (offsetof(flm_model_frontend_v1, apply_chat_template) + \
     sizeof(((flm_model_frontend_v1*)0)->apply_chat_template))

#define FLM_MODEL_FRONTEND_V1_CONFIGURE_SIZE \
    (offsetof(flm_model_frontend_v1, configure) + \
     sizeof(((flm_model_frontend_v1*)0)->configure))

#define FLM_MODEL_FRONTEND_V1_QUERY_FLAGS_SIZE \
    (offsetof(flm_model_frontend_v1, query_flags) + \
     sizeof(((flm_model_frontend_v1*)0)->query_flags))

typedef int (*flm_frontend_create_v1)(
    void* factory_context,
    const flm_frontend_create_context_v1* context,
    flm_model_frontend_v1* frontend,
    const flm_error_sink_v1* error);

/// \brief Host callbacks available while an extension registers.
/// \note The registrar and its host context are valid only for the duration of
///       the `register_extension` call.
/// \note Registration strings and trait arrays are borrowed for each callback;
///       the host copies them before returning.
/// \note `set_extension_context` transfers one provider-state lifetime
///       reference to the host. It may be called at most once; the host invokes
///       its destructor after every provider factory and object is released.
///       Providers must use `struct_size` before accessing appended fields.
typedef struct flm_model_extension_registrar_v1 {
    size_t struct_size;
    void* host_context;
    int (*set_extension_context)(
        void* host_context, void* extension_context,
        void (*destroy_extension_context)(void* extension_context));
    int (*register_backend_provider)(
        void* host_context, const char* family, const char* backend,
        const char* provider, const flm_backend_traits_v1* traits,
        flm_backend_create_v1 create, void* factory_context);
    int (*register_frontend_provider)(
        void* host_context, const char* family, const char* provider,
        flm_frontend_create_v1 create, void* factory_context);
} flm_model_extension_registrar_v1;

#define FLM_MODEL_EXTENSION_REGISTRAR_V1_MIN_SIZE \
    (offsetof(flm_model_extension_registrar_v1, register_backend_provider) + \
     sizeof(((flm_model_extension_registrar_v1*)0) \
                ->register_backend_provider))

#define FLM_MODEL_EXTENSION_REGISTRAR_V1_FRONTEND_SIZE \
    (offsetof(flm_model_extension_registrar_v1, register_frontend_provider) + \
     sizeof(((flm_model_extension_registrar_v1*)0) \
                ->register_frontend_provider))

/// \brief Description returned by `flm_model_extension_query_v1`.
/// \note `name` and `version` are null-terminated UTF-8 strings. They and this
///       descriptor remain provider-owned and valid while the library is
///       loaded.
typedef struct flm_model_extension_v1 {
    size_t struct_size;
    uint32_t abi_version;
    uint32_t npu_runtime;
    const char* name;
    const char* version;
    int (*register_extension)(
        const flm_model_extension_registrar_v1* registrar,
        const flm_error_sink_v1* error);
} flm_model_extension_v1;

#define FLM_MODEL_EXTENSION_V1_MIN_SIZE \
    (offsetof(flm_model_extension_v1, register_extension) + \
     sizeof(((flm_model_extension_v1*)0)->register_extension))

typedef const flm_model_extension_v1*
(*flm_model_extension_query_v1_fn)(void);

/// \brief Symbol every native model extension exports.
FLM_MODEL_EXTENSION_EXPORT const flm_model_extension_v1*
flm_model_extension_query_v1(void);

/// @}

#ifdef __cplusplus
}
#endif
