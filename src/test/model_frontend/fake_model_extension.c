#define FLM_MODEL_EXTENSION_IMPLEMENTATION
#include "AutoModel/model_extension.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int flm_test_model_extension_dependency(void);

#ifndef FLM_TEST_EXTENSION_CLEANUP_PATH
#error FLM_TEST_EXTENSION_CLEANUP_PATH must name the cleanup marker
#endif
#ifndef FLM_TEST_NPU_RUNTIME
#error FLM_TEST_NPU_RUNTIME must name the declared NPU runtime
#endif

/* One implementation backs three narrowly defined provider test doubles. */
#define FLM_TEST_EXTENSION_MODE_BACKEND 0
#define FLM_TEST_EXTENSION_MODE_COMPLETE 1
#define FLM_TEST_EXTENSION_MODE_REGISTRATION_FAILURE 2
#define FLM_TEST_EXTENSION_MODE_NO_CAPABILITIES 3
#ifndef FLM_TEST_EXTENSION_MODE
#define FLM_TEST_EXTENSION_MODE FLM_TEST_EXTENSION_MODE_BACKEND
#endif

#if FLM_TEST_EXTENSION_MODE == FLM_TEST_EXTENSION_MODE_COMPLETE
#define FLM_TEST_FAMILY "complete-plugin-test"
#elif FLM_TEST_EXTENSION_MODE == FLM_TEST_EXTENSION_MODE_NO_CAPABILITIES
#define FLM_TEST_FAMILY "no-capability-plugin-test"
#else
#define FLM_TEST_FAMILY "plugin-test"
#endif

typedef struct fake_backend {
    uint16_t logits[4];
    int32_t context_length;
    int32_t checkpoint_length;
} fake_backend;

static int ReturnLogits(void* opaque, flm_bf16_view_v1* output) {
    fake_backend* backend = (fake_backend*)opaque;
    if (!backend || !output)
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    output->data = backend->logits;
    output->size = 4;
    return FLM_MODEL_EXTENSION_STATUS_OK;
}

static int Forward(void* opaque, int32_t token,
                   flm_bf16_view_v1* output,
                   const flm_error_sink_v1* error) {
    fake_backend* backend = (fake_backend*)opaque;
    static const char message[] = "fake forward failure";
    if (token == -1) {
        if (error && error->set_error) {
            error->set_error(
                error->context, message, sizeof(message) - 1);
        }
        return FLM_MODEL_EXTENSION_STATUS_ERROR;
    }
    if (backend) ++backend->context_length;
    return ReturnLogits(opaque, output);
}

static int Prefill(void* opaque, const int32_t* tokens, size_t token_count,
                   const flm_prefill_payload_v1* payload,
                   flm_bf16_view_v1* output,
                   const flm_error_sink_v1* error) {
    fake_backend* backend = (fake_backend*)opaque;
    (void)tokens;
    (void)error;
#if FLM_TEST_EXTENSION_MODE == FLM_TEST_EXTENSION_MODE_COMPLETE
    if (!payload ||
        payload->struct_size < FLM_PREFILL_PAYLOAD_V1_MIN_SIZE ||
        !payload->object) {
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    }
#else
    (void)payload;
#endif
    if (backend) backend->context_length += (int32_t)token_count;
    return ReturnLogits(opaque, output);
}

static int SetContextLength(void* opaque, int32_t length,
                            const flm_error_sink_v1* error) {
    (void)error;
    if (!opaque) return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    ((fake_backend*)opaque)->context_length = length;
    return FLM_MODEL_EXTENSION_STATUS_OK;
}

static int UpdateMaxLength(void* opaque, uint32_t length,
                           const flm_error_sink_v1* error) {
    (void)opaque;
    (void)length;
    (void)error;
    return FLM_MODEL_EXTENSION_STATUS_OK;
}

static int ClearContext(void* opaque, const flm_error_sink_v1* error) {
    return SetContextLength(opaque, 0, error);
}

static int GetContextLength(void* opaque, int32_t* length,
                            const flm_error_sink_v1* error) {
    (void)error;
    if (!opaque || !length)
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    *length = ((fake_backend*)opaque)->context_length;
    return FLM_MODEL_EXTENSION_STATUS_OK;
}

static int Checkpoint(void* opaque, int32_t* checkpoint,
                      const flm_error_sink_v1* error) {
    fake_backend* backend = (fake_backend*)opaque;
    (void)error;
    if (!backend || !checkpoint)
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    backend->checkpoint_length = backend->context_length;
    *checkpoint = backend->checkpoint_length;
    return FLM_MODEL_EXTENSION_STATUS_OK;
}

static int Restore(void* opaque, int32_t* length,
                   const flm_error_sink_v1* error) {
    fake_backend* backend = (fake_backend*)opaque;
    (void)error;
    if (!backend || !length)
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    backend->context_length = backend->checkpoint_length;
    *length = backend->context_length;
    return FLM_MODEL_EXTENSION_STATUS_OK;
}

static const flm_causal_lm_v1 kCausalLm = {
    sizeof(flm_causal_lm_v1),
    Forward,
    Prefill,
    SetContextLength,
    UpdateMaxLength,
    ClearContext,
    GetContextLength,
};

static const flm_backend_state_v1 kState = {
    sizeof(flm_backend_state_v1),
    Checkpoint,
    Restore,
};

static const void* QueryCapability(
    void* object, uint32_t capability, uint32_t capability_abi) {
    (void)object;
    if (capability == FLM_BACKEND_CAPABILITY_STATE &&
        capability_abi == FLM_BACKEND_STATE_ABI_V1) {
        return &kState;
    }
    return NULL;
}

static void DestroyBackend(void* object) {
    free(object);
}

static int CreateBackend(
    void* factory_context, const flm_backend_create_context_v1* context,
    flm_backend_instance_v1* instance, const flm_error_sink_v1* error) {
    fake_backend* backend;
    size_t instance_capacity;
    static const char detail[] = "fake stable C backend";
    (void)error;
    if (!factory_context || !context ||
        context->struct_size < FLM_BACKEND_CREATE_CONTEXT_V1_MIN_SIZE ||
        !context->runtime_device_handle || !context->model_path.data ||
        !instance ||
        instance->struct_size < FLM_BACKEND_INSTANCE_V1_MIN_SIZE) {
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    }
    instance_capacity = instance->struct_size;
    backend = (fake_backend*)calloc(1, sizeof(fake_backend));
    if (!backend) return FLM_MODEL_EXTENSION_STATUS_ERROR;
    backend->logits[0] = 1;
    backend->logits[1] = 2;
    backend->logits[2] = 3;
    backend->logits[3] = 4;
    instance->struct_size =
        instance_capacity < sizeof(*instance)
            ? instance_capacity
            : sizeof(*instance);
    instance->object = backend;
    instance->destroy = DestroyBackend;
    instance->causal_lm = &kCausalLm;
    instance->detail.data = detail;
    instance->detail.size = sizeof(detail) - 1;
    if (FLM_TEST_EXTENSION_MODE !=
            FLM_TEST_EXTENSION_MODE_NO_CAPABILITIES &&
        instance_capacity >=
        FLM_BACKEND_INSTANCE_V1_QUERY_CAPABILITY_SIZE) {
        instance->query_capability = QueryCapability;
    }
    return FLM_MODEL_EXTENSION_STATUS_OK;
}

#if FLM_TEST_EXTENSION_MODE == FLM_TEST_EXTENSION_MODE_COMPLETE
typedef struct fake_frontend {
    int32_t tokens[2];
    char text[16];
    int payload;
} fake_frontend;

static int LoadFrontend(
    void* object, const flm_frontend_load_context_v1* context,
    flm_frontend_traits_v1* traits, const flm_error_sink_v1* error) {
    static const int32_t eos[] = {3};
    static const char format[] = "content_only";
    (void)error;
    if (!object || !context || !traits ||
        context->struct_size < FLM_FRONTEND_LOAD_CONTEXT_V1_MIN_SIZE ||
        traits->struct_size < FLM_FRONTEND_TRAITS_V1_MIN_SIZE)
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    traits->flags = FLM_FRONTEND_TRAIT_SUPPORTS_LOGPROBS;
    traits->vocabulary_size = 4;
    traits->has_bos_token_id = 0;
    traits->eos_token_ids = eos;
    traits->eos_token_ids_size = 1;
    traits->chat_format.data = format;
    traits->chat_format.size = sizeof(format) - 1;
    return FLM_MODEL_EXTENSION_STATUS_OK;
}

static void ReleasePrepared(void* context) {
    (void)context;
}

static int Prepare(
    void* object, flm_span_v1 request_json,
    const flm_cancellation_v1* cancellation,
    flm_prepared_request_v1* request, const flm_error_sink_v1* error) {
    fake_frontend* frontend = (fake_frontend*)object;
    (void)request_json;
    if (cancellation && cancellation->is_cancelled &&
        cancellation->is_cancelled(cancellation->context))
        return FLM_MODEL_EXTENSION_STATUS_CANCELLED;
    (void)error;
    if (!frontend || !request ||
        request->struct_size < FLM_PREPARED_REQUEST_V1_MIN_SIZE)
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    frontend->tokens[0] = 1;
    frontend->tokens[1] = 2;
    frontend->payload = 7;
    request->tokens.data = frontend->tokens;
    request->tokens.size = 2;
    request->payload.struct_size = sizeof(request->payload);
    request->payload.object = &frontend->payload;
    request->has_payload = 1;
    request->first_payload_token_count = 2;
    request->release_context = frontend;
    request->release = ReleasePrepared;
    return FLM_MODEL_EXTENSION_STATUS_OK;
}

static int Encode(
    void* object, flm_span_v1 text, flm_i32_view_v1* tokens,
    const flm_error_sink_v1* error) {
    fake_frontend* frontend = (fake_frontend*)object;
    (void)text;
    (void)error;
    if (!frontend || !tokens)
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    tokens->data = frontend->tokens;
    tokens->size = 2;
    return FLM_MODEL_EXTENSION_STATUS_OK;
}

static int DecodeToken(
    void* object, int32_t token, flm_span_v1* text,
    const flm_error_sink_v1* error) {
    fake_frontend* frontend = (fake_frontend*)object;
    (void)error;
    if (!frontend || !text)
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    frontend->text[0] = (char)('0' + token);
    text->data = frontend->text;
    text->size = 1;
    return FLM_MODEL_EXTENSION_STATUS_OK;
}

static int ApplyTemplate(
    void* object, flm_span_v1 messages, flm_span_v1 tools,
    uint32_t continued, flm_chat_template_result_v1* result,
    const flm_error_sink_v1* error) {
    (void)object;
    (void)tools;
    (void)continued;
    (void)error;
    if (!result ||
        result->struct_size < FLM_CHAT_TEMPLATE_RESULT_V1_MIN_SIZE)
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    result->text = messages;
    result->prompt_tail = continued ? messages : (flm_span_v1){NULL, 0};
    return FLM_MODEL_EXTENSION_STATUS_OK;
}

static int CreateFrontend(
    void* factory_context, const flm_frontend_create_context_v1* context,
    flm_model_frontend_v1* frontend, const flm_error_sink_v1* error) {
    fake_frontend* object;
    size_t capacity;
    (void)factory_context;
    (void)error;
    if (!context ||
        context->struct_size < FLM_FRONTEND_CREATE_CONTEXT_V1_MIN_SIZE ||
        !frontend ||
        frontend->struct_size < FLM_MODEL_FRONTEND_V1_MIN_SIZE)
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    capacity = frontend->struct_size;
    object = (fake_frontend*)calloc(1, sizeof(*object));
    if (!object) return FLM_MODEL_EXTENSION_STATUS_ERROR;
    frontend->struct_size =
        capacity < sizeof(*frontend) ? capacity : sizeof(*frontend);
    frontend->object = object;
    frontend->destroy = free;
    frontend->load = LoadFrontend;
    frontend->prepare = Prepare;
    frontend->encode = Encode;
    frontend->decode_token = DecodeToken;
    frontend->apply_chat_template = ApplyTemplate;
    return FLM_MODEL_EXTENSION_STATUS_OK;
}
#endif

static void DestroyExtensionContext(void* context) {
    FILE* marker;
    marker = fopen(FLM_TEST_EXTENSION_CLEANUP_PATH, "w");
    if (marker) {
        fputs("destroyed\n", marker);
        fclose(marker);
    }
    free(context);
}

static int RegisterExtension(
    const flm_model_extension_registrar_v1* registrar,
    const flm_error_sink_v1* error) {
    void* state;
    int status;
    const flm_backend_traits_v1 traits = {
        sizeof(flm_backend_traits_v1),
        FLM_BACKEND_TRAIT_REQUIRES_NPU_DEVICE |
            FLM_BACKEND_TRAIT_FORWARDS_PAST_EOS,
        4096,
        0,
        0,
        0,
        NULL,
        0,
    };
    (void)error;
    if (!registrar ||
        registrar->struct_size <
            FLM_MODEL_EXTENSION_REGISTRAR_V1_MIN_SIZE ||
        !registrar->set_extension_context ||
        !registrar->register_backend_provider) {
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    }
#if FLM_TEST_EXTENSION_MODE == \
    FLM_TEST_EXTENSION_MODE_REGISTRATION_FAILURE
    {
        static const char message[] = "fake registration failure";
        if (error && error->set_error) {
            error->set_error(
                error->context, message, sizeof(message) - 1);
        }
        return FLM_MODEL_EXTENSION_STATUS_ERROR;
    }
#endif
    state = malloc(1);
    if (!state) return FLM_MODEL_EXTENSION_STATUS_ERROR;
    status = registrar->set_extension_context(
        registrar->host_context, state, DestroyExtensionContext);
    if (status != 0) {
        free(state);
        return status;
    }
    status = registrar->register_backend_provider(
        registrar->host_context, FLM_TEST_FAMILY, "flm",
        "plugin-provider", &traits, CreateBackend, state);
    if (status != 0) return status;
#if FLM_TEST_EXTENSION_MODE == FLM_TEST_EXTENSION_MODE_COMPLETE
    if (registrar->struct_size <
            FLM_MODEL_EXTENSION_REGISTRAR_V1_FRONTEND_SIZE ||
        !registrar->register_frontend_provider)
        return FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED;
    return registrar->register_frontend_provider(
        registrar->host_context, FLM_TEST_FAMILY, "plugin-provider",
        CreateFrontend, state);
#else
    return FLM_MODEL_EXTENSION_STATUS_OK;
#endif
}

static const flm_model_extension_v1 kExtension = {
    sizeof(flm_model_extension_v1),
    FLM_MODEL_EXTENSION_ABI_V1,
    FLM_TEST_NPU_RUNTIME,
    "fake-model-extension",
    "1.0.0",
    RegisterExtension,
};

FLM_MODEL_EXTENSION_EXPORT
const flm_model_extension_v1* flm_model_extension_query_v1(void) {
    return flm_test_model_extension_dependency() ? &kExtension : NULL;
}
