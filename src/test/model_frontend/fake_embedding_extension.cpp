/// \file fake_embedding_extension.cpp
/// \brief Embedding provider for the dynamic-loader tests.
#define FLM_MODEL_EXTENSION_IMPLEMENTATION
#include "AutoModel/model_extension.h"
#include <stdlib.h>
#include <nlohmann/json.hpp>
#include <string.h>

static int EndsWith(flm_span_v1 value, const char *suffix) {
    size_t n = strlen(suffix);
    return value.size >= n && !memcmp((const char *)value.data + value.size - n, suffix, n);
}
static int Load(void *object, flm_span_v1 path, flm_span_v1 info, uint32_t preemption,
                const flm_error_sink_v1 *error) {
    if (EndsWith(path, "missing")) abort(); /* A failed download must never reach load. */
    if (EndsWith(path, "fail-load")) return FLM_MODEL_EXTENSION_STATUS_ERROR;
    if (EndsWith(path, "unsupported-load")) return FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED;
    if (EndsWith(path, "invalid-load")) return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    ((float *)object)[2] = EndsWith(path, "second") ? 2.0f : 0.5f;
    (void)info;
    (void)error;
    return path.size && !preemption ? 0 : FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED;
}
static int Embed(void *object, flm_span_v1 request, const flm_cancellation_v1 *cancellation,
                 flm_embedding_result_v1 *output, const flm_error_sink_v1 *error) {
    if (cancellation && cancellation->is_cancelled(cancellation->context))
        return FLM_MODEL_EXTENSION_STATUS_CANCELLED;
    (void)error;
    if (output->struct_size < FLM_EMBEDDING_RESULT_V1_MIN_SIZE) return FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED;
    try {
        const auto options = nlohmann::json::parse((const char *)request.data,
                                                 (const char *)request.data + request.size);
        const auto& input = options.at("input");
        const auto text = (input.is_object() ? input.at("text") : input).get<std::string>();
        const auto task = options.value("task", std::string());
        auto dimensions = options.value("dimensions", nlohmann::json(3));
        if (!dimensions.is_number_integer() || dimensions.get<int64_t>() < 1 || dimensions.get<int64_t>() > 3)
            return FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED;
        float *values = (float *)object;
        values[0] = (float)text.size();
        if (options.contains("title")) values[0] += options.at("title").get<std::string>().size();
        if (task.empty()) values[1] = 3.0f;
        else if (task == "query") values[1] = 1.0f;
        else if (task == "document") values[1] = 2.0f;
        else if (task == "provider-task") values[1] = 4.0f;
        else return FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED;
        output->struct_size = text == "short-result" ? 0 : sizeof(*output);
        output->data = text.empty() ? NULL : values;
        output->size = text == "zero-output" ? 0 : dimensions.get<size_t>();
        output->prompt_tokens = text.size();
        return 0;
    } catch (const nlohmann::json::exception&) {
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    }
}
static int Create(void *factory, const flm_frontend_create_context_v1 *context,
                  flm_embedding_model_v1 *model, const flm_error_sink_v1 *error) {
    (void)factory;
    (void)error;
    if (model->struct_size < FLM_EMBEDDING_MODEL_V1_MIN_SIZE)
        return 1;
    model->struct_size = sizeof(*model);
    model->object = calloc(3, sizeof(float));
    model->destroy = free;
    model->load = Load;
    model->embed = EndsWith(context->asset_root, "incomplete") ? NULL : Embed;
    return model->object ? 0 : 1;
}
static int Register(const flm_model_extension_registrar_v1 *registrar,
                    const flm_error_sink_v1 *error) {
    (void)error;
    if (registrar->struct_size < FLM_MODEL_EXTENSION_REGISTRAR_V1_EMBEDDING_SIZE)
        return FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED;
    return registrar->register_embedding_provider(
        registrar->host_context, "test-embedding", "test-embedding", Create, NULL);
}
static const flm_model_extension_v1 extension = {sizeof(flm_model_extension_v1),
                                                 FLM_MODEL_EXTENSION_ABI_V1,
                                                 FLM_TEST_NPU_RUNTIME,
                                                 "test-embedding",
                                                 "1",
                                                 Register};
FLM_MODEL_EXTENSION_EXPORT const flm_model_extension_v1 *
flm_model_extension_query_v1(void) {
    return &extension;
}
