#define FLM_MODEL_EXTENSION_IMPLEMENTATION
#include "AutoModel/model_extension.h"
#include "models/gemma4e/flm/aie2p/gemma4e_npu.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// This integration provider always receives an explicit bundle asset root.
// Keep its source-built engine independent from the host path resolver.
namespace utils {
std::string find_xclbin_path() {
    return {};
}
}  // namespace utils

namespace gemma4_open {
int CreateFrontend(
    void* factory_context, const flm_frontend_create_context_v1* context,
    flm_model_frontend_v1* frontend, const flm_error_sink_v1* error);
}

namespace {

std::string String(flm_span_v1 value) {
    return value.data
        ? std::string(static_cast<const char*>(value.data), value.size)
        : std::string();
}

void SetError(const flm_error_sink_v1* sink, const std::string& message) {
    if (sink && sink->set_error) {
        sink->set_error(sink->context, message.data(), message.size());
    }
}

template <class Operation>
int Guard(const flm_error_sink_v1* error, Operation&& operation) {
    try {
        std::forward<Operation>(operation)();
        return FLM_MODEL_EXTENSION_STATUS_OK;
    } catch (const std::exception& exception) {
        SetError(error, exception.what());
    } catch (...) {
        SetError(error, "unknown Gemma4-open provider exception");
    }
    return FLM_MODEL_EXTENSION_STATUS_ERROR;
}

/// Owns the provider's Gemma engine and every runtime object it depends on.
class GemmaEngine {
public:
    explicit GemmaEngine(const flm_backend_create_context_v1& context) {
        const std::string model_path = String(context.model_path);
        config_.from_pretrained(model_path, String(context.asset_root));
#if defined(FLM_USE_HRX)
        device_ = std::make_unique<flm_rt::device>(0);
#else
        device_ = std::make_unique<flm_rt::device>(
            reinterpret_cast<xclDeviceHandle>(
                context.runtime_device_handle));
#endif
        npu_ = std::make_unique<npu_xclbin_manager>(
            npu_device::device_npu2, device_.get(),
            context.enable_preemption != 0);
        Q4NX weights(model_path);
        engine_ = std::make_unique<gemma4e_npu>(
            config_, npu_.get(), static_cast<int>(context.context_length));
        engine_->load_weights(weights);
        engine_->clear_context();
    }

    gemma4e_npu& engine() { return *engine_; }

    flm_bf16_view_v1 Save(buffer<bf16> output) {
        output_ = std::move(output);
        static_assert(sizeof(bf16) == sizeof(std::uint16_t));
        return {
            reinterpret_cast<const std::uint16_t*>(output_.data()),
            output_.size(),
        };
    }

private:
    LM_Config config_;
    std::unique_ptr<flm_rt::device> device_;
    std::unique_ptr<npu_xclbin_manager> npu_;
    std::unique_ptr<gemma4e_npu> engine_;
    buffer<bf16> output_;
};

int Forward(void* object, std::int32_t token, flm_bf16_view_v1* output,
            const flm_error_sink_v1* error) {
    if (!object || !output) return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    return Guard(error, [&] {
        auto& engine = *static_cast<GemmaEngine*>(object);
        *output = engine.Save(engine.engine().forward(token));
    });
}

int Prefill(void* object, const std::int32_t* tokens, std::size_t token_count,
            const flm_prefill_payload_v1* payload,
            flm_bf16_view_v1* output, const flm_error_sink_v1* error) {
    if (!object || (!tokens && token_count != 0) || !output) {
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    }
    if (payload &&
        (payload->struct_size < FLM_PREFILL_PAYLOAD_V1_MIN_SIZE ||
         !payload->object)) {
        return FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED;
    }
    return Guard(error, [&] {
        std::vector<int> ids;
        if (token_count != 0) {
            ids.assign(tokens, tokens + token_count);
        }
        auto& engine = *static_cast<GemmaEngine*>(object);
        if (!payload) {
            *output = engine.Save(engine.engine().prefill(ids));
            return;
        }
        auto* native = const_cast<gemma4e_multi_modal_payload_t*>(
            static_cast<const gemma4e_multi_modal_payload_t*>(
                payload->object));
        *output = engine.Save(engine.engine().prefill(ids, native));
    });
}

int ClearContext(void* object, const flm_error_sink_v1* error) {
    if (!object) return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    return Guard(error, [&] {
        static_cast<GemmaEngine*>(object)->engine().clear_context();
    });
}

int SetContextLength(void* object, std::int32_t length,
                     const flm_error_sink_v1* error) {
    if (!object) return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    return Guard(error, [&] {
        static_cast<GemmaEngine*>(object)->engine().set_context_length(length);
    });
}

int UpdateMaxLength(void* object, std::uint32_t length,
                    const flm_error_sink_v1* error) {
    if (!object) return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    return Guard(error, [&] {
        static_cast<GemmaEngine*>(object)->engine().update_max_length(length);
    });
}

int GetContextLength(void* object, std::int32_t* length,
                     const flm_error_sink_v1* error) {
    if (!object || !length) {
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    }
    return Guard(error, [&] {
        *length =
            static_cast<GemmaEngine*>(object)->engine()
                .get_current_context_length();
    });
}

int Checkpoint(void* object, std::int32_t* checkpoint,
               const flm_error_sink_v1* error) {
    if (!object || !checkpoint) {
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    }
    return Guard(error, [&] {
        *checkpoint =
            static_cast<GemmaEngine*>(object)->engine().checkpoint();
    });
}

int Restore(void* object, std::int32_t* length,
            const flm_error_sink_v1* error) {
    if (!object || !length) {
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    }
    return Guard(error, [&] {
        *length = static_cast<GemmaEngine*>(object)->engine().restore();
    });
}

const flm_causal_lm_v1 kCausalLm{
    sizeof(flm_causal_lm_v1),
    Forward,
    Prefill,
    SetContextLength,
    UpdateMaxLength,
    ClearContext,
    GetContextLength,
};

const flm_backend_state_v1 kState{
    sizeof(flm_backend_state_v1),
    Checkpoint,
    Restore,
};

const void* QueryCapability(
    void*, std::uint32_t capability, std::uint32_t capability_abi) {
    switch (capability) {
        case FLM_BACKEND_CAPABILITY_STATE:
            return capability_abi == FLM_BACKEND_STATE_ABI_V1
                       ? &kState
                       : nullptr;
        default:
            return nullptr;
    }
}

int CreateBackend(
    void*, const flm_backend_create_context_v1* context,
    flm_backend_instance_v1* instance, const flm_error_sink_v1* error) {
    if (!context ||
        context->struct_size < FLM_BACKEND_CREATE_CONTEXT_V1_MIN_SIZE ||
        !context->runtime_device_handle || !context->model_path.data ||
        !context->asset_root.data || !instance ||
        instance->struct_size < FLM_BACKEND_INSTANCE_V1_MIN_SIZE) {
        return FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT;
    }
    const std::size_t instance_capacity = instance->struct_size;
    return Guard(error, [&] {
        auto backend = std::make_unique<GemmaEngine>(*context);
        instance->struct_size =
            std::min(instance_capacity, sizeof(*instance));
        instance->object = backend.release();
        instance->destroy = [](void* object) {
            delete static_cast<GemmaEngine*>(object);
        };
        instance->causal_lm = &kCausalLm;
        static constexpr char kDetail[] =
            "Gemma4-open stable C integration provider";
        instance->detail = {kDetail, sizeof(kDetail) - 1};
        if (instance_capacity >=
            FLM_BACKEND_INSTANCE_V1_QUERY_CAPABILITY_SIZE) {
            instance->query_capability = QueryCapability;
        }
    });
}

int RegisterExtension(
    const flm_model_extension_registrar_v1* registrar,
    const flm_error_sink_v1* error) {
    if (!registrar ||
        registrar->struct_size <
            FLM_MODEL_EXTENSION_REGISTRAR_V1_FRONTEND_SIZE ||
        !registrar->register_backend_provider ||
        !registrar->register_frontend_provider) {
        SetError(error, "host does not support model-extension frontends");
        return FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED;
    }
    const flm_backend_traits_v1 traits{
        sizeof(flm_backend_traits_v1),
        FLM_BACKEND_TRAIT_REQUIRES_NPU_DEVICE |
            FLM_BACKEND_TRAIT_SUPPORTS_PREEMPTION |
            FLM_BACKEND_TRAIT_FORWARDS_PAST_EOS,
        0,
        0,
        0,
        0,
        nullptr,
        0,
    };
    int status = registrar->register_backend_provider(
        registrar->host_context, "gemma4-open", "flm", "gemma4-open",
        &traits, CreateBackend, nullptr);
    if (status != FLM_MODEL_EXTENSION_STATUS_OK) return status;
    return registrar->register_frontend_provider(
        registrar->host_context, "gemma4-open", "gemma4-open",
        gemma4_open::CreateFrontend, nullptr);
}

const flm_model_extension_v1 kExtension{
    sizeof(flm_model_extension_v1),
    FLM_MODEL_EXTENSION_ABI_V1,
#if defined(FLM_USE_HRX)
    FLM_MODEL_EXTENSION_NPU_RUNTIME_HRX,
#else
    FLM_MODEL_EXTENSION_NPU_RUNTIME_XRT,
#endif
    "gemma4-open",
    "1.0.0",
    RegisterExtension,
};

}  // namespace

extern "C" FLM_MODEL_EXTENSION_EXPORT
const flm_model_extension_v1* flm_model_extension_query_v1(void) {
    return &kExtension;
}
