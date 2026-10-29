/// \file model_extension_backend.cpp
/// \brief Adapts stable-C provider engines to the host C++ backend interfaces
#include "AutoModel/model_extension_backend.hpp"

#include "AutoModel/model_backend.hpp"
#include "model_extension_detail.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace flm::frontend {
namespace {

using flm::extension::detail::copy_span;
using flm::extension::detail::destroy_provider_object;
using flm::extension::detail::ErrorCapture;
using flm::extension::detail::host_npu_runtime;
using flm::extension::detail::native_runtime_device_handle;
using flm::extension::detail::span;

[[noreturn]] void ThrowProviderError(
    const char* operation, int status, const ErrorCapture& error) {
    throw std::runtime_error(
        "Extension backend " + std::string(operation) + " failed with status " +
        std::to_string(status) +
        (error.message.empty() ? std::string() : ": " + error.message));
}

void CheckStatus(const char* operation, int status,
                 const ErrorCapture& error) {
    if (status != FLM_MODEL_EXTENSION_STATUS_OK) {
        ThrowProviderError(operation, status, error);
    }
}

buffer<bf16> BufferFromView(const char* operation, flm_bf16_view_v1 view) {
    static_assert(sizeof(bf16) == sizeof(std::uint16_t));
    if (!view.data && view.size != 0) {
        throw std::runtime_error(
            "Extension backend " + std::string(operation) +
            " returned a null BF16 view");
    }
    return buffer<bf16>(
        reinterpret_cast<bf16*>(const_cast<std::uint16_t*>(view.data)),
        view.size);
}

/// Adapts a provider's causal core and optional capability tables to causal_lm.
class ProviderCausalLm final : public causal_lm {
public:
    ProviderCausalLm(
        std::string provider, bool accepts_private_payloads, void* object,
        const flm_causal_lm_v1* api,
        flm_backend_query_capability_v1 query_capability)
        : provider_(std::move(provider)),
          accepts_private_payloads_(accepts_private_payloads),
          object_(object) {
        std::memcpy(
            &api_, api, std::min(api->struct_size, sizeof(api_)));
        LoadCapabilities(query_capability);
    }

    buffer<bf16> forward(int token) override {
        flm_bf16_view_v1 output{};
        ErrorCapture error;
        auto sink = error.sink();
        CheckStatus(
            "forward", api_.forward(object_, token, &output, &sink), error);
        return BufferFromView("forward", output);
    }

    buffer<bf16> prefill(std::vector<int>& tokens,
                         void* payload = nullptr) override {
        static_assert(sizeof(int) == sizeof(std::int32_t));
        if (payload && !accepts_private_payloads_) {
            throw std::runtime_error(
                "Extension backend provider '" + provider_ +
                "' cannot consume a private payload from another frontend "
                "provider");
        }
        const auto* stable_payload =
            static_cast<const flm_prefill_payload_v1*>(payload);
        if (stable_payload &&
            (stable_payload->struct_size < FLM_PREFILL_PAYLOAD_V1_MIN_SIZE ||
             !stable_payload->object)) {
            throw std::runtime_error(
                "Extension frontend supplied an incomplete opaque payload");
        }
        if (stable_payload &&
            copy_span(stable_payload->provider) != provider_) {
            throw std::runtime_error(
                "Extension frontend payload provider '" +
                copy_span(stable_payload->provider) +
                "' does not match backend provider '" + provider_ + "'");
        }
        flm_bf16_view_v1 output{};
        ErrorCapture error;
        auto sink = error.sink();
        CheckStatus(
            "prefill",
            api_.prefill(
                object_, reinterpret_cast<const std::int32_t*>(tokens.data()),
                tokens.size(), stable_payload, &output, &sink),
            error);
        return BufferFromView("prefill", output);
    }

    void set_context_length(int length) override {
        ErrorCapture error;
        auto sink = error.sink();
        CheckStatus(
            "set_context_length",
            api_.set_context_length(object_, length, &sink), error);
    }

    void load_weights(Q4NX&) override {
        throw std::runtime_error(
            "Extension backends must load their own weights during creation");
    }

    void update_max_length(std::uint32_t length) override {
        ErrorCapture error;
        auto sink = error.sink();
        CheckStatus(
            "update_max_length",
            api_.update_max_length(object_, length, &sink), error);
    }

    void clear_context() override {
        ErrorCapture error;
        auto sink = error.sink();
        CheckStatus(
            "clear_context", api_.clear_context(object_, &sink), error);
    }

    buffer<bf16> get_k_cache(int, int) override {
        throw std::runtime_error(
            "Extension backend does not support get_k_cache");
    }

    buffer<bf16> get_v_cache(int, int) override {
        throw std::runtime_error(
            "Extension backend does not support get_v_cache");
    }

    int get_current_context_length() override {
        std::int32_t length = 0;
        ErrorCapture error;
        auto sink = error.sink();
        CheckStatus(
            "get_current_context_length",
            api_.get_current_context_length(object_, &length, &sink), error);
        return length;
    }

    int checkpoint() override {
        if (!state_.checkpoint) {
            throw std::runtime_error(
                "Extension backend does not support checkpoint");
        }
        std::int32_t checkpoint = 0;
        ErrorCapture error;
        auto sink = error.sink();
        CheckStatus(
            "checkpoint",
            state_.checkpoint(object_, &checkpoint, &sink), error);
        return checkpoint;
    }

    int restore() override {
        if (!state_.restore) {
            throw std::runtime_error(
                "Extension backend does not support restore");
        }
        std::int32_t length = 0;
        ErrorCapture error;
        auto sink = error.sink();
        CheckStatus(
            "restore", state_.restore(object_, &length, &sink), error);
        return length;
    }

    bool supports_speculation() const override {
        if (!speculation_.supports_speculation) return false;
        try {
            return speculation_.supports_speculation(object_) != 0;
        } catch (...) {
            return false;
        }
    }

    std::vector<int> speculate(int last_token, int max_draft) override {
        if (!speculation_.speculate) return {};
        flm_i32_view_v1 output{};
        ErrorCapture error;
        auto sink = error.sink();
        CheckStatus(
            "speculate",
            speculation_.speculate(
                object_, last_token, max_draft, &output, &sink),
            error);
        if (!output.data && output.size != 0) {
            throw std::runtime_error(
                "Extension backend speculate returned a null token view");
        }
        if (output.size == 0) return {};
        return {output.data, output.data + output.size};
    }

    std::uint64_t last_speculation_prime_us() const override {
        if (!speculation_.last_speculation_prime_us) return 0;
        try {
            return speculation_.last_speculation_prime_us(object_);
        } catch (...) {
            return 0;
        }
    }

    bool poisoned() const noexcept {
        if (!health_.poisoned) return false;
        try {
            return health_.poisoned(object_) != 0;
        } catch (...) {
            return true;
        }
    }

    bool supports_state() const noexcept {
        return state_.checkpoint && state_.restore;
    }

private:
    template <typename Capability>
    void LoadCapability(
        flm_backend_query_capability_v1 query, std::uint32_t id,
        std::uint32_t abi, std::size_t minimum_size, Capability& output) {
        const void* value = nullptr;
        try {
            value = query(object_, id, abi);
        } catch (...) {
            throw std::runtime_error(
                "Extension backend capability query allowed an exception "
                "to cross its C boundary");
        }
        if (!value) return;
        const auto* capability = static_cast<const Capability*>(value);
        if (capability->struct_size < minimum_size) {
            throw std::runtime_error(
                "Extension backend returned an incomplete capability table");
        }
        std::memcpy(
            &output, capability,
            std::min(capability->struct_size, sizeof(output)));
    }

    void LoadCapabilities(flm_backend_query_capability_v1 query) {
        if (!query) return;
        LoadCapability(
            query, FLM_BACKEND_CAPABILITY_STATE,
            FLM_BACKEND_STATE_ABI_V1,
            FLM_BACKEND_STATE_V1_MIN_SIZE, state_);
        LoadCapability(
            query, FLM_BACKEND_CAPABILITY_SPECULATION,
            FLM_BACKEND_SPECULATION_ABI_V1,
            FLM_BACKEND_SPECULATION_V1_MIN_SIZE, speculation_);
        LoadCapability(
            query, FLM_BACKEND_CAPABILITY_HEALTH,
            FLM_BACKEND_HEALTH_ABI_V1,
            FLM_BACKEND_HEALTH_V1_MIN_SIZE, health_);
        if ((state_.struct_size &&
             (!state_.checkpoint || !state_.restore)) ||
            (speculation_.struct_size &&
             (!speculation_.supports_speculation ||
              !speculation_.speculate)) ||
            (health_.struct_size && !health_.poisoned)) {
            throw std::runtime_error(
                "Extension backend returned an incomplete capability table");
        }
    }

    std::string provider_;
    bool accepts_private_payloads_ = false;
    void* object_;
    flm_causal_lm_v1 api_{};
    flm_backend_state_v1 state_{};
    flm_backend_speculation_v1 speculation_{};
    flm_backend_health_v1 health_{};
};

/// Adapts an opaque extension engine to the host's ModelBackend interface.
class ProviderBackend final : public flm::backend::ModelBackend {
public:
    ProviderBackend(
        std::string id, std::string provider,
        bool accepts_private_payloads,
        flm_backend_instance_v1 instance,
        flm::backend::BackendTraits traits,
        std::uint32_t max_decode_length,
        std::optional<std::vector<int>> forced_eos_ids,
        std::optional<int> forced_bos_id, std::shared_ptr<void> lifetime)
        : id_(std::move(id)),
          object_(instance.object),
          destroy_(instance.destroy),
          engine_(std::move(provider), accepts_private_payloads,
                  instance.object, instance.causal_lm,
                  instance.struct_size >=
                          FLM_BACKEND_INSTANCE_V1_QUERY_CAPABILITY_SIZE
                      ? instance.query_capability
                      : nullptr),
          detail_(copy_span(instance.detail)),
          traits_(traits),
          max_decode_length_(max_decode_length),
          forced_eos_ids_(std::move(forced_eos_ids)),
          forced_bos_id_(forced_bos_id),
          lifetime_(std::move(lifetime)) {}

    ~ProviderBackend() override {
        destroy_provider_object(destroy_, object_);
    }

    causal_lm& engine() override { return engine_; }
    std::string id() const override { return id_; }
    std::string detail() const override { return detail_; }
    std::uint32_t max_decode_length() const override {
        return max_decode_length_;
    }
    bool supports_preemption() const override {
        return traits_.supports_preemption;
    }
    bool supports_state() const override {
        return engine_.supports_state();
    }
    bool forwards_past_eos() const override {
        return forwards_past_eos_;
    }
    bool aligns_prefill_chunks() const override {
        return aligns_prefill_chunks_;
    }
    bool poisoned() const noexcept override { return engine_.poisoned(); }
    std::optional<std::vector<int>> forced_eos_ids() const override {
        return forced_eos_ids_;
    }
    std::optional<int> forced_bos_id() const override {
        return forced_bos_id_;
    }

    void set_behavior(bool forwards_past_eos, bool aligns_prefill_chunks) {
        forwards_past_eos_ = forwards_past_eos;
        aligns_prefill_chunks_ = aligns_prefill_chunks;
    }

private:
    std::string id_;
    void* object_;
    void (*destroy_)(void*);
    ProviderCausalLm engine_;
    std::string detail_;
    flm::backend::BackendTraits traits_;
    std::uint32_t max_decode_length_;
    std::optional<std::vector<int>> forced_eos_ids_;
    std::optional<int> forced_bos_id_;
    bool forwards_past_eos_ = true;
    bool aligns_prefill_chunks_ = false;
    std::shared_ptr<void> lifetime_;
};

void ValidateInstance(
    const flm_backend_instance_v1& instance,
    const std::string& identity) {
    if (instance.struct_size < FLM_BACKEND_INSTANCE_V1_MIN_SIZE ||
        instance.struct_size > sizeof(instance) ||
        !instance.object ||
        !instance.destroy || !instance.causal_lm ||
        instance.causal_lm->struct_size < FLM_CAUSAL_LM_V1_MIN_SIZE ||
        !instance.causal_lm->forward || !instance.causal_lm->prefill ||
        !instance.causal_lm->set_context_length ||
        !instance.causal_lm->update_max_length ||
        !instance.causal_lm->clear_context ||
        !instance.causal_lm->get_current_context_length) {
        throw std::runtime_error(
            "Extension backend factory for '" + identity +
            "' returned an incomplete stable C engine");
    }
}

}  // namespace

std::vector<ExtensionBackendKey> register_extension_backends(
    std::vector<ExtensionBackendRegistration> registrations,
    std::string extension_root, std::string asset_root,
    std::shared_ptr<void> provider_lifetime) {
    auto& registry = flm::backend::BackendRegistry::instance();
    for (const auto& registration : registrations) {
        if (registry.has_provider(
                registration.family, registration.backend,
                registration.provider)) {
            throw std::runtime_error(
                "backend provider '" + registration.provider +
                "' is already registered for '" + registration.family + "/" +
                registration.backend + "'");
        }
    }

    std::vector<ExtensionBackendKey> inserted;
    inserted.reserve(registrations.size());
    try {
        for (auto& registration : registrations) {
            const flm::backend::BackendTraits traits{
                false,
                (registration.traits.flags &
                 FLM_BACKEND_TRAIT_SUPPORTS_PREEMPTION) != 0,
                registration.traits.max_context_length,
            };
            const auto behavior_flags = registration.traits.flags;
            const auto max_decode_length =
                registration.traits.max_decode_length;
            const auto forced_eos_ids = registration.forced_eos_ids;
            const auto forced_bos_id = registration.traits.has_forced_bos_id
                ? std::optional<int>(registration.traits.forced_bos_id)
                : std::nullopt;
            const auto create = registration.create;
            void* const factory_context = registration.factory_context;
            const std::string family = registration.family;
            const std::string backend = registration.backend;
            const std::string provider = registration.provider;
            registry.register_provider(
                family, backend, provider,
                [create, factory_context, provider_lifetime, extension_root,
                 asset_root, family, backend, provider, traits,
                 behavior_flags, max_decode_length, forced_eos_ids,
                 forced_bos_id](
                    const flm::backend::BackendContext& backend_context) {
                if ((behavior_flags &
                     FLM_BACKEND_TRAIT_REQUIRES_NPU_DEVICE) != 0 &&
                    !backend_context.device) {
                    throw std::runtime_error(
                        "Extension backend '" + family + "/" + backend + "/" +
                        provider + "' requires an NPU device");
                }
                const std::string model_info =
                    backend_context.model_info.dump();
                const std::string tokenizer_config =
                    backend_context.tokenizer_config
                    ? backend_context.tokenizer_config->dump()
                    : std::string();
                flm_backend_instance_v1 instance{};
                instance.struct_size = sizeof(instance);
                const flm_backend_create_context_v1 context{
                    sizeof(flm_backend_create_context_v1),
                    span(backend_context.model_path),
                    span(model_info),
                    span(tokenizer_config),
                    span(extension_root),
                    span(asset_root),
                    native_runtime_device_handle(backend_context.device),
                    host_npu_runtime(),
                    backend_context.context_length,
                    backend_context.enable_preemption ? 1u : 0u,
                };
                ErrorCapture error;
                auto sink = error.sink();
                int status = FLM_MODEL_EXTENSION_STATUS_ERROR;
                try {
                    status = create(
                        factory_context, &context, &instance, &sink);
                } catch (...) {
                    destroy_provider_object(
                        instance.destroy, instance.object);
                    throw std::runtime_error(
                        "Extension backend factory for '" + family + "/" +
                        backend + "/" + provider +
                        "' allowed an exception to cross its C boundary");
                }
                const std::string identity =
                    family + "/" + backend + "/" + provider;
                if (status != FLM_MODEL_EXTENSION_STATUS_OK) {
                    destroy_provider_object(
                        instance.destroy, instance.object);
                    ThrowProviderError(
                        ("factory for '" + identity + "'").c_str(),
                        status, error);
                }
                try {
                    ValidateInstance(instance, identity);
                    auto result = std::make_unique<ProviderBackend>(
                        backend, provider,
                        backend_context.frontend_provider == provider,
                        instance, traits,
                        max_decode_length,
                        forced_eos_ids.empty()
                            ? std::nullopt
                            : std::optional<std::vector<int>>(
                                  std::vector<int>(
                                      forced_eos_ids.begin(),
                                      forced_eos_ids.end())),
                        forced_bos_id, provider_lifetime);
                    result->set_behavior(
                        (behavior_flags &
                         FLM_BACKEND_TRAIT_FORWARDS_PAST_EOS) != 0,
                        (behavior_flags &
                         FLM_BACKEND_TRAIT_ALIGNS_PREFILL_CHUNKS) != 0);
                    return result;
                } catch (...) {
                    destroy_provider_object(
                        instance.destroy, instance.object);
                    throw;
                }
                },
                traits, asset_root);
            inserted.push_back(
                ExtensionBackendKey{family, backend, provider});
        }
    } catch (...) {
        for (auto entry = inserted.rbegin(); entry != inserted.rend(); ++entry) {
            registry.unregister_provider(
                entry->family, entry->backend, entry->provider);
        }
        throw;
    }
    return inserted;
}

}  // namespace flm::frontend
