/// \file model_extension_embedding.cpp
/// \brief Adapt complete embedding providers to AutoEmbeddingModel.
#include "AutoEmbeddingModel/auto_embedding_model.hpp"
#include "AutoModel/model_extension_frontend.hpp"
#include "../AutoModel/model_extension_detail.hpp"

namespace flm::frontend {
namespace {
using namespace flm::extension::detail;

void Check(int status, const ErrorCapture &error, bool request = false) {
    if (request && (status == FLM_MODEL_EXTENSION_STATUS_INVALID_ARGUMENT ||
                    status == FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED))
        throw std::invalid_argument("Embedding provider: " + error.message);
    if (status != FLM_MODEL_EXTENSION_STATUS_OK)
        throw std::runtime_error("Embedding provider: " + error.message);
}

class ExtensionEmbeddingModel final : public AutoEmbeddingModel {
  public:
    ExtensionEmbeddingModel(flm_rt::device *device, flm_embedding_model_v1 api,
                            std::shared_ptr<void> lifetime)
        : AutoEmbeddingModel(device), api_(api), lifetime_(std::move(lifetime)) {}
    ~ExtensionEmbeddingModel() override {
        destroy_provider_object(api_.destroy, api_.object);
    }
    void load_model(std::string path, json info, bool preemption) override {
        is_model_loaded = false;
        const std::string metadata = info.dump();
        ErrorCapture error;
        auto sink = error.sink();
        Check(api_.load(api_.object, span(path), span(metadata), preemption, &sink),
              error);
        model_path = std::move(path);
        is_model_loaded = true;
    }
    EmbeddingResult embed(const json &request, std::function<bool()> cancelled) override {
        if (!is_model_loaded)
            throw std::runtime_error("Embedding model is not loaded");
        flm_embedding_result_v1 output{};
        output.struct_size = sizeof(output);
        const auto serialized = request.dump();
        const flm_cancellation_v1 cancellation{&cancelled, [](void *context) noexcept {
            try { return (*static_cast<std::function<bool()> *>(context))() ? 1 : 0; }
            catch (...) { return 1; }
        }};
        ErrorCapture error;
        auto sink = error.sink();
        Check(api_.embed(api_.object, span(serialized), &cancellation, &output, &sink), error, true);
        if (output.struct_size < FLM_EMBEDDING_RESULT_V1_MIN_SIZE || !output.data || !output.size)
            throw std::runtime_error(
                "Embedding provider returned an invalid result");
        return {{output.data, output.data + output.size}, output.prompt_tokens};
    }

  private:
    flm_embedding_model_v1 api_;
    std::shared_ptr<void> lifetime_;
};
} // namespace

EmbeddingFactory
make_extension_embedding_factory(ExtensionFrontendRegistration registration,
                                 std::string root, std::string assets,
                                 std::shared_ptr<void> lifetime) {
    return [registration, root = std::move(root), assets = std::move(assets),
            lifetime = std::move(lifetime)](
               void *raw_device) -> std::unique_ptr<AutoEmbeddingModel> {
        auto *device = static_cast<flm_rt::device *>(raw_device);
        const flm_frontend_create_context_v1 context{
            sizeof(context), span(root), span(assets),
            native_runtime_device_handle(device), host_npu_runtime()};
        flm_embedding_model_v1 instance{};
        instance.struct_size = sizeof(instance);
        ErrorCapture error;
        auto sink = error.sink();
        try {
            Check(registration.create_embedding(registration.factory_context,
                                                &context, &instance, &sink), error);
            if (instance.struct_size < FLM_EMBEDDING_MODEL_V1_MIN_SIZE ||
                !instance.object || !instance.destroy || !instance.load ||
                !instance.embed)
                throw std::runtime_error(
                    "Embedding factory returned an incomplete model");
            return std::make_unique<ExtensionEmbeddingModel>(device, instance,
                                                            lifetime);
        } catch (...) {
            destroy_provider_object(instance.destroy, instance.object);
            throw;
        }
    };
}
} // namespace flm::frontend
