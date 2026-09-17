/// \file flm_backend.hpp
/// \brief The flm backend, shared by every family that ships one
/// \note The flm backend runs FastFlowLM's own NPU kernels, and every one of
///       those engines is built the same way -- construct, load the Q4NX
///       weights, clear the context -- so one template covers them all.
#pragma once

#include "AutoModel/model_backend.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace flm::backend {

/// \brief the flm backend for a FastFlowLM NPU engine
/// \tparam Engine the concrete engine type, e.g. phi4_npu
/// \note load_weights is called on Engine*, not on causal_lm*. The pure virtual
///       stays in the frozen causal_lm.hpp for ABI, but nothing here depends on
///       it, so a backend whose engine loads its weights some other way is under
///       no obligation to pretend otherwise.
template <class Engine>
class FlmBackend final : public ModelBackend {
public:
    explicit FlmBackend(const BackendContext& context) {
        if (!context.config) {
            throw std::runtime_error("flm backend needs an LM_Config");
        }
        if (!context.npu) {
            throw std::runtime_error("flm backend needs an NPU instance");
        }

        // Scoped: the packed weights are copied into the engine, and the
        // several hundred MB they occupy are freed before load_model returns.
        Q4NX q4nx(context.model_path);
        auto engine = std::make_unique<Engine>(
            *context.config, context.npu,
            static_cast<int>(context.context_length));
        engine->load_weights(q4nx);
        engine->clear_context();
        engine_ = std::move(engine);
    }

    causal_lm& engine() override { return *engine_; }

    std::string id() const override { return kFlmBackendId; }

private:
    std::unique_ptr<Engine> engine_;
};

/// \brief a factory building FlmBackend<Engine>
/// \tparam Engine the concrete engine type
/// \return a factory suitable for BackendRegistry::register_backend
template <class Engine>
BackendFactory flm_factory() {
    return [](const BackendContext& context) -> std::unique_ptr<ModelBackend> {
        return std::make_unique<FlmBackend<Engine>>(context);
    };
}

}  // namespace flm::backend
