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
#include <type_traits>
#include <utility>

namespace flm::backend {

/// \brief whether Engine is built from the frontend's device
/// \note The aie_next engines have no xclbin manager: they take the device the
///       frontend opened, and must never open one of their own. The aie2p
///       engines take an npu_xclbin_manager instead.
template <class Engine>
inline constexpr bool kTakesDevice =
    std::is_constructible_v<Engine, LM_Config, flm_rt::device*, int>;

/// \brief an engine built by an aie_next engine library loaded at run time
/// \note An aie2p build links the aie2p engines. On dev04 silicon a family
///       whose engine also has an aie_next build runs that one instead, from
///       the library <engine>_aie_next beside the linked engines: the same
///       class with the same layout, so the frontend's casts to it still hold.
///       The library stays loaded for the rest of the process.
struct NextEngine {
    causal_lm* engine = nullptr;               ///< owned by the caller
    const flm_state_api* state_api = nullptr;  ///< the library's own table
};

/// \brief whether this machine runs the aie_next engine library `library`
/// \param library e.g. "qwen3_5vl_npu_aie_next"; nullptr for a family without one
/// \note utils::get_device(), the platform the catalog picked the weights for.
bool runs_next_engine(const char* library);

/// \brief build an engine with the aie_next engine library `library`
/// \throws std::runtime_error if the library or its entry point is missing
NextEngine create_next_engine(const char* library, const LM_Config& config,
                              flm_rt::device* device, int max_length);

/// \brief the traits an flm engine registers with
/// \param next_library the family's aie_next engine library, if it has one
template <class Engine>
BackendTraits flm_traits(const char* next_library = nullptr) {
    BackendTraits traits;
    traits.needs_npu_xclbin =
        !kTakesDevice<Engine> && !runs_next_engine(next_library);
    return traits;
}

/// \brief the flm backend for a FastFlowLM NPU engine
/// \tparam Engine the concrete engine type, e.g. phi4_npu
/// \note load_weights is called on Engine*, not on causal_lm*. The pure virtual
///       stays in the frozen causal_lm.hpp for ABI, but nothing here depends on
///       it, so a backend whose engine loads its weights some other way is under
///       no obligation to pretend otherwise.
template <class Engine>
class FlmBackend final : public ModelBackend {
public:
    explicit FlmBackend(const BackendContext& context,
                        const char* next_library = nullptr) {
        if (!context.config) {
            throw std::runtime_error("flm backend needs an LM_Config");
        }
        const bool next =
            !kTakesDevice<Engine> && runs_next_engine(next_library);
        if (kTakesDevice<Engine> || next) {
            if (!context.device) {
                throw std::runtime_error("flm backend needs the NPU device");
            }
        } else if (!context.npu) {
            throw std::runtime_error("flm backend needs an NPU instance");
        }

        // Scoped: the packed weights are copied into the engine, and the
        // several hundred MB they occupy are freed before load_model returns.
        Q4NX q4nx(context.model_path);
        std::unique_ptr<Engine> engine;
        if constexpr (kTakesDevice<Engine>) {
            engine = std::make_unique<Engine>(
                *context.config, context.device,
                static_cast<int>(context.context_length));
        } else if (next) {
            const NextEngine built = create_next_engine(
                next_library, *context.config, context.device,
                static_cast<int>(context.context_length));
            std::unique_ptr<causal_lm> owned(built.engine);
            // the frontend casts the engine to Engine: make sure that holds here
            if (!dynamic_cast<Engine*>(owned.get())) {
                throw std::runtime_error(std::string(next_library) +
                                         " did not build the engine class flm links");
            }
            engine.reset(static_cast<Engine*>(owned.release()));
            state_api_ = built.state_api;
        } else {
            engine = std::make_unique<Engine>(
                *context.config, context.npu,
                static_cast<int>(context.context_length));
        }
        engine->load_weights(q4nx);
        engine->clear_context();
        engine_ = std::move(engine);
    }

    causal_lm& engine() override { return *engine_; }

    std::string id() const override { return kFlmBackendId; }

    const flm_state_api* state_api() const override { return state_api_; }

private:
    std::unique_ptr<Engine> engine_;
    const flm_state_api* state_api_ = nullptr;
};

/// \brief a factory building FlmBackend<Engine>
/// \tparam Engine the concrete engine type
/// \param next_library the family's aie_next engine library, if it has one
/// \return a factory suitable for BackendRegistry::register_backend
template <class Engine>
BackendFactory flm_factory(const char* next_library = nullptr) {
    return [next_library](const BackendContext& context) -> std::unique_ptr<ModelBackend> {
        return std::make_unique<FlmBackend<Engine>>(context, next_library);
    };
}

}  // namespace flm::backend
