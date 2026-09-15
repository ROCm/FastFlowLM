/// \file model_backend.hpp
/// \brief Named execution backends for a model family
/// \note A model family can have more than one engine behind it: Phi-4 runs
///       either on FastFlowLM's own NPU kernels or on ryzenai-corelib. Both are
///       causal_lm subclasses, but they differ in how they are built and in how
///       they must be driven. A ModelBackend owns one engine and states those
///       differences, so the frontends stay backend-agnostic.
/// \note This seam deliberately sits *above* causal_lm. The engine libraries in
///       src/lib/<runtime> are prebuilt against causal_lm.hpp, so that header is
///       a frozen ABI: adding or reordering a virtual there would silently shift
///       vtable slots in code this build cannot recompile.
#pragma once

#include "causal_lm.hpp"
#include "lm_config.hpp"
#include "device_runtime.hpp"
#include "nlohmann/json.hpp"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

class npu_xclbin_manager;

namespace flm::backend {

/// \brief the id every model family falls back on
/// \note This is FastFlowLM's own NPU engine path, i.e. what every catalog
///       entry without an explicit backend has always used.
inline constexpr const char* kDefaultBackendId = "flm_npu";

/// \brief everything a backend factory needs to build its engine
/// \note Assembled by the frontend once the shared model state is initialized,
///       so `config` and `context_length` are already resolved.
struct BackendContext {
    std::string model_path;
    nlohmann::ordered_json model_info;
    const LM_Config* config = nullptr;
    /// \note null for backends that do not drive the NPU through an xclbin
    npu_xclbin_manager* npu = nullptr;
    flm_rt::device* device = nullptr;
    std::uint32_t context_length = 0;
    bool enable_preemption = false;
};

/// \brief what a backend needs from the frontend *before* it can be built
/// \note These cannot live on ModelBackend: the frontend has to know them to
///       assemble the BackendContext in the first place. The defaults describe
///       the FastFlowLM NPU engines.
struct BackendTraits {
    /// \brief whether the frontend should build an npu_xclbin_manager for it
    bool needs_npu_xclbin = true;
    /// \brief whether this backend can run with preemption enabled
    bool supports_preemption = true;
    /// \brief the largest context length it accepts, or 0 when it has no ceiling
    std::uint32_t max_context_length = 0;
};

/// \brief one engine plus every rule for driving it
/// \note The defaults describe the FastFlowLM NPU engines, so a backend only
///       has to state what makes it different.
class ModelBackend {
public:
    virtual ~ModelBackend() = default;

    /// \brief the engine this backend owns
    virtual causal_lm& engine() = 0;

    /// \brief the registered id of this backend, e.g. "flm_npu"
    virtual std::string id() const = 0;

    /// \brief one line of provenance for `flm show`, empty when there is none
    virtual std::string detail() const { return {}; }

    /// \brief hard ceiling on decoded tokens, or 0 when only MAX_L applies
    virtual std::uint32_t max_decode_length() const { return 0; }

    /// \brief whether this backend can run with preemption enabled
    virtual bool supports_preemption() const { return true; }

    /// \brief whether the engine wants one more forward() after an EOS token
    /// \note The FastFlowLM engines use it to keep their KV cache in step;
    ///       corelib rejects a decode past its own limit, so it opts out.
    virtual bool forwards_past_eos() const { return true; }

    /// \brief whether the engine has failed in a way that needs a full reload
    virtual bool poisoned() const noexcept { return false; }

    /// \brief EOS ids proven by the backend's own package, when it has them
    /// \note Returning a value overrides whatever tokenizer_config.json says.
    virtual std::optional<std::vector<int>> forced_eos_ids() const {
        return std::nullopt;
    }
};

using BackendFactory =
    std::function<std::unique_ptr<ModelBackend>(const BackendContext&)>;

/// \brief the family -> (backend id -> factory) table
class BackendRegistry {
public:
    /// \brief the process-wide registry, populated with the built-in backends
    static BackendRegistry& instance();

    /// \brief register a backend for a family
    /// \throws std::runtime_error if that family already has that id
    void register_backend(std::string family, std::string id,
                          BackendFactory factory, BackendTraits traits = {});

    /// \brief register a backend, replacing any backend already under that id
    /// \note Unlike register_backend this never throws on a duplicate. It exists
    ///       for tests, which swap a real engine for a stub; production code
    ///       registers once, through register_builtin_backends.
    void replace_backend(std::string family, std::string id,
                         BackendFactory factory, BackendTraits traits = {});

    /// \brief the ids registered for a family, sorted
    std::vector<std::string> available(const std::string& family) const;

    /// \brief what a registered backend needs before it can be built
    /// \throws std::runtime_error naming the available ids if it is not registered
    BackendTraits traits(const std::string& family,
                         const std::string& id) const;

    /// \brief whether a family has a backend with this id
    bool has(const std::string& family, const std::string& id) const;

    /// \brief build a backend
    /// \throws std::runtime_error naming the available ids if it is not registered
    std::unique_ptr<ModelBackend> create(const std::string& family,
                                         const std::string& id,
                                         const BackendContext& context) const;

private:
    struct Entry {
        BackendFactory factory;
        BackendTraits traits;
    };

    /// \brief look up one entry, or throw naming what the family does provide
    Entry lookup(const std::string& family, const std::string& id) const;

    mutable std::mutex mutex_;
    std::map<std::string, std::map<std::string, Entry>> factories_;
};

/// \brief register every backend that ships with this build
/// \param registry the registry to populate
/// \note Defined in builtin_backends.cpp, which is the one place that knows
///       both the family names and the engine types.
void register_builtin_backends(BackendRegistry& registry);

/// \brief the backend ids a catalog entry allows
/// \param model_info the resolved model_list.json entry
/// \return the "supported_backends" array, or a single-element list holding
///         details.execution_backend (or kDefaultBackendId) when it is absent
/// \note The fallback is what keeps pre-existing catalogs meaning what they did.
std::vector<std::string> supported_backends(const nlohmann::ordered_json& model_info);

/// \brief decide which backend to run a model on
/// \param family the model family, as in details.family
/// \param model_info the resolved model_list.json entry
/// \param requested the --backend value, empty when the flag was not given
/// \param source if non-null, receives a human-readable reason for the choice
/// \return the resolved backend id
/// \throws std::runtime_error naming the allowed ids when the choice is invalid
/// \note Precedence: --backend, then FLM_BACKEND, then details.execution_backend,
///       then kDefaultBackendId.
std::string resolve_backend_id(const std::string& family,
                               const nlohmann::ordered_json& model_info,
                               const std::string& requested = "",
                               std::string* source = nullptr);

}  // namespace flm::backend
