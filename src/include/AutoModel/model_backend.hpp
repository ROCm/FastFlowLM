/// \file model_backend.hpp
/// \brief Named execution backends for a model family
/// \note A backend *is* a piece of hardware. aie2p (Strix / Krackan) runs
///       FastFlowLM's own NPU kernels; aie4 runs ryzenai-corelib. Both
///       are causal_lm subclasses, but they differ in how they are built and in
///       how they must be driven. A ModelBackend owns one engine and states
///       those differences, so the frontends stay backend-agnostic. A backend id
///       is therefore a platform id (utils::platform_id), and a future non-NPU
///       target would join the same namespace rather than open a second axis.
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

/// \brief FastFlowLM's own NPU kernels, on Strix / Krackan Point
/// \note Matches utils::platform_id(npu_platform::aie2p). Every model family
///       registers this one; it is what a machine without aie4 silicon runs.
inline constexpr const char* kAie2pBackendId = "aie2p";

/// \brief ryzenai-corelib, on AIE4
/// \note Matches utils::platform_id(npu_platform::aie4). Only compiled in when
///       FLM_ENABLE_AIE4 is on, and only for families that have a corelib engine.
inline constexpr const char* kAie4BackendId = "aie4";

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
    /// \brief the frontend's parse of tokenizer_config.json
    /// \note Supplied here rather than read by the backend, so the model
    ///       directory layout stays the frontend's knowledge and the file is
    ///       opened once per load. null for families that have no such file.
    const nlohmann::json* tokenizer_config = nullptr;
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

    /// \brief the registered id of this backend, e.g. "aie2p"
    virtual std::string id() const = 0;

    /// \brief one line of provenance for `flm show`, empty when there is none
    virtual std::string detail() const { return {}; }

    /// \brief hard ceiling on decoded tokens, or 0 when only MAX_L applies
    virtual std::uint32_t max_decode_length() const { return 0; }

    /// \brief whether this backend can run with preemption enabled
    virtual bool supports_preemption() const { return true; }

    /// \brief whether the engine wants one more forward() after an EOS token
    /// \note The aie2p engines use it to keep their KV cache in step; the aie4
    ///       engines reject a decode past corelib's own limit, so they opt out.
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

/// \brief decide which backend to run a model on
/// \param family the model family, as in details.family
/// \param platform the detected hardware's id, e.g. utils::platform_id(...)
/// \param requested the --backend value, empty when the flag was not given
/// \param source if non-null, receives a human-readable reason for the choice
/// \return the resolved backend id
/// \throws std::runtime_error naming the registered ids when nothing matches
/// \note Precedence: --backend, then FLM_BACKEND, then the detected hardware.
///       The catalog no longer names a backend: model_list has already pruned
///       itself to the entries this hardware can run, so a per-entry backend
///       list could only restate `platform`.
/// \note `platform` arrives as a string rather than a utils::npu_platform so
///       that this header stays free of the NPU runtime includes, which is what
///       lets test/model_backend build without an XRT toolchain.
std::string resolve_backend_id(const std::string& family,
                               const std::string& platform,
                               const std::string& requested = "",
                               std::string* source = nullptr);

}  // namespace flm::backend
