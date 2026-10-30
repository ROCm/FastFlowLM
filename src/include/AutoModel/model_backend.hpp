/// \file model_backend.hpp
/// \brief Named execution backends for a model family
/// \note A backend names *where the kernels come from*, not which silicon runs
///       them: `flm` is FastFlowLM's own kernel flow, `core` reaches them
///       through ryzenai-corelib. Both are causal_lm subclasses, but they
///       differ in how they are built and in how they must be driven. A
///       ModelBackend owns one engine and states those differences, so the
///       frontends stay backend-agnostic.
/// \note A backend id is *not* a platform id. Which generation the machine is
///       is utils::npu_platform's job, and one binary may carry several kernel
///       flows at once. Which flow runs a model is a property of the model:
///       the weights differ -- NPU2/Q4NX for flm, a Q8_0 GGUF for core -- so a
///       model is packaged, tagged and pulled once per flow, and the tag says
///       which (kCoreFamilySuffix). The catalog's supported_platforms key
///       answers the other question, about the silicon.
/// \note This seam deliberately sits *above* causal_lm. The engine libraries in
///       src/lib/<runtime> are prebuilt against causal_lm.hpp, so that header is
///       a frozen ABI: adding or reordering a virtual there would silently shift
///       vtable slots in code this build cannot recompile.
#pragma once

#include "AutoModel/backend_contract.hpp"
#include "causal_lm.hpp"
#include "lm_config.hpp"
#include "device_runtime.hpp"
#include "nlohmann/json.hpp"

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

class npu_xclbin_manager;

namespace flm::backend {

/// \brief everything a backend factory needs to build its engine
/// \note Assembled by the frontend once the shared model state is initialized,
///       so `config` and `context_length` are already resolved.
struct BackendContext {
    std::string model_path;
    nlohmann::ordered_json model_info;
    /// \brief Provider that owns any opaque prefill payload.
    std::string frontend_provider = kBuiltinProviderId;
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

    /// \brief the registered id of this backend, e.g. "flm"
    virtual std::string id() const = 0;

    /// \brief one line of provenance for `flm show`, empty when there is none
    virtual std::string detail() const { return {}; }

    /// \brief hard ceiling on decoded tokens, or 0 when only MAX_L applies
    virtual std::uint32_t max_decode_length() const { return 0; }

    /// \brief whether this backend can run with preemption enabled
    virtual bool supports_preemption() const { return true; }

    /// \brief whether the engine implements checkpoint and restore
    virtual bool supports_state() const { return true; }

    /// \brief whether the engine wants one more forward() after an EOS token
    /// \note The flm engines use it to keep their KV cache in step; the core
    ///       engines reject a decode past corelib's own limit, so they opt out.
    virtual bool forwards_past_eos() const { return true; }

    /// \brief whether prefill chunks should end on multiples of the chunk
    ///        length counted from position 0, rather than from the turn's start
    /// \note corelib runs a pass that starts on a 4096-row page boundary as
    ///       whole pages on the NPU, and one starting inside a page partly on
    ///       the host, so its paged models want every chunk but a turn's first
    ///       to start on a boundary.
    virtual bool aligns_prefill_chunks() const { return false; }

    /// \brief whether the engine has failed in a way that needs a full reload
    virtual bool poisoned() const noexcept { return false; }

    /// \brief EOS ids proven by the backend's own package, when it has them
    /// \note Returning a value overrides whatever tokenizer_config.json says.
    virtual std::optional<std::vector<int>> forced_eos_ids() const {
        return std::nullopt;
    }

    /// \brief the BOS id proven by the backend's own package, when it has one
    /// \note Overrides tokenizer_config.json's bos_token_id, which some
    ///       packages (Gemma 4) omit. Whether the package has a BOS at all is
    ///       still the file's statement.
    virtual std::optional<int> forced_bos_id() const { return std::nullopt; }
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
    void register_backend(const std::string& family, const std::string& id,
                          BackendFactory factory, BackendTraits traits = {});

    /// \brief Register one provider implementation of a backend contract.
    /// \throws std::runtime_error if that family/backend/provider exists
    void register_provider(const std::string& family, const std::string& id,
                           std::string provider, BackendFactory factory,
                           BackendTraits traits = {},
                           std::string asset_root = {});

    /// \brief Remove an extension provider while rolling back registration.
    /// \note The linked-in "builtin" provider cannot be removed.
    void unregister_provider(const std::string& family,
                             const std::string& id,
                             const std::string& provider) noexcept;

    /// \brief register a backend, replacing any backend already under that id
    /// \note Unlike register_backend this never throws on a duplicate. It exists
    ///       for tests, which swap a real engine for a stub; production code
    ///       registers once, through register_builtin_backends.
    void replace_backend(std::string family, std::string id,
                         BackendFactory factory, BackendTraits traits = {});

    /// \brief the ids registered for a family, sorted
    std::vector<std::string> available(const std::string& family) const;

    /// \brief provider ids registered for one family/backend, sorted
    std::vector<std::string> available_providers(
        const std::string& family, const std::string& id) const;

    /// \brief what a registered backend needs before it can be built
    /// \throws std::runtime_error naming the available ids if it is not registered
    BackendTraits traits(const std::string& family,
                         const std::string& id,
                         const std::string& provider =
                             kBuiltinProviderId) const;

    /// \brief Extension-local asset prefix, empty for linked-in providers.
    std::string asset_root(const std::string& family, const std::string& id,
                           const std::string& provider) const;

    /// \brief whether a family has a backend with this id
    bool has(const std::string& family, const std::string& id) const;

    /// \brief whether one provider implements a family/backend pair
    bool has_provider(const std::string& family, const std::string& id,
                      const std::string& provider) const;

    /// \brief every backend id this build registers for any family, sorted
    /// \note This is what the build links, which is what model_list prunes the
    ///       catalog against: an entry whose kernel flow is absent is not a
    ///       model this binary can run, exactly as an entry for the wrong NPU
    ///       generation is not.
    std::vector<std::string> backend_ids() const;

    using ProviderInventory = BackendProviderInventory;

    /// \brief immutable family/backend/provider snapshot for catalog validation
    ProviderInventory provider_inventory() const;

    /// \brief build a backend
    /// \throws std::runtime_error naming the available ids if it is not registered
    std::unique_ptr<ModelBackend> create(const std::string& family,
                                         const std::string& id,
                                         const std::string& provider,
                                         const BackendContext& context) const;

    /// \brief Build the linked-in provider for compatibility with old callers.
    std::unique_ptr<ModelBackend> create(
        const std::string& family, const std::string& id,
        const BackendContext& context) const {
        return create(family, id, kBuiltinProviderId, context);
    }

private:
    struct Entry {
        BackendFactory factory;
        BackendTraits traits;
        std::string asset_root;
    };

    /// \brief look up one entry, or throw naming what the family does provide
    Entry lookup(const std::string& family, const std::string& id,
                 const std::string& provider) const;

    mutable std::mutex mutex_;
    std::map<std::string,
             std::map<std::string, std::map<std::string, Entry>>> factories_;
};

/// \brief register every backend that ships with this build
/// \param registry the registry to populate
/// \note Defined in builtin_backends.cpp, which is the one place that knows
///       both the family names and the engine types.
void register_builtin_backends(BackendRegistry& registry);

/// \brief Exact backend implementation selected for one model package.
struct BackendSelection {
    std::string backend;
    std::string provider;
};

/// \brief Resolve and validate the backend/provider pair for a model package.
/// \param family the model family, as in details.family
/// \param catalog_backend backend contract declared by the model entry
/// \param catalog_provider provider declared by the model entry
/// \param requested_backend the --backend value, empty when the flag was absent
/// \param source if non-null, receives a human-readable reason for the choice
/// \throws std::runtime_error if the exact pair is unavailable or an override
///         requests a backend the model package does not declare compatible
/// \note Precedence remains --backend, FLM_BACKEND, then the catalog. The
///       current catalog schema declares one exact pair; registration alone
///       does not make a different backend compatible with the package.
BackendSelection resolve_backend_selection(
    const std::string& family,
    const std::string& catalog_backend,
    const std::string& catalog_provider = kBuiltinProviderId,
    const std::string& requested_backend = "",
    std::string* source = nullptr);

}  // namespace flm::backend
