/// \file model_backend.hpp
/// \brief Named execution backends for a model family
/// \note A backend names *where the kernels come from*, not which silicon runs
///       them: `flm` is FastFlowLM's own kernel flow, `rai` reaches them
///       through ryzenai-corelib. Both are causal_lm subclasses, but they
///       differ in how they are built and in how they must be driven. A
///       ModelBackend owns one engine and states those differences, so the
///       frontends stay backend-agnostic.
/// \note A backend id is *not* a platform id. Which generation the machine is
///       is utils::npu_platform's job, and one binary may carry several kernel
///       flows at once. Which flow runs a model is a property of the model:
///       the weights differ -- NPU2/Q4NX for flm, a Q8_0 GGUF for rai -- so a
///       model is packaged, tagged and pulled once per flow, and the tag says
///       which (kRaiFamilySuffix). The catalog's supported_platforms key
///       answers the other question, about the silicon.
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

/// \brief FastFlowLM's own NPU kernel flow
/// \note Every model family registers this one, in every build, which is what
///       makes it the backend a catalog entry gets when it names none.
inline constexpr const char* kFlmBackendId = "flm";

/// \brief kernels reached through ryzenai-corelib
/// \note Only compiled in when FLM_ENABLE_RAI is on, and only for families that
///       have a corelib engine. It is added to a build, never swapped in: a rai
///       build still runs every flm model with the flm kernels.
inline constexpr const char* kRaiBackendId = "rai";

/// \brief the entry key naming the kernel flow an entry's artifacts need
/// \note Written onto the entry by model_list, which derives it from the tag,
///       rather than read from model_list.json: the tag is what decides, and
///       it is not part of the entry that reaches AutoModel. An entry without
///       it runs on kFlmBackendId.
inline constexpr const char* kBackendKey = "backend";

/// \brief the suffix that marks a model family as ryzenai-corelib's
/// \note The tag is the mechanism: a corelib model is packaged differently
///       from its FastFlowLM namesake, so it is a different model to pull and
///       a different tag to ask for. model_list::backend_for_family is where
///       this is applied; it is spelled out here so the two ids and the rule
///       that picks between them sit together.
inline constexpr const char* kRaiFamilySuffix = "-rai";

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

    /// \brief the registered id of this backend, e.g. "flm"
    virtual std::string id() const = 0;

    /// \brief one line of provenance for `flm show`, empty when there is none
    virtual std::string detail() const { return {}; }

    /// \brief hard ceiling on decoded tokens, or 0 when only MAX_L applies
    virtual std::uint32_t max_decode_length() const { return 0; }

    /// \brief whether this backend can run with preemption enabled
    virtual bool supports_preemption() const { return true; }

    /// \brief whether the engine wants one more forward() after an EOS token
    /// \note The flm engines use it to keep their KV cache in step; the rai
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

    /// \brief every backend id this build registers for any family, sorted
    /// \note This is what the build links, which is what model_list prunes the
    ///       catalog against: an entry whose kernel flow is absent is not a
    ///       model this binary can run, exactly as an entry for the wrong NPU
    ///       generation is not.
    std::vector<std::string> backend_ids() const;

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
/// \param fallback the backend the entry asks for -- its kBackendKey, which
///        model_list derived from the tag, or kFlmBackendId when it has none
/// \param requested the --backend value, empty when the flag was not given
/// \param source if non-null, receives a human-readable reason for the choice
/// \return the resolved backend id
/// \throws std::runtime_error naming the registered ids when nothing matches
/// \note Precedence: --backend, then FLM_BACKEND, then `fallback`. Every one
///       of them is a demand, not a preference: an id that this family does
///       not register is an error whoever asked for it. The entry cannot
///       reasonably ask for a flow the build lacks, because model_list has
///       already pruned the entries whose flow is missing -- so a throw here
///       means a genuine mismatch, not a routine packaging gap.
/// \note `fallback` arrives as a string so that this header stays free of the
///       NPU runtime includes, which is what lets test/model_backend build
///       without an XRT toolchain.
std::string resolve_backend_id(const std::string& family,
                               const std::string& fallback,
                               const std::string& requested = "",
                               std::string* source = nullptr);

}  // namespace flm::backend
