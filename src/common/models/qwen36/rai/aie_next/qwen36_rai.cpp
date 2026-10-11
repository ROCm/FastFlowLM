#include "models/qwen36/rai/aie_next/qwen36_rai.hpp"

#include "rai/corelib_object.hpp"
#include "rai/host_ops.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace flm::qwen36 {
namespace {

using flm::corelib::CorelibApi;
using flm::corelib::CorelibRuntime;
using flm::corelib::UniqueModel;
using flm::corelib::UniqueModelSnapshot;
using flm::corelib::UniqueModelWeights;

}  // namespace

class qwen36_rai::Impl {
public:
    Impl(std::filesystem::path gguf, std::shared_ptr<CorelibRuntime> runtime,
         std::uint32_t maximum)
        : runtime(std::move(runtime)),
          api(this->runtime ? this->runtime->api() : nullptr),
          max_length(maximum) {
        if (gguf.empty()) throw std::invalid_argument("Qwen3.6 GGUF path is empty");
        if (!this->runtime || !api) throw std::invalid_argument("corelib runtime is null");
        if (!maximum || maximum > kContextLimit)
            throw std::invalid_argument("Qwen3.6 maximum length must be in 1.." +
                                        std::to_string(kContextLimit));

        std::cout << "[FLM] Loading Qwen3.6-35B-A3B from Q8_0 GGUF" << std::endl;
        auto lease = this->runtime->AcquireExecution();
        const auto path = gguf.string();
        void* raw_weights = nullptr;
        api->Check(api->functions().model_weights_create_path("qwen36", path.c_str(), &raw_weights),
                   "ryzenai_corelib_model_weights_create_path");
        weights = UniqueModelWeights(api, raw_weights);
        Recreate(maximum);
        logits.resize(vocab);
    }

    buffer<bf16> Forward(int id) {
        usable();
        auto lease = runtime->AcquireExecution();
        Run(&id, 1, true);
        history.push_back(id);
        return Logits();
    }

    buffer<bf16> Prefill(const std::vector<int>& ids, void* payload) {
        usable();
        if (payload)
            throw std::invalid_argument("Qwen3.6 rai does not run a vision payload");
        if (ids.empty()) throw std::invalid_argument("Qwen3.6 request contains no token IDs");
        auto lease = runtime->AcquireExecution();
        Run(ids.data(), ids.size(), true);
        history.insert(history.end(), ids.begin(), ids.end());
        return Logits();
    }

    void Clear() {
        usable();
        auto lease = runtime->AcquireExecution();
        ResetModel();
        history.clear();
        saved.reset();
    }

    void Rewind(int to) {
        usable();
        if (to < 0 || static_cast<std::uint32_t>(to) > max_length)
            throw std::out_of_range("Qwen3.6 context length is out of range");
        if (static_cast<std::size_t>(to) > history.size())
            throw std::runtime_error("Qwen3.6 cannot rewind past the tokens it has run");
        if (to == position) return;
        auto lease = runtime->AcquireExecution();
        if (saved && saved->position == to && history.size() >= saved->history.size() &&
            std::equal(saved->history.begin(), saved->history.end(), history.begin())) {
            Load(saved->snapshot, to);
            history = saved->history;
            return;
        }
        history.resize(static_cast<std::size_t>(to));
        ResetModel();
        if (to > 0) Run(history.data(), history.size(), false);
    }

    int Checkpoint() {
        usable();
        if (Matches(saved)) return position;
        auto lease = runtime->AcquireExecution();
        saved = Saved{position, history, Save()};
        return position;
    }

    int Restore() {
        usable();
        if (!saved) return -1;
        if (Matches(saved)) return position;
        auto lease = runtime->AcquireExecution();
        Load(saved->snapshot, saved->position);
        history = saved->history;
        return position;
    }

    void Grow(std::uint32_t n) {
        if (n <= capacity) return;
        auto lease = runtime->AcquireExecution();
        UniqueModelSnapshot current;
        if (position > 0) current = Save();
        Recreate(n);
        if (current) Load(current, position);
    }

    void usable() const {
        if (poisoned) throw std::runtime_error("Qwen3.6 corelib engine is poisoned");
    }

    struct Saved {
        int position = 0;
        std::vector<int> history;
        UniqueModelSnapshot snapshot;
    };

    bool poisoned = false;

    std::shared_ptr<CorelibRuntime> runtime;
    std::shared_ptr<const CorelibApi> api;
    UniqueModelWeights weights;
    UniqueModel model;
    std::uint32_t capacity = 0;
    std::uint32_t max_length = 0;
    std::size_t vocab = 0;
    int position = 0;
    std::vector<int> history;
    std::optional<Saved> saved;
    std::vector<float> logits;

private:
    /// \brief a model over `weights` holding at least `n` positions
    /// \note The capacity is fixed for a model's life; `weights` outlive it, so
    ///       a larger one is a new model over the same packed weights.
    void Recreate(std::uint32_t n) {
        ryzenai_corelib_model_create_options options{};
        options.size = sizeof(options);
        options.max_sequence_length = static_cast<std::int64_t>(n);
        void* raw = nullptr;
        api->Check(api->functions().model_create("qwen36", weights.get(), &options, &raw),
                   "ryzenai_corelib_model_create");
        UniqueModel created(api, raw);
        ryzenai_corelib_model_info info{};
        api->Check(api->functions().model_info_get(created.get(), &info),
                   "ryzenai_corelib_model_info_get");
        if (info.vocab_size <= 0 || info.context_length < static_cast<std::int64_t>(n)) {
            throw std::runtime_error(
                "Qwen3.6 model_info is vocab " + std::to_string(info.vocab_size) +
                " context " + std::to_string(info.context_length) +
                "; this engine needs vocab > 0 and context >= " + std::to_string(n));
        }
        model = std::move(created);
        capacity = static_cast<std::uint32_t>(info.context_length);
        vocab = static_cast<std::size_t>(info.vocab_size);
    }

    void Run(const int* ids, std::size_t rows, bool want_logits) {
        if (static_cast<std::size_t>(position) + rows > max_length)
            throw std::out_of_range("Qwen3.6 request exceeds configured context capacity");
        const std::vector<std::int32_t> tokens(ids, ids + rows);
        ryzenai_corelib_model_forward_options options{};
        options.position = RYZENAI_CORELIB_POSITION_CONTINUE;
        if (want_logits) {
            options.logits = logits.data();
            options.logits_capacity = logits.size();
        }
        try {
            api->Check(api->functions().model_forward(
                           model.get(), tokens.data(), nullptr, rows, nullptr, &options),
                       "ryzenai_corelib_model_forward");
        } catch (...) {
            poisoned = true;
            position = 0;
            history.clear();
            saved.reset();
            throw;
        }
        position += static_cast<int>(rows);
    }

    UniqueModelSnapshot Save() {
        void* raw = nullptr;
        api->Check(api->functions().model_state_save(model.get(), &raw),
                   "ryzenai_corelib_model_state_save");
        return UniqueModelSnapshot(api, raw);
    }

    /// \note corelib leaves the model reset when a restore fails partway, so a
    ///       failure leaves nothing this engine's position or history describe.
    void Load(const UniqueModelSnapshot& snapshot, int at) {
        try {
            api->Check(api->functions().model_state_restore(model.get(), snapshot.get()),
                       "ryzenai_corelib_model_state_restore");
        } catch (...) {
            poisoned = true;
            position = 0;
            history.clear();
            saved.reset();
            throw;
        }
        position = at;
    }

    bool Matches(const std::optional<Saved>& snapshot) const {
        return snapshot && snapshot->position == position && snapshot->history == history;
    }

    buffer<bf16> Logits() const {
        buffer<bf16> out(vocab);
        rai::F32ToBf16(logits, std::span<std::uint16_t>(
                                   reinterpret_cast<std::uint16_t*>(out.data()), vocab));
        return out;
    }

    void ResetModel() {
        api->Check(api->functions().model_reset(model.get()), "ryzenai_corelib_model_reset");
        position = 0;
    }
};

qwen36_rai::qwen36_rai(std::filesystem::path gguf,
                       std::shared_ptr<corelib::CorelibRuntime> runtime,
                       std::uint32_t max_length)
    : impl_(std::make_unique<Impl>(std::move(gguf), std::move(runtime), max_length)) {}

qwen36_rai::~qwen36_rai() = default;

buffer<bf16> qwen36_rai::forward(int id) { return impl_->Forward(id); }

buffer<bf16> qwen36_rai::prefill(std::vector<int>& ids, void* payload) {
    return impl_->Prefill(ids, payload);
}

void qwen36_rai::set_context_length(int n) { impl_->Rewind(n); }

void qwen36_rai::load_weights(Q4NX&) {
    impl_->usable();
    throw std::runtime_error("Qwen3.6 rai weights are loaded only from GGUF");
}

void qwen36_rai::update_max_length(std::uint32_t n) {
    impl_->usable();
    if (!n || n > kContextLimit || n < static_cast<std::uint32_t>(impl_->position))
        throw std::out_of_range("Qwen3.6 maximum length is invalid");
    impl_->Grow(n);
    if (impl_->saved && n < static_cast<std::uint32_t>(impl_->saved->position))
        impl_->saved.reset();
    impl_->max_length = n;
}

void qwen36_rai::clear_context() { impl_->Clear(); }

buffer<bf16> qwen36_rai::get_k_cache(int, int) {
    throw std::runtime_error("Qwen3.6 rai does not expose the KV cache");
}

buffer<bf16> qwen36_rai::get_v_cache(int, int) {
    throw std::runtime_error("Qwen3.6 rai does not expose the KV cache");
}

int qwen36_rai::get_current_context_length() {
    impl_->usable();
    return impl_->position;
}

int qwen36_rai::checkpoint() { return impl_->Checkpoint(); }

int qwen36_rai::restore() { return impl_->Restore(); }

bool qwen36_rai::poisoned() const noexcept { return impl_ && impl_->poisoned; }

}  // namespace flm::qwen36
