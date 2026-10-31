/// \file corelib_model.cpp
#include "core/corelib_model.hpp"

#include "core/corelib_object.hpp"
#include "core/host_ops.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flm::core {

using corelib::CorelibApi;
using corelib::CorelibRuntime;
using corelib::UniqueModel;
using corelib::UniqueModelSession;
using corelib::UniqueModelSnapshot;

struct CorelibModel::Impl {
    Impl(ModelSpec model_spec, const std::filesystem::path& file,
         std::shared_ptr<CorelibRuntime> owner, std::uint32_t maximum)
        : spec(std::move(model_spec)),
          runtime(std::move(owner)),
          api(runtime ? runtime->api() : nullptr),
          max_length(maximum) {
        if (spec.rewind != Rewind::kSeek && spec.continuation != Continuation::kAnywhere)
            throw std::invalid_argument(spec.display_name +
                                        ": a model that cannot seek has to take a pass anywhere");
        if (file.empty()) throw std::invalid_argument(spec.display_name + " weights path is empty");
        if (!runtime || !api) throw std::invalid_argument("corelib runtime is null");
        if (!maximum || maximum > spec.context_limit)
            throw std::invalid_argument(spec.display_name + " maximum length must be in 1.." +
                                        std::to_string(spec.context_limit));

        std::cout << "[FLM] Loading " << spec.display_name << " from "
                  << file.filename().string() << std::endl;
        auto lease = runtime->AcquireExecution();
        const auto path = file.string();
        void* raw = nullptr;
        api->Check(api->functions().create_model_from_path(spec.corelib_name.c_str(),
                                                           path.c_str(), &raw),
                   "ryzenai_corelib_create_model_from_path");
        model = UniqueModel(api, raw);
        Recreate(maximum);
    }

    struct Saved {
        std::vector<int> history;
        /// \brief empty unless Rewind::kSnapshot
        UniqueModelSnapshot snapshot;
    };

    ModelSpec spec;
    std::shared_ptr<CorelibRuntime> runtime;
    std::shared_ptr<const CorelibApi> api;
    UniqueModel model;
    UniqueModelSession session;
    std::uint32_t capacity = 0;
    std::uint32_t max_length = 0;
    std::size_t vocab = 0;
    /// \brief every token the model holds, in order; its size is the position
    std::vector<int> history;
    std::optional<Saved> saved;
    std::vector<float> logits;
    bool poisoned = false;

    int Position() const { return static_cast<int>(history.size()); }

    void Usable() const {
        if (poisoned) throw std::runtime_error(spec.display_name + " corelib engine is poisoned");
    }

    buffer<bf16> Prefill(const std::vector<int>& ids, void* payload) {
        Usable();
        if (payload)
            throw std::invalid_argument(spec.display_name + " core does not run a vision payload");
        if (ids.empty())
            throw std::invalid_argument(spec.display_name + " request contains no token IDs");
        CheckIds(ids);
        auto lease = runtime->AcquireExecution();
        Advance(ids, true);
        return Logits();
    }

    buffer<bf16> Forward(int id) {
        Usable();
        CheckIds(std::span<const int>(&id, 1));
        auto lease = runtime->AcquireExecution();
        Advance(std::span<const int>(&id, 1), true);
        return Logits();
    }

    // Not named Rewind: inside Impl that would hide the enum of the same name.
    void RewindTo(int to) {
        Usable();
        if (to < 0 || static_cast<std::uint32_t>(to) > max_length)
            throw std::out_of_range(spec.display_name + " context length is out of range");
        if (to > Position())
            throw std::runtime_error(spec.display_name + " cannot rewind past the tokens it has run");
        if (to == Position()) return;
        auto lease = runtime->AcquireExecution();
        GoTo(std::vector<int>(history.begin(), history.begin() + to));
    }

    int Checkpoint() {
        Usable();
        if (saved && saved->history == history) return Position();
        auto lease = runtime->AcquireExecution();
        Saved next{history, {}};
        if (spec.rewind == Rewind::kSnapshot) next.snapshot = Save();
        saved = std::move(next);
        return Position();
    }

    int Restore() {
        Usable();
        if (!saved) return -1;
        if (saved->history == history) return Position();
        auto lease = runtime->AcquireExecution();
        GoTo(saved->history);
        return Position();
    }

    void Clear() {
        Usable();
        // A seekable model never reads its cache past the position, so nothing needs clearing.
        if (spec.rewind == Rewind::kSeek) {
            history.clear();
        } else {
            auto lease = runtime->AcquireExecution();
            ResetModel();
        }
        saved.reset();
    }

    void UpdateMaxLength(std::uint32_t n) {
        Usable();
        if (!n || n > spec.context_limit || n < static_cast<std::uint32_t>(Position()))
            throw std::out_of_range(spec.display_name + " maximum length is invalid");
        if (n > capacity) Grow(n);
        if (saved && n < saved->history.size()) saved.reset();
        max_length = n;
    }

private:
    void CheckIds(std::span<const int> ids) const {
        for (const int id : ids)
            if (id < 0 || static_cast<std::size_t>(id) >= vocab)
                throw std::out_of_range(spec.display_name + " token " + std::to_string(id) +
                                        " is outside the vocabulary of " + std::to_string(vocab));
    }

    /// \brief run `fresh` after every token the model holds, and keep it
    void Advance(std::span<const int> fresh, bool want_logits) {
        const auto at = history.size();
        if (at + fresh.size() > max_length)
            throw std::out_of_range(spec.display_name +
                                    " request exceeds configured context capacity");
        const auto from = PassStart(at, fresh.size());
        std::vector<std::int32_t> rows(history.begin() + static_cast<std::ptrdiff_t>(from),
                                       history.end());
        rows.insert(rows.end(), fresh.begin(), fresh.end());
        try {
            Dispatch(rows, from, want_logits);
        } catch (const corelib::CorelibError&) {
            RunAgainFromStart(fresh, want_logits);
            return;
        } catch (...) {
            Poison();
            throw;
        }
        history.insert(history.end(), fresh.begin(), fresh.end());
    }

    /// \brief after a failed pass, reset and run everything the model held plus
    ///        `fresh` again from position 0, once
    /// \note corelib fails only the call that failed, and some failures are
    ///       intermittent (PAGED_PREFILL_PREEMPTION_TIMEOUT in corelib's
    ///       docs/known-problems.md); a second failure poisons the engine.
    void RunAgainFromStart(std::span<const int> fresh, bool want_logits) {
        std::vector<std::int32_t> all(history.begin(), history.end());
        all.insert(all.end(), fresh.begin(), fresh.end());
        std::cout << "[FLM] " << spec.display_name << ": a corelib call failed; running the "
                  << all.size() << " tokens again from the start" << std::endl;
        try {
            api->Check(api->functions().model_reset(session.get()), "ryzenai_corelib_model_reset");
            history.clear();
            Dispatch(all, 0, want_logits);
        } catch (...) {
            Poison();
            throw;
        }
        history.assign(all.begin(), all.end());
    }

    /// \brief where a pass adding `rows` tokens at position `at` has to start
    std::size_t PassStart(std::size_t at, std::size_t rows) const {
        if (at == 0 || rows == 1) return at;
        switch (spec.continuation) {
        case Continuation::kAnywhere:
            return at;
        case Continuation::kFromStart:
            return 0;
        case Continuation::kFromPage:
            return capacity > kPageRows ? at - at % kPageRows : 0;
        }
        return 0;
    }

    void Dispatch(const std::vector<std::int32_t>& rows, std::size_t from, bool want_logits) {
        ryzenai_corelib_model_forward_options options{};
        options.size = sizeof(options);
        options.position =
            spec.rewind == Rewind::kSeek ? from : RYZENAI_CORELIB_POSITION_CONTINUE;
        float* out_logits = want_logits ? logits.data() : nullptr;
        const std::size_t logits_capacity = want_logits ? logits.size() : 0;
        api->Check(api->functions().model_forward(session.get(), rows.data(), nullptr, rows.size(),
                                                  nullptr, 0, &options, nullptr, out_logits,
                                                  logits_capacity),
                   "ryzenai_corelib_model_forward");
    }

    /// \brief make `target` the sequence the model holds, as cheaply as its state allows
    void GoTo(std::vector<int> target) {
        const bool prefix = target.size() <= history.size() &&
                            std::equal(target.begin(), target.end(), history.begin());
        if (spec.rewind == Rewind::kSeek && prefix) {
            history = std::move(target);
            return;
        }
        if (saved && saved->snapshot && saved->history == target) {
            Load(saved->snapshot);
            history = std::move(target);
            return;
        }
        ResetModel();
        if (!target.empty()) Advance(target, false);
    }

    /// \note The capacity is fixed for a session's life, so a larger one is a
    ///       new session of the same model.
    void Grow(std::uint32_t n) {
        auto lease = runtime->AcquireExecution();
        UniqueModelSnapshot current;
        if (spec.rewind == Rewind::kSnapshot && !history.empty()) current = Save();
        Recreate(n);
        if (current) {
            Load(current);
        } else if (!history.empty()) {
            auto replay = std::move(history);
            history.clear();
            Advance(replay, false);
        }
    }

    void Recreate(std::uint32_t n) {
        ryzenai_corelib_model_session_options options{};
        options.size = sizeof(options);
        options.max_sequence_length = static_cast<std::int64_t>(n);
        void* raw = nullptr;
        api->Check(api->functions().model_create_session(model.get(), &options, &raw),
                   "ryzenai_corelib_model_create_session");
        UniqueModelSession created(api, raw);
        ryzenai_corelib_model_info info{};
        api->Check(api->functions().model_get_info(created.get(), &info),
                   "ryzenai_corelib_model_get_info");
        if (info.vocab_size <= 0 || info.context_length < static_cast<std::int64_t>(n))
            throw std::runtime_error(spec.display_name + " model_info is vocab " +
                                     std::to_string(info.vocab_size) + " context " +
                                     std::to_string(info.context_length) +
                                     "; this engine needs vocab > 0 and context >= " +
                                     std::to_string(n));
        session = std::move(created);
        capacity = static_cast<std::uint32_t>(info.context_length);
        vocab = static_cast<std::size_t>(info.vocab_size);
        logits.resize(vocab);
    }

    UniqueModelSnapshot Save() {
        void* raw = nullptr;
        api->Check(api->functions().model_state_save(session.get(), &raw),
                   "ryzenai_corelib_model_state_save");
        return UniqueModelSnapshot(api, raw);
    }

    /// \note corelib leaves the session reset when a restore fails partway, which
    ///       no history here describes.
    void Load(const UniqueModelSnapshot& snapshot) {
        try {
            api->Check(api->functions().model_state_restore(session.get(), snapshot.get()),
                       "ryzenai_corelib_model_state_restore");
        } catch (...) {
            Poison();
            throw;
        }
    }

    void ResetModel() {
        try {
            api->Check(api->functions().model_reset(session.get()), "ryzenai_corelib_model_reset");
        } catch (...) {
            Poison();
            throw;
        }
        history.clear();
    }

    void Poison() {
        poisoned = true;
        history.clear();
        saved.reset();
    }

    buffer<bf16> Logits() const {
        buffer<bf16> out(vocab);
        F32ToBf16(logits,
                  std::span<std::uint16_t>(reinterpret_cast<std::uint16_t*>(out.data()), vocab));
        return out;
    }
};

CorelibModel::CorelibModel(ModelSpec spec, std::filesystem::path weights,
                           std::shared_ptr<corelib::CorelibRuntime> runtime,
                           std::uint32_t max_length)
    : impl_(std::make_unique<Impl>(std::move(spec), weights, std::move(runtime), max_length)) {}

CorelibModel::~CorelibModel() = default;

buffer<bf16> CorelibModel::forward(int id) { return impl_->Forward(id); }

buffer<bf16> CorelibModel::prefill(std::vector<int>& ids, void* payload) {
    return impl_->Prefill(ids, payload);
}

void CorelibModel::set_context_length(int length) { impl_->RewindTo(length); }

// An ABI shim, not a capability: load_weights is pure virtual in the frozen
// causal_lm.hpp, and corelib packs this engine's weights from the file itself.
void CorelibModel::load_weights(Q4NX&) {
    impl_->Usable();
    throw std::runtime_error(impl_->spec.display_name + " core weights are loaded by corelib");
}

void CorelibModel::update_max_length(std::uint32_t max_length) {
    impl_->UpdateMaxLength(max_length);
}

void CorelibModel::clear_context() { impl_->Clear(); }

buffer<bf16> CorelibModel::get_k_cache(int, int) {
    throw std::runtime_error(impl_->spec.display_name + " core does not expose the KV cache");
}

buffer<bf16> CorelibModel::get_v_cache(int, int) {
    throw std::runtime_error(impl_->spec.display_name + " core does not expose the KV cache");
}

int CorelibModel::get_current_context_length() {
    impl_->Usable();
    return impl_->Position();
}

int CorelibModel::checkpoint() { return impl_->Checkpoint(); }

int CorelibModel::restore() { return impl_->Restore(); }

bool CorelibModel::poisoned() const noexcept { return impl_ && impl_->poisoned; }

}  // namespace flm::core
